"""Base Workspace for Training Policies.

This module provides a general training workspace that follows the same
structure as the original diffusion_policy workspace but is adapted to
work with any policy inheriting from BasePolicy.

It handles:
- Configuration management via OmegaConf/Hydra
- Checkpoint saving/loading with state preservation
- EMA model management
- Dataset and dataloader setup
- Training loop with validation and evaluation
- Logging (WandB, JSON, CSV)
"""

from __future__ import annotations

import copy
import os
import pathlib
import random
import threading
from typing import Any, Dict, Optional, Type

import dill
import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf

try:
    from hydra.core.hydra_config import HydraConfig
    HAS_HYDRA = True
except ImportError:
    HAS_HYDRA = False

__all__ = ["BaseWorkspace", "copy_to_cpu"]


def copy_to_cpu(x: Any) -> Any:
    """Recursively copy tensors to CPU."""
    if isinstance(x, torch.Tensor):
        return x.detach().to('cpu')
    elif isinstance(x, dict):
        return {k: copy_to_cpu(v) for k, v in x.items()}
    elif isinstance(x, list):
        return [copy_to_cpu(item) for item in x]
    else:
        return copy.deepcopy(x)


class BaseWorkspace:
    """Base workspace class for training policies.
    
    This class provides the foundation for training any policy that inherits
    from BasePolicy. It handles:
    - Configuration management
    - Checkpoint saving and loading
    - Snapshot saving and loading
    - State management for resumable training
    
    Subclasses should implement the `run()` method for their specific
    training loop.
    
    Attributes:
        include_keys: Tuple of attribute names to include in checkpoints
                     (will be pickled).
        exclude_keys: Tuple of attribute names to exclude from checkpoints.
    """
    
    include_keys: tuple = ('global_step', 'epoch')
    exclude_keys: tuple = ()
    
    def __init__(
        self,
        cfg: OmegaConf,
        output_dir: Optional[str] = None
    ):
        """Initialize the workspace.
        
        Args:
            cfg: Configuration object (OmegaConf)
            output_dir: Output directory for checkpoints and logs.
                       If None, uses Hydra's runtime output dir.
        """
        self.cfg = cfg
        self._output_dir = output_dir
        self._saving_thread: Optional[threading.Thread] = None
    
    @property
    def output_dir(self) -> str:
        """Get the output directory for this workspace."""
        if self._output_dir is not None:
            return self._output_dir
        
        if HAS_HYDRA:
            try:
                return HydraConfig.get().runtime.output_dir
            except Exception:
                pass
        
        # Fallback to current directory
        return os.getcwd()
    
    def run(self):
        """Main training loop. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement run()")
    
    # ==================== Checkpoint Management ====================
    
    def save_checkpoint(
        self,
        path: Optional[str] = None,
        tag: str = 'latest',
        exclude_keys: Optional[tuple] = None,
        include_keys: Optional[tuple] = None,
        use_thread: bool = True
    ) -> str:
        """Save a checkpoint of the workspace state.
        
        Args:
            path: Full path to save checkpoint. If None, uses default location.
            tag: Tag for the checkpoint (used in default filename).
            exclude_keys: Keys to exclude from state_dicts.
            include_keys: Additional keys to include (will be pickled).
            use_thread: If True, save in background thread.
            
        Returns:
            Absolute path to saved checkpoint.
        """
        if path is None:
            path = pathlib.Path(self.output_dir) / 'checkpoints' / f'{tag}.ckpt'
        else:
            path = pathlib.Path(path)
        
        if exclude_keys is None:
            exclude_keys = tuple(self.exclude_keys)
        if include_keys is None:
            include_keys = tuple(self.include_keys) + ('_output_dir',)
        
        # Create checkpoint directory
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Build payload
        payload = {
            'cfg': self.cfg,
            'state_dicts': {},
            'pickles': {}
        }
        
        for key, value in self.__dict__.items():
            if hasattr(value, 'state_dict') and hasattr(value, 'load_state_dict'):
                # Modules, optimizers, schedulers, etc.
                if key not in exclude_keys:
                    if use_thread:
                        payload['state_dicts'][key] = copy_to_cpu(value.state_dict())
                    else:
                        payload['state_dicts'][key] = value.state_dict()
            elif key in include_keys:
                payload['pickles'][key] = dill.dumps(value)
        
        # Save
        if use_thread:
            self._saving_thread = threading.Thread(
                target=lambda: torch.save(
                    payload,
                    path.open('wb'),
                    pickle_module=dill
                )
            )
            self._saving_thread.start()
        else:
            torch.save(payload, path.open('wb'), pickle_module=dill)
        
        return str(path.absolute())
    
    def get_checkpoint_path(self, tag: str = 'latest') -> pathlib.Path:
        """Get the default checkpoint path for a given tag."""
        return pathlib.Path(self.output_dir) / 'checkpoints' / f'{tag}.ckpt'
    
    def load_payload(
        self,
        payload: Dict[str, Any],
        exclude_keys: Optional[tuple] = None,
        include_keys: Optional[tuple] = None,
        **kwargs
    ):
        """Load state from a checkpoint payload.
        
        Args:
            payload: Checkpoint payload dictionary.
            exclude_keys: Keys to skip when loading state_dicts.
            include_keys: Keys to load from pickles.
            **kwargs: Additional arguments passed to load_state_dict.
        """
        if exclude_keys is None:
            exclude_keys = tuple()
        if include_keys is None:
            include_keys = payload['pickles'].keys()
        
        for key, value in payload['state_dicts'].items():
            if key not in exclude_keys:
                if key in self.__dict__:
                    self.__dict__[key].load_state_dict(value, **kwargs)
        
        for key in include_keys:
            if key in payload['pickles']:
                self.__dict__[key] = dill.loads(payload['pickles'][key])
    
    def load_checkpoint(
        self,
        path: Optional[str] = None,
        tag: str = 'latest',
        exclude_keys: Optional[tuple] = None,
        include_keys: Optional[tuple] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Load a checkpoint.
        
        Args:
            path: Path to checkpoint file. If None, uses default location.
            tag: Tag for default checkpoint location.
            exclude_keys: Keys to skip when loading.
            include_keys: Keys to include from pickles.
            **kwargs: Additional arguments passed to load_state_dict.
            
        Returns:
            The loaded payload dictionary.
        """
        if path is None:
            path = self.get_checkpoint_path(tag=tag)
        else:
            path = pathlib.Path(path)
        
        payload = torch.load(path.open('rb'), pickle_module=dill, **kwargs)
        self.load_payload(
            payload,
            exclude_keys=exclude_keys,
            include_keys=include_keys
        )
        return payload
    
    @classmethod
    def create_from_checkpoint(
        cls,
        path: str,
        exclude_keys: Optional[tuple] = None,
        include_keys: Optional[tuple] = None,
        **kwargs
    ) -> 'BaseWorkspace':
        """Create a workspace instance from a checkpoint.
        
        Args:
            path: Path to checkpoint file.
            exclude_keys: Keys to skip when loading.
            include_keys: Keys to include from pickles.
            **kwargs: Additional arguments passed to load_state_dict.
            
        Returns:
            New workspace instance with loaded state.
        """
        payload = torch.load(open(path, 'rb'), pickle_module=dill)
        instance = cls(payload['cfg'])
        instance.load_payload(
            payload=payload,
            exclude_keys=exclude_keys,
            include_keys=include_keys,
            **kwargs
        )
        return instance
    
    # ==================== Snapshot Management ====================
    
    def save_snapshot(self, tag: str = 'latest') -> str:
        """Save a complete snapshot of the workspace.
        
        Note: Snapshots save the entire Python object and require
        the code to be unchanged when loading. Use checkpoints for
        long-term storage.
        
        Args:
            tag: Tag for the snapshot filename.
            
        Returns:
            Absolute path to saved snapshot.
        """
        path = pathlib.Path(self.output_dir) / 'snapshots' / f'{tag}.pkl'
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self, path.open('wb'), pickle_module=dill)
        return str(path.absolute())
    
    @classmethod
    def create_from_snapshot(cls, path: str) -> 'BaseWorkspace':
        """Load a workspace from a snapshot.
        
        Args:
            path: Path to snapshot file.
            
        Returns:
            Loaded workspace instance.
        """
        return torch.load(open(path, 'rb'), pickle_module=dill)
