"""Unified Training Workspace.

A single workspace that handles all policy types (BFN, Diffusion, etc.) 
and all task types (PushT, Robomimic). The policy type is determined
entirely by the config, following Google research best practices.

Usage:
    python scripts/train.py --config-name=bfn_pusht
    python scripts/train.py --config-name=diffusion_lift
"""

from __future__ import annotations

import copy
import os
import pathlib
import random
from typing import Optional

import numpy as np
import torch
import tqdm
import wandb
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

try:
    import hydra
    HAS_HYDRA = True
except ImportError:
    HAS_HYDRA = False

from workspaces.base_workspace import BaseWorkspace, copy_to_cpu

# Core imports
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner

try:
    from diffusion_policy.model.diffusion.ema_model import EMAModel
except ImportError:
    from utils.ema import EMAModel

try:
    from diffusion_policy.model.common.lr_scheduler import get_scheduler
except ImportError:
    from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
    
    def get_scheduler(name, optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
        if name == 'cosine':
            return CosineAnnealingLR(optimizer, T_max=num_training_steps, last_epoch=last_epoch)
        elif name == 'linear':
            def lr_lambda(step):
                if step < num_warmup_steps:
                    return float(step) / float(max(1, num_warmup_steps))
                return max(0.0, float(num_training_steps - step) / float(max(1, num_training_steps - num_warmup_steps)))
            return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)
        raise ValueError(f"Unknown scheduler: {name}")

__all__ = ["TrainWorkspace"]

# Register eval resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)


class TrainWorkspace(BaseWorkspace):
    """Unified workspace for training any policy on any task.
    
    The policy and task are fully determined by Hydra config.
    No task-specific or policy-specific code in this class.
    """
    
    include_keys = ['global_step', 'epoch']
    
    def __init__(self, cfg: OmegaConf, output_dir: Optional[str] = None):
        super().__init__(cfg, output_dir=output_dir)
        
        # Seed everything
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        # Instantiate policy from config (works for BFN, Diffusion, etc.)
        if not HAS_HYDRA:
            raise ImportError("Hydra required")
        self.model = hydra.utils.instantiate(cfg.policy)
        
        # EMA (optional)
        self.ema_model = None
        if cfg.training.get('use_ema', False):
            self.ema_model = copy.deepcopy(self.model)
        
        # Optimizer
        self.optimizer = self.model.get_optimizer(
            learning_rate=cfg.optimizer.learning_rate,
            weight_decay=cfg.optimizer.get('weight_decay', 1e-6),
            betas=cfg.optimizer.get('betas', [0.9, 0.999]),
        )
        
        # Device
        self.device = torch.device(cfg.training.get('device', 'cuda'))
        
        # State
        self.global_step = 0
        self.epoch = 0
        
    def run(self):
        cfg = self.cfg
        
        # Build dataset
        dataset: BaseImageDataset = hydra.utils.instantiate(cfg.task.dataset)
        
        # Build normalizer 
        normalizer = self._build_normalizer(dataset, cfg)
        self.model.set_normalizer(normalizer)
        if self.ema_model is not None:
            self.ema_model.set_normalizer(normalizer)
            
        # DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=cfg.dataloader.batch_size,
            shuffle=True,
            num_workers=cfg.dataloader.num_workers,
            pin_memory=cfg.dataloader.get('pin_memory', True),
            persistent_workers=cfg.dataloader.get('persistent_workers', False),
        )
        
        # Scheduler
        lr_scheduler = get_scheduler(
            name=cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=len(dataloader) * cfg.training.num_epochs,
        )
        
        # EMA
        ema = None
        if self.ema_model is not None:
            ema = EMAModel(
                model=self.ema_model,
                power=cfg.training.get('ema_power', 0.75),
            )
        
        # Move to device
        self.model.to(self.device)
        if self.ema_model is not None:
            self.ema_model.to(self.device)
        optimizer_to(self.optimizer, self.device)
        
        # Env runner (optional, may be None)
        env_runner = None
        if cfg.task.get('env_runner') is not None:
            env_runner: BaseImageRunner = hydra.utils.instantiate(
                cfg.task.env_runner,
                output_dir=self.output_dir,
            )
        
        # Logging
        if cfg.logging.get('mode', 'online') != 'disabled':
            wandb.init(
                dir=str(self.output_dir),
                config=OmegaConf.to_container(cfg, resolve=True),
                **cfg.logging,
            )
        
        # Checkpoint manager
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.get('topk', {'k': 5}),
        )
        
        # Training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as json_logger:
            for epoch in range(cfg.training.num_epochs):
                self.epoch = epoch
                self.model.train()
                
                epoch_loss = 0.0
                with tqdm.tqdm(dataloader, desc=f'Epoch {epoch}', leave=False) as pbar:
                    for batch in pbar:
                        batch = dict_apply(batch, lambda x: x.to(self.device, non_blocking=True))
                        
                        # Forward pass
                        loss = self.model.compute_loss(batch)
                        
                        # Backward pass  
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                        lr_scheduler.step()
                        
                        # EMA update
                        if ema is not None:
                            ema.step(self.model)
                            
                        # Logging
                        epoch_loss += loss.item()
                        self.global_step += 1
                        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                        
                # Epoch logging
                epoch_loss /= len(dataloader)
                log_data = {
                    'epoch': epoch,
                    'global_step': self.global_step,
                    'train_loss': epoch_loss,
                    'lr': lr_scheduler.get_last_lr()[0],
                }
                
                # Evaluation (if env_runner available)
                if env_runner is not None and (epoch + 1) % cfg.training.get('eval_every', 10) == 0:
                    policy = self.ema_model if self.ema_model is not None else self.model
                    policy.eval()
                    
                    with torch.no_grad():
                        eval_result = env_runner.run(policy)
                    log_data.update(eval_result)
                    
                    # Save best checkpoint
                    metric = eval_result.get('test_mean_score', epoch_loss)
                    topk_manager.update(
                        {
                            'epoch': epoch,
                            'state_dict': copy_to_cpu(self.model.state_dict()),
                            'ema_state_dict': copy_to_cpu(self.ema_model.state_dict()) if self.ema_model else None,
                            'optimizer': copy_to_cpu(self.optimizer.state_dict()),
                            'cfg': OmegaConf.to_container(cfg, resolve=True),
                        },
                        metric,
                    )
                
                # Log
                wandb.log(log_data, step=self.global_step)
                json_logger.log(log_data)
                
                # Checkpoint
                if (epoch + 1) % cfg.training.get('checkpoint_every', 50) == 0:
                    self.save_checkpoint(tag=f'epoch_{epoch:04d}')
                    
        # Final save
        self.save_checkpoint(tag='final')
        wandb.finish()
        
    def _build_normalizer(self, dataset, cfg) -> LinearNormalizer:
        """Build normalizer from dataset statistics."""
        normalizer = LinearNormalizer()
        
        # Get stats from dataset
        data = {
            'action': dataset.get_all_actions() if hasattr(dataset, 'get_all_actions') else dataset[:]['action'],
        }
        normalizer.fit(data=data, last_n_dims=1, mode='limits', output_max=1.0)
        
        # Image observations don't need normalization (already [0,1])
        image_keys = [k for k in dataset[0]['obs'].keys() if 'image' in k.lower() or 'rgb' in k.lower()]
        for key in image_keys:
            normalizer[f'obs/{key}'] = LinearNormalizer.create_identity()
            
        return normalizer


# Hydra entrypoint
def _main(cfg):
    workspace = TrainWorkspace(cfg)
    workspace.run()


# For use with Hydra
if HAS_HYDRA:
    @hydra.main(version_base=None, config_path='../config')
    def main(cfg):
        _main(cfg)
        
    @hydra.main(version_base=None, config_path='config')  
    def main(cfg):
        _main(cfg)

