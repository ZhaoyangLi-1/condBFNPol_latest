"""Training Workspace for BFN Policy on Robomimic tasks.

This workspace handles training of the BFNUnetHybridImagePolicy on Robomimic
manipulation tasks (Lift, Can, Square, etc.). It follows the same structure 
as TrainDiffusionUnetHybridWorkspace for fair comparison.
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

# Import from diffusion_policy
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
    from torch.optim.lr_scheduler import LambdaLR
    
    def get_scheduler(name, optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
        """Simple scheduler factory."""
        if name == 'cosine':
            from torch.optim.lr_scheduler import CosineAnnealingLR
            return CosineAnnealingLR(optimizer, T_max=num_training_steps, last_epoch=last_epoch)
        elif name == 'linear':
            def lr_lambda(step):
                if step < num_warmup_steps:
                    return float(step) / float(max(1, num_warmup_steps))
                return max(0.0, float(num_training_steps - step) / float(max(1, num_training_steps - num_warmup_steps)))
            return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)
        else:
            raise ValueError(f"Unknown scheduler: {name}")

__all__ = ["TrainBFNRobomimicWorkspace"]

# Register OmegaConf resolver for eval
OmegaConf.register_new_resolver("eval", eval, replace=True)


class TrainBFNRobomimicWorkspace(BaseWorkspace):
    """Workspace for training BFN Policy on Robomimic tasks.
    
    This workspace manages the complete training pipeline for Robomimic
    manipulation tasks (Lift, Can, Square, Transport, Tool Hang).
    """
    
    include_keys = ['global_step', 'epoch']
    
    def __init__(self, cfg: OmegaConf, output_dir: Optional[str] = None):
        """Initialize the workspace.
        
        Args:
            cfg: Hydra configuration object.
            output_dir: Output directory for logs and checkpoints.
        """
        super().__init__(cfg, output_dir=output_dir)
        
        # Set random seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        # Instantiate model
        if HAS_HYDRA:
            self.model = hydra.utils.instantiate(cfg.policy)
        else:
            raise ImportError("Hydra is required for config-based instantiation")
        
        # EMA model
        self.ema_model = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)
        
        # Optimizer
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer,
            params=self.model.parameters()
        )
        
        # Training state
        self.global_step = 0
        self.epoch = 0
    
    def run(self):
        """Execute the training loop."""
        cfg = copy.deepcopy(self.cfg)
        
        # Resume training if checkpoint exists
        if cfg.training.resume:
            latest_ckpt_path = self.get_checkpoint_path()
            if latest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {latest_ckpt_path}")
                self.load_checkpoint(path=latest_ckpt_path)
        
        # ========= Dataset Setup =========
        # Instantiate dataset from task config (Robomimic format)
        dataset: BaseImageDataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset), \
            f"Expected BaseImageDataset, got {type(dataset)}"
        
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()
        
        # Validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)
        
        # Set normalizer
        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)
        
        # ========= LR Scheduler =========
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs
            ) // cfg.training.gradient_accumulate_every,
            last_epoch=self.global_step - 1
        )
        
        # ========= EMA =========
        ema = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(cfg.ema, model=self.ema_model)
        
        # ========= Environment Runner =========
        env_runner: BaseImageRunner = None
        if hasattr(cfg.task, 'env_runner') and cfg.task.env_runner is not None:
            env_runner = hydra.utils.instantiate(
                cfg.task.env_runner,
                output_dir=self.output_dir
            )
        
        # ========= Logging =========
        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **cfg.logging
        )
        wandb.config.update({"output_dir": self.output_dir})
        
        # ========= Checkpoint Manager =========
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )
        
        # ========= Device Setup =========
        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)
        
        # Save batch for sampling
        train_sampling_batch = None
        
        # Debug mode
        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1
        
        # ========= Training Loop =========
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                step_log = dict()
                
                # ========= Train for this epoch =========
                train_losses = list()
                with tqdm.tqdm(
                    train_dataloader,
                    desc=f"Training epoch {self.epoch}",
                    leave=False,
                    mininterval=cfg.training.tqdm_interval_sec
                ) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        # Device transfer
                        batch = dict_apply(
                            batch,
                            lambda x: x.to(device, non_blocking=True)
                        )
                        if train_sampling_batch is None:
                            train_sampling_batch = batch
                        
                        # Compute loss
                        raw_loss = self.model.compute_loss(batch)
                        loss = raw_loss / cfg.training.gradient_accumulate_every
                        loss.backward()
                        
                        # Step optimizer
                        if self.global_step % cfg.training.gradient_accumulate_every == 0:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()
                        
                        # Update EMA
                        if cfg.training.use_ema and ema is not None:
                            ema.step(self.model)
                        
                        # Logging
                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        
                        step_log = {
                            'train_loss': raw_loss_cpu,
                            'global_step': self.global_step,
                            'epoch': self.epoch,
                            'lr': lr_scheduler.get_last_lr()[0]
                        }
                        
                        is_last_batch = (batch_idx == (len(train_dataloader) - 1))
                        if not is_last_batch:
                            wandb_run.log(step_log, step=self.global_step)
                            json_logger.log(step_log)
                            self.global_step += 1
                        
                        if (cfg.training.max_train_steps is not None
                                and batch_idx >= (cfg.training.max_train_steps - 1)):
                            break
                
                # End of epoch - compute average train loss
                train_loss = np.mean(train_losses)
                step_log['train_loss'] = train_loss
                
                # ========= Evaluation =========
                policy = self.model
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()
                
                # Run rollout (environment evaluation)
                if env_runner is not None and (self.epoch % cfg.training.rollout_every) == 0:
                    runner_log = env_runner.run(policy)
                    step_log.update(runner_log)
                
                # Run validation
                if (self.epoch % cfg.training.val_every) == 0:
                    with torch.no_grad():
                        val_losses = list()
                        with tqdm.tqdm(
                            val_dataloader,
                            desc=f"Validation epoch {self.epoch}",
                            leave=False,
                            mininterval=cfg.training.tqdm_interval_sec
                        ) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                batch = dict_apply(
                                    batch,
                                    lambda x: x.to(device, non_blocking=True)
                                )
                                loss = self.model.compute_loss(batch)
                                val_losses.append(loss.item())
                                if (cfg.training.max_val_steps is not None
                                        and batch_idx >= (cfg.training.max_val_steps - 1)):
                                    break
                        
                        if len(val_losses) > 0:
                            val_loss = np.mean(val_losses)
                            step_log['val_loss'] = val_loss
                
                # BFN sampling on training batch
                if (self.epoch % cfg.training.sample_every) == 0:
                    with torch.no_grad():
                        batch = dict_apply(
                            train_sampling_batch,
                            lambda x: x.to(device, non_blocking=True)
                        )
                        obs = batch['obs']
                        gt_action = batch['action']
                        
                        # Get predicted actions
                        pred_action = policy.predict_action(obs)['action_pred']
                        
                        mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                        step_log['train_action_mse_error'] = mse.item()
                        
                        del batch, obs, gt_action, pred_action, mse
                
                # ========= Checkpointing =========
                if (self.epoch % cfg.training.checkpoint_every) == 0:
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()
                    
                    # Sanitize metric names for checkpoint manager
                    metric_dict = {
                        key.replace('/', '_'): value
                        for key, value in step_log.items()
                    }
                    
                    topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)
                    if topk_ckpt_path is not None:
                        self.save_checkpoint(path=topk_ckpt_path)
                
                # ========= End of epoch =========
                policy.train()
                
                wandb_run.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1


def _main(cfg):
    """Entry point for Hydra."""
    workspace = TrainBFNRobomimicWorkspace(cfg)
    workspace.run()


if HAS_HYDRA:
    @hydra.main(
        version_base=None,
        config_path=str(pathlib.Path(__file__).parent.parent / "config"),
        config_name="train_bfn_robomimic_image"
    )
    def main(cfg):
        _main(cfg)
else:
    def main(cfg):
        _main(cfg)


if __name__ == "__main__":
    main()

