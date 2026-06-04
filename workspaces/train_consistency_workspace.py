"""Training Workspace for Consistency Policy.

This workspace handles training of the ConsistencyImagePolicy by distilling
from a pretrained EDM teacher. It includes special handling for:
- CTM + DSM combined loss computation
- Internal EMA model updates
- Multi-loss logging

Adapted from train_bfn_workspace.py with Consistency Policy modifications.
"""

from __future__ import annotations

import copy
import os
import pathlib
import random
from typing import Any, Dict, Optional

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

from workspaces.base_workspace import BaseWorkspace

# Import from diffusion_policy
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger

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

__all__ = ["TrainConsistencyWorkspace"]

# Register OmegaConf resolver for eval
OmegaConf.register_new_resolver("eval", eval, replace=True)


class TrainConsistencyWorkspace(BaseWorkspace):
    """Workspace for training Consistency Policy via distillation.

    This workspace handles the two-loss training scheme (CTM + DSM) and
    the internal EMA model updates required for Consistency Policy.
    """

    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir: Optional[str] = None):
        """Initialize the workspace."""
        super().__init__(cfg, output_dir=output_dir)

        # Set random seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # Check teacher path
        if cfg.policy.teacher_path is None:
            raise ValueError(
                "teacher_path is required for Consistency Policy training. "
                "Set it via: policy.teacher_path=/path/to/edm_teacher.ckpt"
            )

        # Instantiate model
        if HAS_HYDRA:
            self.model = hydra.utils.instantiate(cfg.policy)
        else:
            raise ImportError("Hydra is required for config-based instantiation")

        # Optimizer (only for student model parameters, not teacher or EMA)
        trainable_params = list(self.model.model.parameters()) + \
                          list(self.model.obs_encoder.parameters())
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer,
            params=trainable_params
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
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        train_dataloader = DataLoader(dataset, **cfg.dataloader)

        # Get normalizer from dataset
        if hasattr(dataset, 'get_normalizer'):
            normalizer = dataset.get_normalizer()
        else:
            normalizer = self._build_normalizer(dataset, cfg)

        # Validation dataset
        if hasattr(dataset, 'get_validation_dataset'):
            val_dataset = dataset.get_validation_dataset()
            val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)
        else:
            val_dataloader = None

        # Set normalizer
        self.model.set_normalizer(normalizer)

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

        # ========= Environment Runner =========
        env_runner = None
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
                train_losses = {'total': [], 'ctm': [], 'dsm': []}

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

                        # Compute loss (returns dict of losses)
                        loss_dict = self.model.compute_loss(batch)

                        # Sum all losses
                        total_loss = sum(loss_dict.values())

                        # Skip step if loss is NaN to prevent model corruption
                        if torch.isnan(total_loss) or torch.isinf(total_loss):
                            self.optimizer.zero_grad()
                            continue

                        scaled_loss = total_loss / cfg.training.gradient_accumulate_every
                        scaled_loss.backward()

                        # Step optimizer with gradient clipping
                        if self.global_step % cfg.training.gradient_accumulate_every == 0:
                            # Check for NaN/Inf in gradients before stepping (consistency
                            # backward can produce NaN grads even when forward loss is finite)
                            grad_has_nan = False
                            for p in self.model.parameters():
                                if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                                    grad_has_nan = True
                                    break
                            if grad_has_nan:
                                self.optimizer.zero_grad()
                                continue

                            # Gradient clipping to prevent explosion
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()

                            # Update internal EMA model
                            self.model.ema_update()

                        # Logging
                        total_loss_cpu = total_loss.item()
                        tepoch.set_postfix(loss=total_loss_cpu, refresh=False)

                        train_losses['total'].append(total_loss_cpu)
                        if 'ctm' in loss_dict:
                            train_losses['ctm'].append(loss_dict['ctm'].item())
                        if 'dsm' in loss_dict:
                            train_losses['dsm'].append(loss_dict['dsm'].item())

                        step_log = {
                            'train_loss': total_loss_cpu,
                            'train_ctm_loss': loss_dict.get('ctm', torch.tensor(0)).item(),
                            'train_dsm_loss': loss_dict.get('dsm', torch.tensor(0)).item(),
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

                # End of epoch - compute average losses
                step_log['train_loss'] = np.mean(train_losses['total'])
                if train_losses['ctm']:
                    step_log['train_ctm_loss'] = np.mean(train_losses['ctm'])
                if train_losses['dsm']:
                    step_log['train_dsm_loss'] = np.mean(train_losses['dsm'])

                # ========= Evaluation =========
                self.model.eval()

                # Run rollout
                if env_runner is not None and (self.epoch % cfg.training.rollout_every) == 0:
                    runner_log = env_runner.run(self.model)
                    step_log.update(runner_log)

                # Run validation
                if val_dataloader is not None and (self.epoch % cfg.training.val_every) == 0:
                    with torch.no_grad():
                        val_losses = {'total': [], 'ctm': [], 'dsm': []}

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
                                loss_dict = self.model.compute_loss(batch)
                                total_loss = sum(loss_dict.values())

                                val_losses['total'].append(total_loss.item())
                                if 'ctm' in loss_dict:
                                    val_losses['ctm'].append(loss_dict['ctm'].item())
                                if 'dsm' in loss_dict:
                                    val_losses['dsm'].append(loss_dict['dsm'].item())

                                if (cfg.training.max_val_steps is not None
                                        and batch_idx >= (cfg.training.max_val_steps - 1)):
                                    break

                        if val_losses['total']:
                            step_log['val_loss'] = np.mean(val_losses['total'])
                            if val_losses['ctm']:
                                step_log['val_ctm_loss'] = np.mean(val_losses['ctm'])
                            if val_losses['dsm']:
                                step_log['val_dsm_loss'] = np.mean(val_losses['dsm'])

                # Sampling on training batch
                if (self.epoch % cfg.training.sample_every) == 0:
                    with torch.no_grad():
                        batch = dict_apply(
                            train_sampling_batch,
                            lambda x: x.to(device, non_blocking=True)
                        )
                        obs = batch['obs']
                        gt_action = batch['action']

                        # Get predicted actions
                        pred_action = self.model(obs)

                        # Handle shape matching
                        if pred_action.ndim == 3 and gt_action.ndim == 3:
                            n_action_steps = pred_action.shape[1]
                            n_obs_steps = getattr(self.model, 'n_obs_steps', 2)
                            start_idx = n_obs_steps - 1
                            end_idx = start_idx + n_action_steps
                            gt_action = gt_action[:, start_idx:end_idx, :]
                        elif pred_action.ndim == 3 and gt_action.ndim == 2:
                            pred_action = pred_action.reshape(pred_action.shape[0], -1)
                        elif pred_action.ndim == 2 and gt_action.ndim == 3:
                            gt_action = gt_action.reshape(gt_action.shape[0], -1)

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
                self.model.train()

                wandb_run.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1

    def _build_normalizer(self, dataset, cfg) -> LinearNormalizer:
        """Build normalizer from dataset when get_normalizer is not available."""
        normalizer = LinearNormalizer()

        # Sample a subset for fitting
        n_samples = min(len(dataset), 1000)
        indices = np.random.choice(len(dataset), n_samples, replace=False)

        obs_list = []
        action_list = []

        for idx in indices:
            item = dataset[idx]
            if isinstance(item, dict):
                if 'obs' in item:
                    obs_list.append(item['obs'])
                if 'action' in item:
                    action_list.append(item['action'])
            elif isinstance(item, (list, tuple)):
                obs_list.append(item[0])
                action_list.append(item[1])

        # Stack and fit
        fit_data = {}
        if obs_list:
            if isinstance(obs_list[0], dict):
                for key in obs_list[0].keys():
                    fit_data[key] = np.stack([o[key] for o in obs_list])
            else:
                fit_data['obs'] = np.stack(obs_list)

        if action_list:
            fit_data['action'] = np.stack(action_list)

        normalizer.fit(fit_data)
        return normalizer


def _main(cfg):
    """Entry point for Hydra."""
    workspace = TrainConsistencyWorkspace(cfg)
    workspace.run()


if HAS_HYDRA:
    @hydra.main(
        version_base=None,
        config_path=str(pathlib.Path(__file__).parent.parent / "config"),
        config_name="train_consistency_pusht"
    )
    def main(cfg):
        _main(cfg)
else:
    def main(cfg):
        _main(cfg)


if __name__ == "__main__":
    main()
