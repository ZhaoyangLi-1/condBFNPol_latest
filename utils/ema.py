"""Exponential Moving Average (EMA) wrapper for model parameters.

Adapted from the reference snippet provided by the user, with safe copying,
state (de)serialization, and batch-norm handling.
"""

from __future__ import annotations

import copy
import torch
from torch.nn.modules.batchnorm import _BatchNorm


class EMAModel:
    """Maintain an exponential moving average of model weights."""

    def __init__(
        self,
        model: torch.nn.Module,
        update_after_step: int = 0,
        inv_gamma: float = 1.0,
        power: float = 2 / 3,
        min_value: float = 0.0,
        max_value: float = 0.9999,
        clone_model: bool = True,
    ):
        """
        Args mirror the provided reference implementation.
        clone_model: when True, deep-copies the input model to keep EMA weights separate.
        """
        self.averaged_model = copy.deepcopy(model) if clone_model else model
        self.averaged_model.eval()
        self.averaged_model.requires_grad_(False)

        self.update_after_step = update_after_step
        self.inv_gamma = inv_gamma
        self.power = power
        self.min_value = min_value
        self.max_value = max_value

        self.decay = 0.0
        self.optimization_step = 0

    def get_decay(self, optimization_step: int) -> float:
        """Compute the decay factor for the exponential moving average."""
        step = max(0, optimization_step - self.update_after_step - 1)
        value = 1 - (1 + step / self.inv_gamma) ** -self.power
        if step <= 0:
            return 0.0
        return max(self.min_value, min(value, self.max_value))

    @torch.no_grad()
    def step(self, new_model: torch.nn.Module):
        """Update EMA weights from the new_model parameters."""
        self.decay = self.get_decay(self.optimization_step)

        for module, ema_module in zip(
            new_model.modules(), self.averaged_model.modules()
        ):
            for param, ema_param in zip(
                module.parameters(recurse=False),
                ema_module.parameters(recurse=False),
            ):
                if isinstance(module, _BatchNorm):
                    ema_param.copy_(param.to(dtype=ema_param.dtype).data)
                elif not param.requires_grad:
                    ema_param.copy_(param.to(dtype=ema_param.dtype).data)
                else:
                    ema_param.mul_(self.decay)
                    ema_param.add_(
                        param.data.to(dtype=ema_param.dtype), alpha=1 - self.decay
                    )

        self.optimization_step += 1

    def state_dict(self):
        return {
            "averaged_model": self.averaged_model.state_dict(),
            "optimization_step": self.optimization_step,
            "decay": self.decay,
            "update_after_step": self.update_after_step,
            "inv_gamma": self.inv_gamma,
            "power": self.power,
            "min_value": self.min_value,
            "max_value": self.max_value,
        }

    def load_state_dict(self, state_dict):
        self.averaged_model.load_state_dict(state_dict["averaged_model"])
        self.optimization_step = state_dict.get("optimization_step", 0)
        self.decay = state_dict.get("decay", 0.0)
        self.update_after_step = state_dict.get(
            "update_after_step", self.update_after_step
        )
        self.inv_gamma = state_dict.get("inv_gamma", self.inv_gamma)
        self.power = state_dict.get("power", self.power)
        self.min_value = state_dict.get("min_value", self.min_value)
        self.max_value = state_dict.get("max_value", self.max_value)
