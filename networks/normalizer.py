"""Robust Data Normalizers for Robotics Policies.

This module provides implementations for:
1. LinearNormalizer (MinMax or Gaussian)
2. LogLinearNormalizer (Log transform + Linear)
3. CDFNormalizer (Empirical CDF -> Gaussian)
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Callable, Dict, Optional, Union, TypeVar

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Handle optional zarr dependency
try:
    import zarr

    ZarrArray = zarr.Array
except ImportError:
    ZarrArray = Any  # type: ignore

log = logging.getLogger(__name__)

# =============================================================================
# Utilities
# =============================================================================


def dict_apply(x: Dict[str, Any], func: Callable[[Any], Any]) -> Dict[str, Any]:
    """Apply a function to every value in a dictionary recursively."""
    result = dict()
    for key, value in x.items():
        if isinstance(value, dict):
            result[key] = dict_apply(value, func)
        else:
            result[key] = func(value)
    return result


class DictOfTensorMixin(nn.Module):
    """Mixin to handle dictionary of tensors/parameters in nn.Module."""

    def __init__(self, params_dict: Dict[str, Any] | None = None):
        super().__init__()
        if params_dict is None:
            params_dict = nn.ParameterDict()
        if not isinstance(params_dict, nn.ParameterDict):
            params_dict = nn.ParameterDict(params_dict)
        self.params_dict = params_dict

    def __getitem__(self, key: str) -> Any:
        return self.params_dict[key]

    def __setitem__(self, key: str, value: Any):
        self.params_dict[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self.params_dict

    def keys(self):
        return self.params_dict.keys()

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True):
        """Robust loading that pre-creates keys if missing."""
        # Infer keys from state_dict to support dynamic population
        top_level_keys = set()
        for k in state_dict.keys():
            if k.startswith("params_dict."):
                parts = k.split(".")
                if len(parts) >= 2:
                    top_level_keys.add(parts[1])

        # Subclass hook to create dummy params structure if needed
        self._create_dummy_params(top_level_keys, state_dict)

        super().load_state_dict(state_dict, strict=strict)

    def _create_dummy_params(self, keys, state_dict):
        pass  # Override in subclasses


# =============================================================================
# 1. Linear Normalizer (Standard)
# =============================================================================


def _fit_linear(
    data: Union[torch.Tensor, np.ndarray, ZarrArray],
    last_n_dims: int = 1,
    dtype: torch.dtype = torch.float32,
    mode: str = "limits",
    output_max: float = 1.0,
    output_min: float = -1.0,
    range_eps: float = 1e-4,
    fit_offset: bool = True,
) -> nn.ParameterDict:
    # ... (Implementation identical to previous _fit) ...
    # For brevity, I am reusing the robust logic from the previous file.
    # Assuming data is already tensor-ified.
    if ZarrArray is not Any and isinstance(data, ZarrArray):
        data = data[:]
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
    if dtype is not None:
        data = data.type(dtype)

    dim = 1
    if last_n_dims > 0:
        dim = np.prod(data.shape[-last_n_dims:])
    data_flat = data.reshape(-1, dim)

    input_min = data_flat.min(dim=0).values
    input_max = data_flat.max(dim=0).values
    input_mean = data_flat.mean(dim=0)
    input_std = data_flat.std(dim=0)

    if mode == "limits":
        if fit_offset:
            input_range = input_max - input_min
            ignore = input_range < range_eps
            input_range[ignore] = output_max - output_min
            scale = (output_max - output_min) / input_range
            offset = output_min - scale * input_min
            offset[ignore] = (output_max + output_min) / 2 - input_min[ignore] * scale[
                ignore
            ]
        else:
            output_abs = min(abs(output_min), abs(output_max))
            input_abs = torch.maximum(torch.abs(input_min), torch.abs(input_max))
            ignore = input_abs < range_eps
            input_abs[ignore] = output_abs
            scale = output_abs / input_abs
            offset = torch.zeros_like(input_mean)
    elif mode == "gaussian":
        ignore = input_std < range_eps
        scale = input_std.clone()
        scale[ignore] = 1.0
        scale = 1.0 / scale
        if fit_offset:
            offset = -input_mean * scale
        else:
            offset = torch.zeros_like(input_mean)

    return nn.ParameterDict(
        {
            "scale": nn.Parameter(scale, requires_grad=False),
            "offset": nn.Parameter(offset, requires_grad=False),
            "input_stats": nn.ParameterDict(
                {
                    "min": nn.Parameter(input_min, requires_grad=False),
                    "max": nn.Parameter(input_max, requires_grad=False),
                    "mean": nn.Parameter(input_mean, requires_grad=False),
                    "std": nn.Parameter(input_std, requires_grad=False),
                }
            ),
        }
    )


class LinearNormalizer(DictOfTensorMixin):
    def fit(self, data, last_n_dims=1, dtype=torch.float32, mode="limits", **kwargs):
        if isinstance(data, dict):
            for k, v in data.items():
                self.params_dict[k] = _fit_linear(v, last_n_dims, dtype, mode, **kwargs)
        else:
            self.params_dict["_default"] = _fit_linear(
                data, last_n_dims, dtype, mode, **kwargs
            )

    def normalize(self, x):
        return self._apply(x, forward=True)

    def unnormalize(self, x):
        return self._apply(x, forward=False)

    def _apply(self, x, forward=True):
        if isinstance(x, dict):
            out = {}
            for k, v in x.items():
                if k in self.params_dict:
                    out[k] = self._apply_single(v, self.params_dict[k], forward)
                else:
                    out[k] = v
            return out
        return self._apply_single(x, self.params_dict.get("_default"), forward)

    def _apply_single(self, x, params, forward):
        if params is None:
            return x
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        scale, offset = params["scale"], params["offset"]
        x = x.to(scale.device, scale.dtype)

        # Reshape handling
        orig_shape = x.shape
        x = x.reshape(-1, scale.shape[0])

        if forward:
            x = x * scale + offset
        else:
            x = (x - offset) / scale

        return x.reshape(orig_shape)

    def _create_dummy_params(self, keys, state_dict):
        for k in keys:
            if k not in self.params_dict:
                # Infer dimension from state dict to create dummy
                scale_key = f"params_dict.{k}.scale"
                if scale_key in state_dict:
                    dim = state_dict[scale_key].shape[0]
                    self.params_dict[k] = nn.ParameterDict(
                        {
                            "scale": nn.Parameter(torch.ones(dim)),
                            "offset": nn.Parameter(torch.zeros(dim)),
                            "input_stats": nn.ParameterDict({}),  # Dummy
                        }
                    )


# =============================================================================
# 2. Log-Linear Normalizer
# =============================================================================


class LogLinearNormalizer(LinearNormalizer):
    """Applies log(x + shift) transform, then Linear Normalization.

    Useful for distributions with heavy tails (e.g. audio, force sensors).
    Note: Assumes data > -shift.
    """

    def __init__(self, shift: float = 1e-6):
        super().__init__()
        self.shift = shift

    def fit(self, data, **kwargs):
        # Pre-process data with log
        if isinstance(data, dict):
            # FIX: Avoid redundant torch.tensor wrap
            log_data = {}
            for k, v in data.items():
                v_t = torch.as_tensor(v)
                log_data[k] = torch.log(v_t + self.shift)
        else:
            d_t = torch.as_tensor(data)
            log_data = torch.log(d_t + self.shift)

        super().fit(log_data, **kwargs)

    def normalize(self, x):
        # Apply Log -> Linear Normalize
        if isinstance(x, dict):
            x_log = {k: torch.log(v + self.shift) for k, v in x.items()}
        else:
            x_log = torch.log(x + self.shift)
        return super().normalize(x_log)

    def unnormalize(self, x):
        # Linear Unnormalize -> Exp
        x_un = super().unnormalize(x)
        if isinstance(x_un, dict):
            return {k: torch.exp(v) - self.shift for k, v in x_un.items()}
        return torch.exp(x_un) - self.shift


# =============================================================================
# 3. CDF Normalizer (Gaussianization)
# =============================================================================


def _fit_cdf(data, bins=1024):
    """Computes quantiles for CDF normalization."""
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
    # Flatten [N, D]
    data = data.reshape(-1, data.shape[-1])
    N, D = data.shape

    # We want 'bins' quantiles for each dimension
    # Sort data per dimension
    sorted_data, _ = torch.sort(data, dim=0)

    # Select indices for quantiles
    # If N < bins, we just use all data
    if N > bins:
        indices = torch.linspace(0, N - 1, bins, dtype=torch.long)
        quantiles = sorted_data[indices]  # [Bins, D]
    else:
        quantiles = sorted_data  # [N, D]

    # Make unique/strictly increasing to allow interpolation?
    # For simple torch.bucketize, sorted is enough.

    return nn.ParameterDict(
        {
            "quantiles": nn.Parameter(quantiles, requires_grad=False),
            "min": nn.Parameter(data.min(0).values, requires_grad=False),
            "max": nn.Parameter(data.max(0).values, requires_grad=False),
        }
    )


class CDFNormalizer(DictOfTensorMixin):
    """Normalizes data by mapping to Uniform[0,1] via empirical CDF, then to Normal(0,1)."""

    def fit(self, data, bins=1024):
        if isinstance(data, dict):
            for k, v in data.items():
                self.params_dict[k] = _fit_cdf(v, bins)
        else:
            self.params_dict["_default"] = _fit_cdf(data, bins)

    def _normalize_single(self, x, params):
        # 1. Map x -> u in [0, 1] (CDF)
        quantiles = params["quantiles"]  # [Bins, D]

        # Handle shapes
        orig_shape = x.shape
        D = quantiles.shape[1]
        x = x.reshape(-1, D)

        # For each dimension, find rank.
        # Torch doesn't have batched searchsorted easily for different dim values.
        # Loop is slow but robust for simple implementation.
        # Optimized: vmap or broadcased comparison?

        u_list = []
        for d in range(D):
            q = quantiles[:, d]
            # Find position
            idx = torch.searchsorted(q, x[:, d]).float()
            u = idx / (len(q) - 1)
            u_list.append(u)

        u = torch.stack(u_list, dim=-1)

        # Clamp to avoid infs in inverse CDF
        u = torch.clamp(u, 1e-6, 1 - 1e-6)

        # 2. Map u -> z (Inverse Gaussian CDF / Probit)
        z = torch.special.erfinv(2 * u - 1) * 1.41421356  # sqrt(2)

        return z.reshape(orig_shape)

    def _unnormalize_single(self, z, params):
        # 1. Map z -> u (Gaussian CDF)
        u = 0.5 * (1 + torch.erf(z / 1.41421356))

        # 2. Map u -> x (Inverse Empirical CDF / Quantile)
        quantiles = params["quantiles"]
        orig_shape = z.shape
        D = quantiles.shape[1]
        z = z.reshape(-1, D)

        x_list = []
        for d in range(D):
            q = quantiles[:, d]
            # Indices in q
            idx = u[:, d] * (len(q) - 1)

            # Linear Interpolation manually
            idx_low = idx.floor().long()
            idx_high = idx.ceil().long()
            weight = idx - idx_low.float()

            val_low = q[idx_low]
            val_high = q[idx_high]
            val = val_low * (1 - weight) + val_high * weight
            x_list.append(val)

        x = torch.stack(x_list, dim=-1)
        return x.reshape(orig_shape)

    def normalize(self, x):
        if isinstance(x, dict):
            return {
                k: self._normalize_single(v, self.params_dict[k])
                for k, v in x.items()
                if k in self.params_dict
            }
        return self._normalize_single(x, self.params_dict["_default"])

    def unnormalize(self, x):
        if isinstance(x, dict):
            return {
                k: self._unnormalize_single(v, self.params_dict[k])
                for k, v in x.items()
                if k in self.params_dict
            }
        return self._unnormalize_single(x, self.params_dict["_default"])

    def _create_dummy_params(self, keys, state_dict):
        for k in keys:
            if k not in self.params_dict:
                # Infer D
                q_key = f"params_dict.{k}.quantiles"
                if q_key in state_dict:
                    bins, dim = state_dict[q_key].shape
                    self.params_dict[k] = nn.ParameterDict(
                        {
                            "quantiles": nn.Parameter(torch.zeros(bins, dim)),
                            "min": nn.Parameter(torch.zeros(dim)),
                            "max": nn.Parameter(torch.zeros(dim)),
                        }
                    )


# =============================================================================
# Factory
# =============================================================================


def get_normalizer(type_str: str = "linear", **kwargs):
    if type_str == "linear":
        return LinearNormalizer()  # Default mode='limits'
    elif type_str == "gaussian":
        # Helper to pre-configure linear normalizer
        # (Fit args are passed at fit time, this is just the class)
        # To force mode='gaussian' at fit time, we might need a partial or config wrapper.
        # For simplicity, we return LinearNormalizer and user must specify mode in fit()
        # OR we assume this factory returns an uninitialized object.
        return LinearNormalizer()
    elif type_str == "log":
        return LogLinearNormalizer(**kwargs)
    elif type_str == "cdf":
        return CDFNormalizer()
    else:
        raise ValueError(f"Unknown normalizer type: {type_str}")
