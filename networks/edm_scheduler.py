"""EDM (Elucidating Diffusion Models) Noise Scheduler.

Implements the Karras et al. noise schedule and ODE solvers for EDM-style
diffusion and consistency distillation.

Adapted from: https://github.com/Aaditya-Prasad/consistency-policy
Reference: Karras et al. "Elucidating the Design Space of Diffusion-Based
           Generative Models" (NeurIPS 2022)
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor


__all__ = ["EDMScheduler", "CTMScheduler", "huber_loss"]


def append_dims(x: Tensor, target_dims: int) -> Tensor:
    """Appends dimensions to the end of a tensor until it has target_dims."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target is {target_dims}")
    return x[(...,) + (None,) * dims_to_append]


def reduce_dims(x: Tensor, target_dims: int) -> Tensor:
    """Reduces dimensions from the end of a tensor until it has target_dims."""
    dims_to_reduce = x.ndim - target_dims
    if dims_to_reduce < 0:
        raise ValueError(f"input has {x.ndim} dims but target is {target_dims}")
    for _ in range(dims_to_reduce):
        x = x.squeeze(-1)
    return x


def huber_loss(
    pred: Tensor,
    target: Tensor,
    delta: float = 0.0,
    weights: Optional[Tensor] = None,
) -> Tensor:
    """Pseudo-Huber loss function.

    Args:
        pred: Predicted tensor [B, T, D].
        target: Target tensor [B, T, D].
        delta: Boundary between L1 and L2 loss. 0 = MSE, -1 = auto-compute.
        weights: Optional per-sample weights [B].

    Returns:
        Scalar loss value.
    """
    if delta == -1:
        # iCT's recommended delta
        delta = math.sqrt(math.prod(pred.shape[1:])) * 0.00054

    mse = F.mse_loss(pred, target, reduction="none")
    loss = torch.sqrt(mse**2 + delta**2) - delta

    if weights is not None:
        loss = torch.einsum("b T D, b -> b T D", loss, weights)

    return loss.mean()


class EDMScheduler:
    """EDM noise scheduler with Karras sigma schedule.

    Implements the noise schedule and ODE solvers from the EDM paper.

    Args:
        time_min: Minimum sigma value (default: 0.002).
        time_max: Maximum sigma value (default: 80.0).
        rho: Schedule curvature parameter (default: 7.0).
        bins: Number of discretization steps (default: 80).
        solver: ODE solver type ("euler", "heun", "second_order").
        scaling: Output scaling type ("boundary", "no_boundary").
        data_std: Dataset standard deviation assumption (default: 0.5).
        P_mean: Log-normal sampling mean (default: -1.2).
        P_std: Log-normal sampling std (default: 1.2).
    """

    def __init__(
        self,
        time_min: float = 0.002,
        time_max: float = 80.0,
        rho: float = 7.0,
        bins: int = 80,
        solver: str = "heun",
        scaling: str = "boundary",
        data_std: float = 0.5,
        P_mean: float = -1.2,
        P_std: float = 1.2,
        time_sampler: str = "uniform",
        **kwargs,
    ):
        self.time_min = time_min
        self.time_max = time_max
        self.rho = rho
        self.bins = bins
        self.solver = solver
        self.scaling = scaling
        self.data_std = data_std
        self.P_mean = P_mean
        self.P_std = P_std
        self.time_sampler = time_sampler

    # ==================== CORE METHODS ====================

    def timesteps_to_times(self, timesteps: Tensor) -> Tensor:
        """Convert discrete bin indices to continuous sigma values."""
        t = self.time_max ** (1 / self.rho) + timesteps / (self.bins - 1) * (
            self.time_min ** (1 / self.rho) - self.time_max ** (1 / self.rho)
        )
        t = t**self.rho
        return t.clamp(self.time_min, self.time_max)

    def times_to_timesteps(self, times: Tensor) -> Tensor:
        """Convert continuous sigma values to discrete bin indices."""
        r = 1 / self.rho
        timesteps = (times**r - self.time_max**r) * (self.bins - 1) / (
            self.time_min**r - self.time_max**r
        )
        return torch.round(timesteps).long()

    def get_sigmas(self, device: torch.device) -> Tensor:
        """Get the full discretized sigma schedule."""
        timesteps = torch.arange(0, self.bins, device=device)
        return self.timesteps_to_times(timesteps)

    # ==================== NOISE OPERATIONS ====================

    def add_noise(self, trajectory: Tensor, times: Tensor) -> Tensor:
        """Add noise to trajectory at given sigma levels."""
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        return trajectory + self._trajectory_time_product(noise, times)

    def sample_initial_position(
        self,
        trajectory: Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        """Sample initial noisy trajectory.

        Note: Uses reduced variance (not multiplied by time_max) as per
        Consistency Policy paper trick.
        """
        return torch.randn(
            size=trajectory.shape,
            dtype=trajectory.dtype,
            device=trajectory.device,
            generator=generator,
        )

    # ==================== TIME SAMPLING ====================

    def sample_times(
        self,
        trajectory: Tensor,
        time_sampler: Optional[str] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Sample timesteps for training."""
        sampler = time_sampler or self.time_sampler
        batch = trajectory.shape[0]
        device = trajectory.device

        if sampler == "uniform":
            return self._uniform_sampler(batch, device)
        elif sampler == "log_normal":
            return self._log_normal_sampler(batch, device)
        elif sampler == "ctm_dsm":
            # CTM DSM sampler - uniform over sigma range for denoising score matching
            return self._ctm_dsm_sampler(batch, device)
        else:
            raise ValueError(f"Unknown sampler: {sampler}")

    def _uniform_sampler(self, batch: int, device: torch.device) -> Tuple[Tensor, Tensor]:
        """Uniform sampling over bins."""
        timesteps = torch.randint(0, self.bins - 1, (batch,), device=device).long()
        return self.timesteps_to_times(timesteps), self.timesteps_to_times(timesteps + 1)

    def _log_normal_sampler(self, batch: int, device: torch.device) -> Tuple[Tensor, Tensor]:
        """Log-normal sampling (biased towards beginning of diffusion)."""
        sigma = (
            torch.randn((batch,), device=device) * self.P_std + self.P_mean
        ).exp()
        # Clamp to time_min to ensure boundary scaling is valid
        sigma = sigma.clamp(min=self.time_min, max=self.time_max)
        return sigma, sigma

    def _ctm_dsm_sampler(self, batch: int, device: torch.device) -> Tuple[Tensor, Tensor]:
        """CTM DSM sampler - uniform sampling in log space for DSM loss."""
        # Sample uniformly in log space between time_min and time_max
        log_min = math.log(self.time_min)
        log_max = math.log(self.time_max)
        log_sigma = torch.rand((batch,), device=device) * (log_max - log_min) + log_min
        sigma = log_sigma.exp()
        return sigma, sigma

    # ==================== SCALINGS (EDM PARAMETERIZATION) ====================

    def get_scalings(self, time: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Get EDM scalings without boundary condition."""
        c_skip = self.data_std**2 / (time**2 + self.data_std**2)
        c_out = time * self.data_std / ((time**2 + self.data_std**2) ** 0.5)
        c_in = 1 / (time**2 + self.data_std**2) ** 0.5
        return c_skip, c_out, c_in

    def get_scalings_for_boundary_condition(
        self, time: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Get EDM scalings with boundary condition."""
        c_skip = self.data_std**2 / ((time - self.time_min) ** 2 + self.data_std**2)
        c_out = (
            (time - self.time_min)
            * self.data_std
            / (time**2 + self.data_std**2) ** 0.5
        )
        c_in = 1 / (time**2 + self.data_std**2) ** 0.5
        return c_skip, c_out, c_in

    # ==================== MODEL OUTPUT ====================

    def calc_out(
        self,
        model: Callable,
        trajectory: Tensor,
        times: Tensor,
        clamp: bool = False,
    ) -> Tensor:
        """Compute denoised output with EDM parameterization."""
        if self.scaling == "boundary":
            c_skip, c_out, c_in = [
                append_dims(c, trajectory.ndim)
                for c in self.get_scalings_for_boundary_condition(times)
            ]
        else:
            c_skip, c_out, c_in = [
                append_dims(c, trajectory.ndim) for c in self.get_scalings(times)
            ]

        if times.ndim > 1:
            times = reduce_dims(times, 1)

        # Rescale times for network input (EDM convention)
        rescaled_times = 1000 * 0.25 * torch.log(times + 1e-44)

        model_output = model(trajectory * c_in, rescaled_times)

        out = model_output * c_out + trajectory * c_skip
        if clamp:
            out = out.clamp(-1.0, 1.0)

        return out

    # ==================== LOSS WEIGHTING ====================

    def get_weights(
        self,
        times: Tensor,
        next_times: Optional[Tensor] = None,
        weighting: str = "karras",
    ) -> Optional[Tensor]:
        """Get loss weights for different weighting schemes."""
        if weighting == "none":
            return None
        elif weighting == "karras":
            # Karras weighting as in original EDM paper
            # Note: times should be clamped to time_min at sampling time
            weights = (times**2 + self.data_std**2) / ((times * self.data_std) ** 2)
            # Clamp weights to prevent gradient explosion for very small sigma values
            # At time_min=0.02, weight ≈ 2500 which is too high
            weights = weights.clamp(max=100.0)
            return weights
        elif weighting == "ict":
            if next_times is None:
                raise ValueError("ICT weighting requires next_times")
            # Add small epsilon to prevent division by zero
            return 1 / (times - next_times + 1e-6)
        else:
            raise ValueError(f"Unknown weighting: {weighting}")

    # ==================== ODE SOLVERS ====================

    def step(
        self,
        model: Callable,
        samples: Tensor,
        t: Tensor,
        next_t: Tensor,
        clamp: bool = False,
    ) -> Tensor:
        """Single ODE step from time t to next_t."""
        if self.solver in ("euler", "first_order"):
            return self._euler_solver(model, samples, t, next_t, clamp)
        elif self.solver in ("heun", "second_order"):
            return self._heun_solver(model, samples, t, next_t, clamp)
        else:
            raise ValueError(f"Unknown solver: {self.solver}")

    @torch.no_grad()
    def _euler_solver(
        self,
        model: Callable,
        samples: Tensor,
        t: Tensor,
        next_t: Tensor,
        clamp: bool = False,
    ) -> Tensor:
        """First-order Euler solver."""
        dims = samples.ndim
        step = append_dims(next_t - t, dims)

        denoised = self.calc_out(model, samples, t, clamp=clamp)
        dy = (samples - denoised) / append_dims(t, dims)

        return samples + step * dy

    @torch.no_grad()
    def _heun_solver(
        self,
        model: Callable,
        samples: Tensor,
        t: Tensor,
        next_t: Tensor,
        clamp: bool = False,
    ) -> Tensor:
        """Second-order Heun solver."""
        dims = samples.ndim
        step = append_dims(next_t - t, dims)

        denoised = self.calc_out(model, samples, t, clamp=clamp)
        dy = (samples - denoised) / append_dims(t, dims)

        y_next = samples + step * dy

        denoised_next = self.calc_out(model, y_next, next_t, clamp=clamp)
        dy_next = (y_next - denoised_next) / append_dims(next_t, dims)

        return samples + step * (dy + dy_next) / 2

    # ==================== HELPERS ====================

    @staticmethod
    def _trajectory_time_product(traj: Tensor, times: Tensor) -> Tensor:
        """Multiply trajectory by time (sigma) values."""
        return torch.einsum("b T D, b -> b T D", traj, times)


class CTMScheduler(EDMScheduler):
    """Consistency Training Model (CTM) scheduler.

    Extends EDMScheduler with additional methods for consistency distillation,
    including stop-time conditioning and CTM-specific loss computation.

    Args:
        ode_steps_max: Maximum ODE steps for teacher trajectory (default: 1).
        **kwargs: Arguments passed to EDMScheduler.
    """

    def __init__(self, ode_steps_max: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.ode_steps_max = ode_steps_max

    def sample_times_ctm(
        self, trajectory: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Sample t, s, u for CTM training.

        Returns:
            t: Start timestep (bins).
            s: Stop timestep (bins, >= t).
            u: Intermediate timestep (bins, between t and s).
        """
        batch = trajectory.shape[0]
        device = trajectory.device

        # t is uniform over bins
        t = torch.randint(0, self.bins, (batch,), device=device).long()

        # s is uniform over bins >= t
        s = torch.cat(
            [torch.randint(int(t_i.item()), self.bins + 1, (1,)) for t_i in t]
        ).to(device)

        # u is between t and s, clamped by ode_steps_max
        u = torch.cat(
            [
                torch.randint(int(t_i.item()), int((s_i + 1).item()), (1,))
                for t_i, s_i in zip(t, s)
            ]
        ).to(device)

        maxes = t + self.ode_steps_max
        mask = (u > maxes).float()
        u = u * (1 - mask) + maxes * mask

        return t, s, u.long()

    def ctm_calc_out(
        self,
        model: Callable,
        trajectory: Tensor,
        times: Tensor,
        stops: Tensor,
        clamp: bool = False,
    ) -> Tensor:
        """Compute CTM output with stop-time conditioning.

        The model predicts g_theta, which is then combined with the input
        to produce G_theta via: G = x * (s/t) + g * (1 - s/t)
        """
        if self.scaling == "boundary":
            c_skip, c_out, c_in = [
                append_dims(c, trajectory.ndim)
                for c in self.get_scalings_for_boundary_condition(times)
            ]
        else:
            c_skip, c_out, c_in = [
                append_dims(c, trajectory.ndim) for c in self.get_scalings(times)
            ]

        if times.ndim > 1:
            times = reduce_dims(times, 1)
        if stops.ndim > 1:
            stops = reduce_dims(stops, 1)

        # Rescale times for network input
        rescaled_times = (1000 * 0.25 * torch.log(times + 1e-44)).expand(
            trajectory.shape[0]
        )
        rescaled_stops = (1000 * 0.25 * torch.log(stops + 1e-44)).expand(
            trajectory.shape[0]
        )

        model_output = model(trajectory * c_in, rescaled_times, rescaled_stops)
        out = model_output * c_out + trajectory * c_skip  # g_theta

        # Combine to get G_theta
        ratio = (stops / times).unsqueeze(-1).unsqueeze(-1).expand(*out.shape)
        out = trajectory * ratio + out * (1 - ratio)

        if clamp:
            out = out.clamp(-1.0, 1.0)

        return out

    @torch.no_grad()
    def _heun_solver(
        self,
        model: Callable,
        samples: Tensor,
        t: Tensor,
        next_t: Tensor,
        clamp: bool = False,
    ) -> Tensor:
        """Heun solver that handles zero step size (for CTM)."""
        dims = samples.ndim
        step = append_dims(next_t - t, dims)
        mask = (step == 0).float()

        denoised = self.calc_out(model, samples, t, clamp=clamp)
        dy = (samples - denoised) / (append_dims(t, dims) + mask)

        y_next = samples + step * dy

        denoised_next = self.calc_out(model, y_next, next_t, clamp=clamp)
        dy_next = (y_next - denoised_next) / (append_dims(next_t, dims) + mask)

        y_next = samples + step * (dy + dy_next) / 2
        y_next = y_next * (1 - mask) + samples * mask

        return y_next
