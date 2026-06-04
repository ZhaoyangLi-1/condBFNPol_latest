from __future__ import annotations

import os
import sys
import time
import random
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import gymnasium as gym
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from omegaconf import DictConfig, OmegaConf
from torch.distributions import Categorical, Normal
from tqdm.auto import tqdm

try:
    import wandb
except ImportError:
    wandb = None

from environments.lunar_lander_extended import ExtendedHybridLunarLander


LOG_PROB_EPS = 1e-6


def set_global_seeds(base_seed: int, torch_deterministic: bool = False) -> None:
    random.seed(base_seed)
    np.random.seed(base_seed)
    torch.manual_seed(base_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(base_seed)
        torch.cuda.manual_seed_all(base_seed)

    if torch_deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception as exc:
            print(f"[warn] Could not enable full deterministic algorithms: {exc}")
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def sample_gaussian_seed(
    rng: np.random.Generator,
    mean: float,
    std: float,
    seed_min: int,
    seed_max: int,
) -> int:
    raw = int(round(rng.normal(mean, std)))
    raw = abs(raw)
    return int(np.clip(raw, seed_min, seed_max))


def maybe_init_wandb(cfg: DictConfig):
    if cfg.logging.mode == "disabled":
        return None
    if wandb is None:
        raise ImportError(
            "wandb is not installed, but logging.mode is not 'disabled'. "
            "Install wandb or set logging.mode=disabled."
        )

    run = wandb.init(
        project=cfg.logging.project,
        entity=cfg.logging.entity,
        mode=cfg.logging.mode,
        name=cfg.logging.name,
        group=cfg.logging.group,
        tags=list(cfg.logging.tags) if cfg.logging.tags is not None else None,
        notes=cfg.logging.notes,
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
    )
    wandb.config.update({"project_root": project_root}, allow_val_change=True)
    return run


def wandb_log(data: Dict[str, float], step: int) -> None:
    if wandb is not None and wandb.run is not None:
        wandb.log(data, step=step)


def build_eval_seeds(base_seed: int, n_episodes: int) -> List[int]:
    rng = np.random.default_rng(base_seed + 10_000)
    return [
        int(rng.integers(low=0, high=2_147_483_647, endpoint=False))
        for _ in range(n_episodes)
    ]


@dataclass(frozen=True)
class HybridActionLayout:
    num_discrete: int
    param_dims: Dict[int, int]
    branch_slices: Dict[int, Tuple[int, int]]
    total_param_dim: int
    max_param_dim: int
    action_names: Dict[int, str]

    @classmethod
    def from_env_spec(cls, action_spec: Dict) -> "HybridActionLayout":
        num_discrete = int(action_spec["num_discrete"])
        param_dims = {int(k): int(v) for k, v in action_spec["param_dims"].items()}
        action_names = {int(k): str(v) for k, v in action_spec["action_names"].items()}

        branch_slices: Dict[int, Tuple[int, int]] = {}
        offset = 0
        for k in range(num_discrete):
            dim = param_dims[k]
            branch_slices[k] = (offset, offset + dim)
            offset += dim

        return cls(
            num_discrete=num_discrete,
            param_dims=param_dims,
            branch_slices=branch_slices,
            total_param_dim=offset,
            max_param_dim=int(action_spec["max_param_dim"]),
            action_names=action_names,
        )

    def flat_to_env_action(self, discrete_action: int, flat_action: np.ndarray) -> Dict[str, np.ndarray]:
        discrete_action = int(discrete_action)
        flat_action = np.asarray(flat_action, dtype=np.float32)
        padded = np.zeros(self.max_param_dim, dtype=np.float32)
        start, end = self.branch_slices[discrete_action]
        if end > start:
            padded[: end - start] = flat_action[start:end]
        return {"k": discrete_action, "x_k": padded}

    def env_to_flat_action(self, discrete_action: int, x_k: np.ndarray) -> np.ndarray:
        discrete_action = int(discrete_action)
        x_k = np.asarray(x_k, dtype=np.float32)
        flat = np.zeros(self.total_param_dim, dtype=np.float32)
        start, end = self.branch_slices[discrete_action]
        if end > start:
            flat[start:end] = x_k[: end - start]
        return flat


class HybridSACLunarLanderWrapper(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        use_image_obs: bool = False,
        img_size: Tuple[int, int] = (84, 84),
        sub_steps: int = 1,
        seed_mean: float = 0.0,
        seed_std: float = 5000.0,
        seed_min: int = 0,
        seed_max: int = 2_147_483_647,
        rng_seed: int = 0,
    ):
        super().__init__()
        self._env = ExtendedHybridLunarLander(
            use_image_obs=use_image_obs,
            img_size=tuple(img_size),
        )
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        self.action_spec = self._env.get_action_spec()

        self.seed_mean = seed_mean
        self.seed_std = seed_std
        self.seed_min = seed_min
        self.seed_max = seed_max
        self._rng = np.random.default_rng(rng_seed)

    def reset(self, *, seed=None, options=None):
        if seed is None:
            seed = sample_gaussian_seed(
                self._rng,
                self.seed_mean,
                self.seed_std,
                self.seed_min,
                self.seed_max,
            )
        obs, info = self._env.reset(seed=int(seed), options=options)
        info = dict(info) if info is not None else {}
        info["reset_seed"] = int(seed)
        if isinstance(obs, np.ndarray):
            obs = obs.astype(np.float32)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)
        if isinstance(obs, np.ndarray):
            obs = obs.astype(np.float32)
        return obs, float(reward), bool(terminated), bool(truncated), info

    def render(self):
        return self._env.render()

    def close(self):
        return self._env.close()


def get_activation(name: str):
    name = str(name).lower()
    if name == "relu":
        return nn.ReLU
    if name == "tanh":
        return nn.Tanh
    if name == "leaky_relu":
        return lambda: nn.LeakyReLU(0.01)
    raise ValueError(f"Unsupported activation: {name}")


def init_linear_weights(layer: nn.Module, weights_init: str, bias_init: str) -> None:
    if not isinstance(layer, nn.Linear):
        return

    if weights_init == "xavier":
        nn.init.xavier_uniform_(layer.weight)
    elif weights_init == "orthogonal":
        nn.init.orthogonal_(layer.weight)
    elif weights_init == "kaiming":
        nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
    else:
        raise ValueError(f"Unsupported weights_init: {weights_init}")

    if bias_init == "zeros":
        nn.init.zeros_(layer.bias)
    elif bias_init == "uniform":
        bound = 1.0 / np.sqrt(layer.in_features)
        nn.init.uniform_(layer.bias, -bound, bound)
    else:
        raise ValueError(f"Unsupported bias_init: {bias_init}")


def build_mlp(
    input_dim: int,
    hidden_dims: List[int],
    activation_name: str,
    weights_init: str,
    bias_init: str,
) -> nn.Sequential:
    activation_cls = get_activation(activation_name)
    layers: List[nn.Module] = []
    prev_dim = input_dim
    for hidden_dim in hidden_dims:
        linear = nn.Linear(prev_dim, hidden_dim)
        init_linear_weights(linear, weights_init, bias_init)
        layers.append(linear)
        layers.append(activation_cls())
        prev_dim = hidden_dim
    return nn.Sequential(*layers)


class Policy(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_layout: HybridActionLayout,
        hidden_dims: List[int],
        activation: str,
        weights_init: str,
        bias_init: str,
        log_std_min: float,
        log_std_max: float,
    ):
        super().__init__()
        self.action_layout = action_layout
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)

        self.backbone = build_mlp(
            input_dim=obs_dim,
            hidden_dims=hidden_dims,
            activation_name=activation,
            weights_init=weights_init,
            bias_init=bias_init,
        )
        last_dim = hidden_dims[-1] if hidden_dims else obs_dim
        self.pi_d = nn.Linear(last_dim, action_layout.num_discrete)
        init_linear_weights(self.pi_d, weights_init, bias_init)

        self.branch_means = nn.ModuleList()
        self.branch_logstds = nn.ModuleList()
        for k in range(action_layout.num_discrete):
            dim = action_layout.param_dims[k]
            if dim > 0:
                mean_head = nn.Linear(last_dim, dim)
                logstd_head = nn.Linear(last_dim, dim)
                init_linear_weights(mean_head, weights_init, bias_init)
                init_linear_weights(logstd_head, weights_init, bias_init)
            else:
                mean_head = nn.Identity()
                logstd_head = nn.Identity()
            self.branch_means.append(mean_head)
            self.branch_logstds.append(logstd_head)

    def _branch_params(self, h: torch.Tensor, k: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        dim = self.action_layout.param_dims[k]
        if dim == 0:
            return None, None
        mean = torch.tanh(self.branch_means[k](h))
        log_std = torch.tanh(self.branch_logstds[k](h))
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1.0)
        return mean, log_std

    def forward(self, obs: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.backbone(obs)
        logits_d = self.pi_d(h)
        return {"h": h, "logits_d": logits_d}

    def sample_all_branches(self, obs: torch.Tensor, deterministic: bool = False):
        out = self.forward(obs)
        h = out["h"]
        logits_d = out["logits_d"]

        dist_d = Categorical(logits=logits_d)
        prob_d = dist_d.probs
        log_prob_d = torch.log(prob_d + LOG_PROB_EPS)

        batch_size = obs.shape[0]
        total_dim = self.action_layout.total_param_dim
        device = obs.device
        dtype = obs.dtype

        action_bank = torch.zeros(
            batch_size,
            self.action_layout.num_discrete,
            total_dim,
            device=device,
            dtype=dtype,
        )
        log_prob_c = torch.zeros(
            batch_size,
            self.action_layout.num_discrete,
            device=device,
            dtype=dtype,
        )
        deterministic_bank = torch.zeros_like(action_bank)

        for k in range(self.action_layout.num_discrete):
            start, end = self.action_layout.branch_slices[k]
            dim = end - start
            if dim == 0:
                continue
            mean, log_std = self._branch_params(h, k)
            std = log_std.exp()
            normal = Normal(mean, std)

            raw = mean if deterministic else normal.rsample()
            squashed = torch.tanh(raw)
            lp = normal.log_prob(raw) - torch.log(1.0 - squashed.pow(2) + LOG_PROB_EPS)

            action_bank[:, k, start:end] = squashed
            log_prob_c[:, k] = lp.sum(dim=-1)
            deterministic_bank[:, k, start:end] = torch.tanh(mean)

        return action_bank, log_prob_c, log_prob_d, prob_d, logits_d, deterministic_bank

    def act(self, obs: torch.Tensor, deterministic: bool = False):
        action_bank, _, _, prob_d, logits_d, deterministic_bank = self.sample_all_branches(
            obs,
            deterministic=deterministic,
        )
        if deterministic:
            action_d = torch.argmax(logits_d, dim=-1)
            source_bank = deterministic_bank
        else:
            dist_d = Categorical(logits=logits_d)
            action_d = dist_d.sample()
            source_bank = action_bank

        batch_idx = torch.arange(obs.shape[0], device=obs.device)
        chosen_action_c = source_bank[batch_idx, action_d]
        return chosen_action_c, action_d


class SoftQNetwork(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_layout: HybridActionLayout,
        hidden_dims: List[int],
        activation: str,
        weights_init: str,
        bias_init: str,
    ):
        super().__init__()
        self.action_layout = action_layout
        input_dim = obs_dim + action_layout.total_param_dim + action_layout.num_discrete
        self.backbone = build_mlp(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            activation_name=activation,
            weights_init=weights_init,
            bias_init=bias_init,
        )
        last_dim = hidden_dims[-1] if hidden_dims else input_dim
        self.q_head = nn.Linear(last_dim, 1)
        init_linear_weights(self.q_head, weights_init, bias_init)

    def forward(self, obs: torch.Tensor, action_c: torch.Tensor, action_d: torch.Tensor) -> torch.Tensor:
        if action_d.dtype in (torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8):
            action_d_onehot = F.one_hot(action_d.long(), num_classes=self.action_layout.num_discrete).float()
        else:
            action_d_onehot = action_d.float()
        x = torch.cat([obs, action_c, action_d_onehot], dim=-1)
        h = self.backbone(x)
        return self.q_head(h).squeeze(-1)

    def forward_all_actions(self, obs: torch.Tensor, action_bank: torch.Tensor) -> torch.Tensor:
        batch_size = obs.shape[0]
        num_discrete = self.action_layout.num_discrete
        obs_rep = obs.unsqueeze(1).expand(-1, num_discrete, -1)
        onehots = F.one_hot(
            torch.arange(num_discrete, device=obs.device),
            num_classes=num_discrete,
        ).float().unsqueeze(0).expand(batch_size, -1, -1)
        flat_obs = obs_rep.reshape(batch_size * num_discrete, -1)
        flat_action = action_bank.reshape(batch_size * num_discrete, -1)
        flat_d = onehots.reshape(batch_size * num_discrete, -1)
        q = self.forward(flat_obs, flat_action, flat_d)
        return q.view(batch_size, num_discrete)


class ReplayBuffer:
    def __init__(self, obs_dim: int, action_dim: int, capacity: int):
        self.capacity = int(capacity)
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.action_c = np.zeros((capacity, action_dim), dtype=np.float32)
        self.action_d = np.zeros((capacity,), dtype=np.int64)
        self.reward = np.zeros((capacity,), dtype=np.float32)
        self.done = np.zeros((capacity,), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add(
        self,
        obs: np.ndarray,
        action_c: np.ndarray,
        action_d: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        self.obs[self.ptr] = obs
        self.next_obs[self.ptr] = next_obs
        self.action_c[self.ptr] = action_c
        self.action_d[self.ptr] = int(action_d)
        self.reward[self.ptr] = float(reward)
        self.done[self.ptr] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        idxs = np.random.randint(0, self.size, size=batch_size)
        return {
            "obs": torch.as_tensor(self.obs[idxs], dtype=torch.float32, device=device),
            "next_obs": torch.as_tensor(self.next_obs[idxs], dtype=torch.float32, device=device),
            "action_c": torch.as_tensor(self.action_c[idxs], dtype=torch.float32, device=device),
            "action_d": torch.as_tensor(self.action_d[idxs], dtype=torch.long, device=device),
            "reward": torch.as_tensor(self.reward[idxs], dtype=torch.float32, device=device),
            "done": torch.as_tensor(self.done[idxs], dtype=torch.float32, device=device),
        }


def save_checkpoint(
    path: Path,
    *,
    policy: Policy,
    qf1: SoftQNetwork,
    qf2: SoftQNetwork,
    qf1_target: SoftQNetwork,
    qf2_target: SoftQNetwork,
    policy_optimizer: torch.optim.Optimizer,
    q_optimizer: torch.optim.Optimizer,
    cfg: DictConfig,
    action_layout: HybridActionLayout,
    global_step: int,
    episode_count: int,
    best_eval_reward: float,
    log_alpha: Optional[torch.Tensor] = None,
    log_alpha_d: Optional[torch.Tensor] = None,
    alpha_optimizer: Optional[torch.optim.Optimizer] = None,
    alpha_d_optimizer: Optional[torch.optim.Optimizer] = None,
) -> None:
    payload = {
        "policy": policy.state_dict(),
        "qf1": qf1.state_dict(),
        "qf2": qf2.state_dict(),
        "qf1_target": qf1_target.state_dict(),
        "qf2_target": qf2_target.state_dict(),
        "policy_optimizer": policy_optimizer.state_dict(),
        "q_optimizer": q_optimizer.state_dict(),
        "cfg": OmegaConf.to_container(cfg, resolve=True),
        "action_layout": {
            "num_discrete": action_layout.num_discrete,
            "param_dims": action_layout.param_dims,
            "branch_slices": action_layout.branch_slices,
            "total_param_dim": action_layout.total_param_dim,
            "max_param_dim": action_layout.max_param_dim,
            "action_names": action_layout.action_names,
        },
        "global_step": int(global_step),
        "episode_count": int(episode_count),
        "best_eval_reward": float(best_eval_reward),
    }
    if log_alpha is not None:
        payload["log_alpha"] = log_alpha.detach().cpu()
    if log_alpha_d is not None:
        payload["log_alpha_d"] = log_alpha_d.detach().cpu()
    if alpha_optimizer is not None:
        payload["alpha_optimizer"] = alpha_optimizer.state_dict()
    if alpha_d_optimizer is not None:
        payload["alpha_d_optimizer"] = alpha_d_optimizer.state_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def aggregate_metric_buffer(metric_buffer: Dict[str, List[float]]) -> Dict[str, float]:
    out = {}
    for key, values in metric_buffer.items():
        if not values:
            continue
        arr = np.asarray(values, dtype=np.float32)
        finite = arr[np.isfinite(arr)]
        if finite.size > 0:
            out[key] = float(finite.mean())
    return out


def evaluate_policy(
    policy: Policy,
    env: HybridSACLunarLanderWrapper,
    action_layout: HybridActionLayout,
    eval_seeds: List[int],
    device: torch.device,
) -> Dict[str, float]:
    policy.eval()
    episode_rewards = []
    episode_lengths = []
    action_hist: List[int] = []

    for seed in eval_seeds:
        obs, _ = env.reset(seed=int(seed))
        done = False
        ep_reward = 0.0
        ep_length = 0

        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action_c_t, action_d_t = policy.act(obs_t, deterministic=True)
            discrete_action = int(action_d_t.item())
            flat_action = action_c_t.cpu().numpy()[0]
            env_action = action_layout.flat_to_env_action(discrete_action, flat_action)
            obs, reward, terminated, truncated, _ = env.step(env_action)
            ep_reward += reward
            ep_length += 1
            action_hist.append(discrete_action)
            done = terminated or truncated

        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_length)

    policy.train()
    rewards = np.asarray(episode_rewards, dtype=np.float32)
    lengths = np.asarray(episode_lengths, dtype=np.float32)
    action_freq = np.bincount(np.asarray(action_hist, dtype=np.int64), minlength=action_layout.num_discrete).astype(np.float32)
    if action_freq.sum() > 0:
        action_freq /= action_freq.sum()

    metrics = {
        "eval/mean_reward": float(rewards.mean()),
        "eval/std_reward": float(rewards.std()),
        "eval/min_reward": float(rewards.min()),
        "eval/max_reward": float(rewards.max()),
        "eval/mean_length": float(lengths.mean()),
    }
    for k in range(action_layout.num_discrete):
        metrics[f"eval/action_{k}_freq"] = float(action_freq[k])
    return metrics


def train_hybrid_sac(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)
    set_global_seeds(
        base_seed=cfg.seed.base_seed,
        torch_deterministic=bool(cfg.seed.torch_deterministic),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(cfg.training.output_dir).expanduser().resolve()
    checkpoint_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output_dir / "resolved_config.yaml")

    print("=" * 80)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)
    print(f"Using device: {device}")
    print(f"Output dir: {output_dir}")

    env = HybridSACLunarLanderWrapper(
        use_image_obs=cfg.env.use_image_obs,
        img_size=tuple(cfg.env.img_size),
        sub_steps=cfg.env.sub_steps,
        seed_mean=cfg.env.seed_mean,
        seed_std=cfg.env.seed_std,
        seed_min=cfg.env.seed_min,
        seed_max=cfg.env.seed_max,
        rng_seed=cfg.seed.base_seed,
    )
    eval_env = HybridSACLunarLanderWrapper(
        use_image_obs=cfg.env.use_image_obs,
        img_size=tuple(cfg.env.img_size),
        sub_steps=cfg.env.sub_steps,
        seed_mean=cfg.env.seed_mean,
        seed_std=cfg.env.seed_std,
        seed_min=cfg.env.seed_min,
        seed_max=cfg.env.seed_max,
        rng_seed=cfg.seed.base_seed + 9999,
    )

    if not isinstance(env.observation_space, spaces.Box):
        raise ValueError("Hybrid SAC script currently supports only vector observations.")

    obs_dim = int(env.observation_space.shape[0])
    action_layout = HybridActionLayout.from_env_spec(env.action_spec)

    policy = Policy(
        obs_dim=obs_dim,
        action_layout=action_layout,
        hidden_dims=list(cfg.network.hidden_dims),
        activation=cfg.network.activation,
        weights_init=cfg.network.weights_init,
        bias_init=cfg.network.bias_init,
        log_std_min=cfg.sac.log_std_min,
        log_std_max=cfg.sac.log_std_max,
    ).to(device)
    qf1 = SoftQNetwork(
        obs_dim=obs_dim,
        action_layout=action_layout,
        hidden_dims=list(cfg.network.hidden_dims),
        activation=cfg.network.activation,
        weights_init=cfg.network.weights_init,
        bias_init=cfg.network.bias_init,
    ).to(device)
    qf2 = SoftQNetwork(
        obs_dim=obs_dim,
        action_layout=action_layout,
        hidden_dims=list(cfg.network.hidden_dims),
        activation=cfg.network.activation,
        weights_init=cfg.network.weights_init,
        bias_init=cfg.network.bias_init,
    ).to(device)
    qf1_target = SoftQNetwork(
        obs_dim=obs_dim,
        action_layout=action_layout,
        hidden_dims=list(cfg.network.hidden_dims),
        activation=cfg.network.activation,
        weights_init=cfg.network.weights_init,
        bias_init=cfg.network.bias_init,
    ).to(device)
    qf2_target = SoftQNetwork(
        obs_dim=obs_dim,
        action_layout=action_layout,
        hidden_dims=list(cfg.network.hidden_dims),
        activation=cfg.network.activation,
        weights_init=cfg.network.weights_init,
        bias_init=cfg.network.bias_init,
    ).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())

    q_optimizer = torch.optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=cfg.sac.q_lr)
    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.sac.policy_lr)
    loss_fn = nn.MSELoss()

    if cfg.sac.autotune:
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        log_alpha_d = torch.zeros(1, requires_grad=True, device=device)
        alpha_optimizer = torch.optim.Adam([log_alpha], lr=cfg.sac.alpha_lr)
        alpha_d_optimizer = torch.optim.Adam([log_alpha_d], lr=cfg.sac.alpha_lr)
    else:
        log_alpha = None
        log_alpha_d = None
        alpha_optimizer = None
        alpha_d_optimizer = None

    replay_buffer = ReplayBuffer(obs_dim=obs_dim, action_dim=action_layout.total_param_dim, capacity=cfg.sac.buffer_size)

    run = maybe_init_wandb(cfg)
    if run is not None:
        wandb.config.update({"output_dir": str(output_dir)}, allow_val_change=True)
        if cfg.logging.watch_model:
            wandb.watch(policy, log="all", log_freq=cfg.logging.watch_log_freq)
            wandb.watch(qf1, log="all", log_freq=cfg.logging.watch_log_freq)
            wandb.watch(qf2, log="all", log_freq=cfg.logging.watch_log_freq)

    obs, reset_info = env.reset(seed=cfg.seed.base_seed)
    global_step = 0
    episode_count = 0
    episode_reward = 0.0
    episode_length = 0
    best_eval_reward = -float("inf")
    recent_episode_rewards = deque(maxlen=100)
    recent_episode_lengths = deque(maxlen=100)
    recent_reset_seeds = deque(maxlen=100)
    if "reset_seed" in reset_info:
        recent_reset_seeds.append(int(reset_info["reset_seed"]))

    metric_buffer: Dict[str, List[float]] = defaultdict(list)
    recent_action_counts = np.zeros(action_layout.num_discrete, dtype=np.int64)
    recent_flat_actions: List[np.ndarray] = []
    t0 = time.time()
    eval_seeds = build_eval_seeds(cfg.seed.base_seed, cfg.training.eval_episodes)

    print(
        "Training Hybrid SAC | "
        f"obs_dim={obs_dim} | "
        f"discrete={action_layout.num_discrete} | "
        f"flat_action_dim={action_layout.total_param_dim} | "
        f"total_steps={cfg.sac.total_timesteps}"
    )

    progress = tqdm(
        total=cfg.sac.total_timesteps,
        disable=not cfg.tqdm.enable,
        leave=cfg.tqdm.leave,
        desc="Hybrid SAC",
    )

    while global_step < cfg.sac.total_timesteps:
        global_step += 1

        if global_step < cfg.sac.learning_starts:
            sampled_action = env.action_space.sample()
            discrete_action = int(sampled_action["k"])
            flat_action = action_layout.env_to_flat_action(discrete_action=discrete_action, x_k=sampled_action["x_k"])
            env_action = {"k": discrete_action, "x_k": np.asarray(sampled_action["x_k"], dtype=np.float32)}
        else:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action_c_t, action_d_t = policy.act(obs_t, deterministic=False)
            discrete_action = int(action_d_t.item())
            flat_action = action_c_t.detach().cpu().numpy()[0].astype(np.float32)
            env_action = action_layout.flat_to_env_action(discrete_action, flat_action)

        next_obs, reward, terminated, truncated, info = env.step(env_action)
        done = terminated or truncated
        replay_buffer.add(obs=obs, action_c=flat_action, action_d=discrete_action, reward=reward, next_obs=next_obs, done=done)

        recent_action_counts[discrete_action] += 1
        recent_flat_actions.append(flat_action.copy())
        episode_reward += reward
        episode_length += 1
        obs = next_obs

        if global_step >= cfg.sac.learning_starts and replay_buffer.size >= cfg.sac.batch_size:
            batch = replay_buffer.sample(cfg.sac.batch_size, device=device)

            with torch.no_grad():
                alpha = log_alpha.exp() if log_alpha is not None else torch.tensor(cfg.sac.alpha, device=device)
                alpha_d = log_alpha_d.exp() if log_alpha_d is not None else torch.tensor(cfg.sac.alpha, device=device)
                next_action_bank, next_log_pi_c, next_log_pi_d, next_prob_d, _, _ = policy.sample_all_branches(
                    batch["next_obs"],
                    deterministic=False,
                )
                qf1_next_target = qf1_target.forward_all_actions(batch["next_obs"], next_action_bank)
                qf2_next_target = qf2_target.forward_all_actions(batch["next_obs"], next_action_bank)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
                next_v = (next_prob_d * (min_qf_next_target - alpha * next_log_pi_c - alpha_d * next_log_pi_d)).sum(dim=1)
                next_q_value = batch["reward"] + (1.0 - batch["done"]) * cfg.sac.gamma * next_v
                next_q_value = next_q_value.detach()

            qf1_a_values = qf1(batch["obs"], batch["action_c"], batch["action_d"])
            qf2_a_values = qf2(batch["obs"], batch["action_c"], batch["action_d"])
            qf1_loss = loss_fn(qf1_a_values, next_q_value)
            qf2_loss = loss_fn(qf2_a_values, next_q_value)
            qf_loss = 0.5 * (qf1_loss + qf2_loss)

            q_optimizer.zero_grad(set_to_none=True)
            qf_loss.backward()
            if cfg.sac.max_grad_norm is not None:
                q_grad_norm = nn.utils.clip_grad_norm_(list(qf1.parameters()) + list(qf2.parameters()), float(cfg.sac.max_grad_norm))
            else:
                q_grad_norm = torch.nan
            q_optimizer.step()

            metric_buffer["loss/qf1_loss"].append(float(qf1_loss.item()))
            metric_buffer["loss/qf2_loss"].append(float(qf2_loss.item()))
            metric_buffer["loss/qf_loss"].append(float(qf_loss.item()))
            metric_buffer["debug/next_q"].append(float(next_q_value.mean().item()))
            metric_buffer["debug/mean_reward_batch"].append(float(batch["reward"].mean().item()))
            metric_buffer["debug/q_grad_norm"].append(float(q_grad_norm))
            metric_buffer["loss/alpha"].append(float(alpha.item()))
            metric_buffer["loss/alpha_d"].append(float(alpha_d.item()))

            if global_step % cfg.sac.policy_frequency == 0:
                alpha = log_alpha.exp() if log_alpha is not None else torch.tensor(cfg.sac.alpha, device=device)
                alpha_d = log_alpha_d.exp() if log_alpha_d is not None else torch.tensor(cfg.sac.alpha, device=device)

                action_bank, log_pi_c, log_pi_d, prob_d, _, _ = policy.sample_all_branches(batch["obs"], deterministic=False)
                qf1_pi = qf1.forward_all_actions(batch["obs"], action_bank)
                qf2_pi = qf2.forward_all_actions(batch["obs"], action_bank)
                min_qf_pi = torch.min(qf1_pi, qf2_pi)

                policy_loss = (prob_d * (alpha * log_pi_c + alpha_d * log_pi_d - min_qf_pi)).sum(dim=1).mean()
                policy_loss_d = (prob_d * (alpha_d * log_pi_d - min_qf_pi)).sum(dim=1).mean()
                policy_loss_c = (prob_d * (alpha * log_pi_c - min_qf_pi)).sum(dim=1).mean()

                policy_optimizer.zero_grad(set_to_none=True)
                policy_loss.backward()
                if cfg.sac.max_grad_norm is not None:
                    policy_grad_norm = nn.utils.clip_grad_norm_(policy.parameters(), float(cfg.sac.max_grad_norm))
                else:
                    policy_grad_norm = torch.nan
                policy_optimizer.step()

                metric_buffer["loss/policy_loss"].append(float(policy_loss.item()))
                metric_buffer["debug/policy_loss_d"].append(float(policy_loss_d.item()))
                metric_buffer["debug/policy_loss_c"].append(float(policy_loss_c.item()))
                metric_buffer["debug/mean_q"].append(float(min_qf_pi.mean().item()))
                metric_buffer["debug/policy_ent_d"].append(float((-(prob_d * log_pi_d).sum(dim=1).mean()).item()))
                metric_buffer["debug/policy_ent_c"].append(float((-(prob_d * log_pi_c).sum(dim=1).mean()).item()))
                metric_buffer["debug/policy_grad_norm"].append(float(policy_grad_norm))

                if log_alpha is not None and alpha_optimizer is not None and log_alpha_d is not None and alpha_d_optimizer is not None:
                    with torch.no_grad():
                        _, alpha_log_pi_c, alpha_log_pi_d, alpha_prob_d, _, _ = policy.sample_all_branches(
                            batch["obs"], deterministic=False
                        )

                    alpha_loss = (-log_alpha * (alpha_prob_d * (alpha_log_pi_c + cfg.sac.ent_c)).sum(dim=1)).mean()
                    alpha_d_loss = (-log_alpha_d * (alpha_prob_d * (alpha_log_pi_d + cfg.sac.ent_d)).sum(dim=1)).mean()

                    alpha_optimizer.zero_grad(set_to_none=True)
                    alpha_loss.backward()
                    alpha_optimizer.step()

                    alpha_d_optimizer.zero_grad(set_to_none=True)
                    alpha_d_loss.backward()
                    alpha_d_optimizer.step()

                    metric_buffer["loss/alpha_loss"].append(float(alpha_loss.item()))
                    metric_buffer["loss/alpha_d_loss"].append(float(alpha_d_loss.item()))

            if global_step % cfg.sac.target_network_frequency == 0:
                with torch.no_grad():
                    for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                        target_param.data.copy_(cfg.sac.tau * param.data + (1.0 - cfg.sac.tau) * target_param.data)
                    for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                        target_param.data.copy_(cfg.sac.tau * param.data + (1.0 - cfg.sac.tau) * target_param.data)

        if done:
            recent_episode_rewards.append(float(episode_reward))
            recent_episode_lengths.append(float(episode_length))
            episode_count += 1
            episode_metrics = {
                "episode/reward": float(episode_reward),
                "episode/length": float(episode_length),
            }
            obs, reset_info = env.reset()
            if "reset_seed" in reset_info:
                recent_reset_seeds.append(int(reset_info["reset_seed"]))
                episode_metrics["episode/next_reset_seed"] = float(reset_info["reset_seed"])
            wandb_log(episode_metrics, step=global_step)
            episode_reward = 0.0
            episode_length = 0

        if global_step % cfg.training.log_interval == 0 or global_step == cfg.sac.total_timesteps:
            elapsed = time.time() - t0
            fps = global_step / max(elapsed, 1e-6)
            action_freq = recent_action_counts.astype(np.float32)
            if action_freq.sum() > 0:
                action_freq /= action_freq.sum()

            action_array = np.stack(recent_flat_actions, axis=0) if recent_flat_actions else np.zeros((0, action_layout.total_param_dim), dtype=np.float32)
            aggregated = aggregate_metric_buffer(metric_buffer)

            train_metrics = {
                "train/global_step": float(global_step),
                "train/episodes": float(episode_count),
                "train/fps": float(fps),
                "train/replay_size": float(replay_buffer.size),
                "train/replay_fill_ratio": float(replay_buffer.size / replay_buffer.capacity),
                "train/learning_started": float(global_step >= cfg.sac.learning_starts),
                "train/recent_episode_reward_100": float(np.mean(recent_episode_rewards)) if recent_episode_rewards else 0.0,
                "train/recent_episode_length_100": float(np.mean(recent_episode_lengths)) if recent_episode_lengths else 0.0,
                "train/recent_reset_seed_mean": float(np.mean(recent_reset_seeds)) if recent_reset_seeds else 0.0,
                "train/recent_reset_seed_std": float(np.std(recent_reset_seeds)) if recent_reset_seeds else 0.0,
            }
            train_metrics.update(aggregated)

            for k in range(action_layout.num_discrete):
                train_metrics[f"train/action_{k}_freq"] = float(action_freq[k])

            if action_array.size > 0:
                train_metrics["train/flat_action_mean"] = float(action_array.mean())
                train_metrics["train/flat_action_std"] = float(action_array.std())
                for k in range(action_layout.num_discrete):
                    start, end = action_layout.branch_slices[k]
                    if end > start:
                        branch_vals = action_array[:, start:end]
                        train_metrics[f"train/action_{k}_param_mean"] = float(branch_vals.mean())
                        train_metrics[f"train/action_{k}_param_std"] = float(branch_vals.std())

            wandb_log(train_metrics, step=global_step)
            progress.set_postfix(
                step=global_step,
                ep_rew=f"{train_metrics['train/recent_episode_reward_100']:.1f}",
                q=f"{train_metrics.get('loss/qf_loss', 0.0):.3f}",
                pi=f"{train_metrics.get('loss/policy_loss', 0.0):.3f}",
                alpha=f"{train_metrics.get('loss/alpha', cfg.sac.alpha):.3f}",
                fps=f"{fps:.0f}",
            )
            print(
                f"[step {global_step:>8d}] "
                f"episodes={episode_count:>5d} | "
                f"reward_100={train_metrics['train/recent_episode_reward_100']:>8.2f} | "
                f"len_100={train_metrics['train/recent_episode_length_100']:>7.2f} | "
                f"qf={train_metrics.get('loss/qf_loss', 0.0):>8.4f} | "
                f"pi={train_metrics.get('loss/policy_loss', 0.0):>8.4f} | "
                f"alpha={train_metrics.get('loss/alpha', cfg.sac.alpha):>7.4f} | "
                f"alpha_d={train_metrics.get('loss/alpha_d', cfg.sac.alpha):>7.4f} | "
                f"replay={replay_buffer.size:>7d} | "
                f"fps={fps:.0f}"
            )
            metric_buffer.clear()
            recent_action_counts.fill(0)
            recent_flat_actions.clear()

        if global_step % cfg.training.eval_interval == 0 or global_step == cfg.sac.total_timesteps:
            eval_metrics = evaluate_policy(policy=policy, env=eval_env, action_layout=action_layout, eval_seeds=eval_seeds, device=device)
            eval_metrics["eval/best_mean_reward"] = float(max(best_eval_reward, eval_metrics["eval/mean_reward"]))
            wandb_log(eval_metrics, step=global_step)
            print(
                f"  [eval step {global_step}] "
                f"mean_reward={eval_metrics['eval/mean_reward']:.2f} +/- {eval_metrics['eval/std_reward']:.2f} | "
                f"mean_length={eval_metrics['eval/mean_length']:.1f}"
            )

            if eval_metrics["eval/mean_reward"] > best_eval_reward:
                best_eval_reward = eval_metrics["eval/mean_reward"]
                if cfg.training.save_best:
                    best_path = checkpoint_dir / "best_model.pt"
                    save_checkpoint(
                        best_path,
                        policy=policy,
                        qf1=qf1,
                        qf2=qf2,
                        qf1_target=qf1_target,
                        qf2_target=qf2_target,
                        policy_optimizer=policy_optimizer,
                        q_optimizer=q_optimizer,
                        cfg=cfg,
                        action_layout=action_layout,
                        global_step=global_step,
                        episode_count=episode_count,
                        best_eval_reward=best_eval_reward,
                        log_alpha=log_alpha,
                        log_alpha_d=log_alpha_d,
                        alpha_optimizer=alpha_optimizer,
                        alpha_d_optimizer=alpha_d_optimizer,
                    )
                    print(f"  New best checkpoint: {best_path}")
                    if run is not None and cfg.logging.log_model:
                        wandb.save(str(best_path))

        progress.update(1)

    progress.close()

    if cfg.training.save_final:
        final_path = checkpoint_dir / "final_model.pt"
        save_checkpoint(
            final_path,
            policy=policy,
            qf1=qf1,
            qf2=qf2,
            qf1_target=qf1_target,
            qf2_target=qf2_target,
            policy_optimizer=policy_optimizer,
            q_optimizer=q_optimizer,
            cfg=cfg,
            action_layout=action_layout,
            global_step=global_step,
            episode_count=episode_count,
            best_eval_reward=best_eval_reward,
            log_alpha=log_alpha,
            log_alpha_d=log_alpha_d,
            alpha_optimizer=alpha_optimizer,
            alpha_d_optimizer=alpha_d_optimizer,
        )
        print(f"Final checkpoint saved to {final_path}")
        if run is not None and cfg.logging.log_model:
            wandb.save(str(final_path))

    env.close()
    eval_env.close()
    if run is not None:
        run.finish()


OmegaConf.register_new_resolver("eval", eval, replace=True)


@hydra.main(version_base=None, config_path="../config", config_name="train_hybrid_sac_extended")
def main(cfg: DictConfig):
    train_hybrid_sac(cfg)


if __name__ == "__main__":
    main()
