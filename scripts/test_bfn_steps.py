#!/usr/bin/env python3
"""Test BFN with different inference step counts."""

import os
os.environ['MUJOCO_GL'] = 'osmesa'
import sys
sys.path.insert(0, '$PROJECT_ROOT')

import torch
import numpy as np
import time
import hydra
from omegaconf import OmegaConf
OmegaConf.register_new_resolver('eval', eval, replace=True)

from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper

def run_rollouts(policy, n_episodes=50, device='cuda'):
    """Run rollout evaluation."""
    n_obs_steps = 2
    n_action_steps = 8
    max_steps = 300
    test_start_seed = 100000

    scores = []
    for ep in range(n_episodes):
        seed = test_start_seed + ep
        env = PushTImageEnv(render_size=96)
        env = MultiStepWrapper(env, n_obs_steps=n_obs_steps, n_action_steps=n_action_steps, max_episode_steps=max_steps)
        env.seed(seed)
        obs = env.reset()
        done = False
        max_reward = 0.0

        while not done:
            obs_dict = {k: torch.from_numpy(v).unsqueeze(0).to(device) for k, v in obs.items()}
            with torch.no_grad():
                action = policy.predict_action(obs_dict)['action'].squeeze(0).cpu().numpy()
            obs, reward, done, info = env.step(action)
            max_reward = max(max_reward, reward)

        scores.append(max_reward)
        env.close()

        if (ep + 1) % 10 == 0:
            print(f"  Episode {ep+1}/{n_episodes}: score={max_reward:.3f}")

    return scores

# Load BFN checkpoint
ckpt_path = 'outputs/2026.01.26/19.06.04_bfn_seed42/checkpoints/epoch=0200-test_mean_score=0.907.ckpt'
print(f'Loading BFN checkpoint...')
ckpt = torch.load(ckpt_path, map_location='cpu')
cfg = ckpt['cfg']

policy = hydra.utils.instantiate(cfg.policy)
policy.load_state_dict(ckpt['state_dicts']['ema_model'])
if 'normalizer' in ckpt:
    policy.set_normalizer(ckpt['normalizer'])
policy.to('cuda')
policy.eval()

print(f'Original n_timesteps: {policy.n_timesteps}')
print()

# Test with different step counts
results = {}
for n_steps in [20, 10, 5, 3, 1]:
    print(f"=" * 60)
    print(f"Testing BFN with {n_steps} steps")
    print(f"=" * 60)

    policy.n_timesteps = n_steps

    # Measure inference time
    obs_dict = {
        'image': torch.randn(1, 2, 3, 96, 96).cuda(),
        'agent_pos': torch.randn(1, 2, 2).cuda()
    }

    # Warmup
    for _ in range(5):
        with torch.no_grad():
            _ = policy.predict_action(obs_dict)

    torch.cuda.synchronize()

    # Time
    times = []
    for _ in range(20):
        start = time.perf_counter()
        with torch.no_grad():
            _ = policy.predict_action(obs_dict)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)

    inference_ms = np.mean(times)
    print(f"Inference time: {inference_ms:.1f} ms")

    # Run rollouts
    print("Running rollouts...")
    scores = run_rollouts(policy, n_episodes=50)
    mean_score = np.mean(scores)
    success_90 = np.mean([s >= 0.9 for s in scores])
    success_80 = np.mean([s >= 0.8 for s in scores])

    print(f"Mean score: {mean_score:.3f}")
    print(f"Success@90: {success_90*100:.0f}%")
    print(f"Success@80: {success_80*100:.0f}%")
    print()

    results[n_steps] = {
        'inference_ms': inference_ms,
        'mean_score': mean_score,
        'success_90': success_90,
        'success_80': success_80,
        'scores': scores
    }

# Summary
print("=" * 70)
print("SUMMARY: BFN with different inference steps")
print("=" * 70)
print(f"{'Steps':>6} {'Inference (ms)':>15} {'Mean Score':>12} {'Success@90':>12} {'Success@80':>12}")
print("-" * 62)
for n_steps in [20, 10, 5, 3, 1]:
    r = results[n_steps]
    print(f"{n_steps:>6} {r['inference_ms']:>15.1f} {r['mean_score']:>12.3f} {r['success_90']*100:>11.0f}% {r['success_80']*100:>11.0f}%")
