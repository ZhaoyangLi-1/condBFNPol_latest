#!/usr/bin/env python3
"""Benchmark all policies (BFN, Diffusion, Consistency) on PushT.

Computes:
- Success Rate / Mean Score (coverage IoU)
- Inference Time per action
- Number of denoising steps

Usage:
    python scripts/benchmark_policies.py \
        --bfn_ckpt /path/to/bfn/checkpoint.ckpt \
        --diffusion_ckpt /path/to/diffusion/checkpoint.ckpt \
        --consistency_ckpt /path/to/consistency/checkpoint.ckpt \
        --output results/benchmark.json
"""

import os
import sys
sys.path.insert(0, '$PROJECT_ROOT')
os.environ['MUJOCO_GL'] = 'osmesa'

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import numpy as np
import hydra
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("eval", eval, replace=True)


def load_policy(ckpt_path: str, device: str = 'cuda'):
    """Load policy from checkpoint."""
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu')
    cfg = ckpt['cfg']

    policy = hydra.utils.instantiate(cfg.policy)

    # Load state dict (prefer EMA if available)
    if 'ema_model' in ckpt['state_dicts']:
        policy.load_state_dict(ckpt['state_dicts']['ema_model'])
        print("  Loaded EMA model weights")
    else:
        policy.load_state_dict(ckpt['state_dicts']['model'])
        print("  Loaded model weights")

    # Load normalizer
    if 'normalizer' in ckpt:
        policy.set_normalizer(ckpt['normalizer'])

    policy.to(device)
    policy.eval()

    return policy, cfg


def measure_inference_time(policy, obs_dict: Dict[str, torch.Tensor],
                           n_warmup: int = 5, n_runs: int = 20) -> Dict[str, float]:
    """Measure inference time for a single action prediction."""
    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = policy.predict_action(obs_dict)

    # Synchronize CUDA
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Time multiple runs
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            start = time.perf_counter()
            _ = policy.predict_action(obs_dict)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append(end - start)

    return {
        'mean_ms': np.mean(times) * 1000,
        'std_ms': np.std(times) * 1000,
        'min_ms': np.min(times) * 1000,
        'max_ms': np.max(times) * 1000,
    }


def run_rollout_evaluation(policy, cfg, n_test: int = 50, device: str = 'cuda') -> Dict[str, Any]:
    """Run full rollout evaluation using a simple single environment (no AsyncVectorEnv)."""
    from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv
    from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper

    n_obs_steps = cfg.n_obs_steps
    n_action_steps = cfg.n_action_steps
    max_steps = 300
    test_start_seed = 100000

    test_scores = []

    for episode_idx in range(n_test):
        seed = test_start_seed + episode_idx

        # Create fresh environment for each episode
        env = PushTImageEnv(render_size=96)
        env = MultiStepWrapper(
            env,
            n_obs_steps=n_obs_steps,
            n_action_steps=n_action_steps,
            max_episode_steps=max_steps
        )

        # Reset with seed
        env.seed(seed)
        obs = env.reset()

        done = False
        max_reward = 0.0

        while not done:
            # Prepare observation dict for policy
            obs_dict = {}
            for key, value in obs.items():
                obs_dict[key] = torch.from_numpy(value).unsqueeze(0).to(device)

            # Get action from policy
            with torch.no_grad():
                action_dict = policy.predict_action(obs_dict)
                action = action_dict['action'].squeeze(0).cpu().numpy()

            # Step environment
            obs, reward, done, info = env.step(action)
            max_reward = max(max_reward, reward)

        test_scores.append(max_reward)
        env.close()

        if (episode_idx + 1) % 10 == 0:
            print(f"    Episode {episode_idx + 1}/{n_test}: max_reward={max_reward:.3f}")

    results = {
        'mean_score': np.mean(test_scores),
        'all_scores': test_scores,
        'std_score': np.std(test_scores),
        'n_episodes': len(test_scores),
    }

    # Compute success rate (threshold 0.9 is common for PushT)
    results['success_rate_90'] = np.mean([s >= 0.9 for s in test_scores])
    results['success_rate_80'] = np.mean([s >= 0.8 for s in test_scores])
    results['success_rate_70'] = np.mean([s >= 0.7 for s in test_scores])

    return results


def get_policy_info(policy) -> Dict[str, Any]:
    """Get policy metadata (num steps, architecture info)."""
    info = {
        'n_params': sum(p.numel() for p in policy.parameters()),
        'n_trainable_params': sum(p.numel() for p in policy.parameters() if p.requires_grad),
    }

    # Get number of denoising steps
    if hasattr(policy, 'noise_scheduler'):
        info['n_diffusion_steps'] = getattr(policy.noise_scheduler, 'bins', None)
    if hasattr(policy, 'num_inference_steps'):
        info['num_inference_steps'] = policy.num_inference_steps
    if hasattr(policy, 'inference_steps'):
        info['inference_steps'] = policy.inference_steps

    # Policy type
    info['policy_class'] = policy.__class__.__name__

    return info


def create_dummy_obs(cfg, device: str = 'cuda') -> Dict[str, torch.Tensor]:
    """Create dummy observation for timing measurements."""
    shape_meta = cfg.shape_meta
    n_obs_steps = cfg.n_obs_steps

    obs_dict = {}
    for key, attr in shape_meta.obs.items():
        shape = list(attr.shape)
        obs_type = attr.get('type', 'low_dim')

        if obs_type == 'rgb':
            # Image: (B, T, C, H, W)
            obs_dict[key] = torch.randn(1, n_obs_steps, *shape, device=device)
        else:
            # Low-dim: (B, T, D)
            obs_dict[key] = torch.randn(1, n_obs_steps, *shape, device=device)

    return obs_dict


def benchmark_policy(name: str, ckpt_path: str, device: str = 'cuda',
                     run_rollouts: bool = True, enable_chaining: bool = False) -> Dict[str, Any]:
    """Full benchmark for a single policy.

    Args:
        enable_chaining: For consistency policy, enable multi-step chaining (3-step).
    """
    results = {'name': name, 'checkpoint': ckpt_path}

    try:
        policy, cfg = load_policy(ckpt_path, device)

        # For consistency policy, configure chaining mode
        if hasattr(policy, 'enable_chaining') and hasattr(policy, 'disable_chaining'):
            if enable_chaining:
                policy.enable_chaining()
                results['chaining'] = True
                print(f"  Chaining: ENABLED (3-step)")
            else:
                policy.disable_chaining()
                results['chaining'] = False
                print(f"  Chaining: DISABLED (1-step)")

        # Policy info
        results['info'] = get_policy_info(policy)
        print(f"  Policy: {results['info']['policy_class']}")
        print(f"  Parameters: {results['info']['n_params']:,}")

        # Inference time
        print("  Measuring inference time...")
        obs_dict = create_dummy_obs(cfg, device)
        results['inference_time'] = measure_inference_time(policy, obs_dict)
        print(f"  Inference: {results['inference_time']['mean_ms']:.2f} ± {results['inference_time']['std_ms']:.2f} ms")

        # Rollout evaluation
        if run_rollouts:
            print("  Running rollout evaluation...")
            results['rollout'] = run_rollout_evaluation(policy, cfg, device=device)
            print(f"  Mean Score: {results['rollout']['mean_score']:.3f}")
            if 'success_rate_90' in results['rollout']:
                print(f"  Success Rate (≥90%): {results['rollout']['success_rate_90']:.1%}")

        results['status'] = 'success'

    except Exception as e:
        results['status'] = 'error'
        results['error'] = str(e)
        import traceback
        traceback.print_exc()
        print(f"  ERROR: {e}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Benchmark policies on PushT')
    parser.add_argument('--bfn_ckpt', type=str, help='BFN checkpoint path')
    parser.add_argument('--diffusion_ckpt', type=str, help='Diffusion Policy (DDPM) checkpoint path')
    parser.add_argument('--ddim_ckpt', type=str, help='DDIM baseline checkpoint path')
    parser.add_argument('--consistency_ckpt', type=str, help='Consistency Policy checkpoint path')
    parser.add_argument('--edm_ckpt', type=str, help='EDM teacher checkpoint path')
    parser.add_argument('--output', type=str, default='results/benchmark.json', help='Output file')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--no_rollouts', action='store_true', help='Skip rollout evaluation')
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    results = {}

    # Benchmark each policy
    policies = [
        ('BFN', args.bfn_ckpt),
        ('Diffusion', args.diffusion_ckpt),
        ('DDIM', args.ddim_ckpt),
        ('EDM', args.edm_ckpt),
    ]

    for name, ckpt_path in policies:
        if ckpt_path and Path(ckpt_path).exists():
            print(f"\n{'='*60}")
            print(f"Benchmarking: {name}")
            print(f"{'='*60}")
            results[name.lower()] = benchmark_policy(
                name, ckpt_path, device,
                run_rollouts=not args.no_rollouts
            )
        else:
            if ckpt_path:
                print(f"\nSkipping {name}: checkpoint not found at {ckpt_path}")

    # Benchmark Consistency Policy with both 1-step and 3-step
    if args.consistency_ckpt and Path(args.consistency_ckpt).exists():
        # 1-step (fastest)
        print(f"\n{'='*60}")
        print(f"Benchmarking: Consistency (1-step)")
        print(f"{'='*60}")
        results['consistency_1step'] = benchmark_policy(
            'Consistency-1step', args.consistency_ckpt, device,
            run_rollouts=not args.no_rollouts, enable_chaining=False
        )

        # 3-step (better quality)
        print(f"\n{'='*60}")
        print(f"Benchmarking: Consistency (3-step)")
        print(f"{'='*60}")
        results['consistency_3step'] = benchmark_policy(
            'Consistency-3step', args.consistency_ckpt, device,
            run_rollouts=not args.no_rollouts, enable_chaining=True
        )
    elif args.consistency_ckpt:
        print(f"\nSkipping Consistency: checkpoint not found at {args.consistency_ckpt}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    print(f"\n{'Policy':<15} {'Mean Score':<12} {'Success@90':<12} {'Inference (ms)':<15} {'Steps':<8}")
    print("-" * 62)

    for name, data in results.items():
        if data.get('status') == 'success':
            score = data.get('rollout', {}).get('mean_score', '-')
            success = data.get('rollout', {}).get('success_rate_90', '-')
            inf_time = data.get('inference_time', {}).get('mean_ms', '-')
            steps = data.get('info', {}).get('num_inference_steps',
                    data.get('info', {}).get('n_diffusion_steps', '-'))

            score_str = f"{score:.3f}" if isinstance(score, float) else str(score)
            success_str = f"{success:.1%}" if isinstance(success, float) else str(success)
            inf_str = f"{inf_time:.2f}" if isinstance(inf_time, float) else str(inf_time)

            print(f"{name.upper():<15} {score_str:<12} {success_str:<12} {inf_str:<15} {steps}")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
