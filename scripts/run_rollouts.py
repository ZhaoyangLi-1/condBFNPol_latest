#!/usr/bin/env python3
"""
Run simulation rollouts to evaluate trained policies.
Measures actual task success rate - the gold standard metric.
"""

import os
import sys

# Set project path FIRST before any other imports
sys.path.insert(0, '$PROJECT_ROOT')

# Force OSMesa (software) rendering BEFORE importing mujoco_py
os.environ['MUJOCO_GL'] = 'osmesa'
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'

import json
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
import hydra
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("eval", eval, replace=True)

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')


def load_policy_from_checkpoint(ckpt_path, device='cuda'):
    """Load policy from checkpoint."""
    print(f"  Loading: {os.path.basename(ckpt_path)}")
    ckpt = torch.load(ckpt_path, map_location='cpu')
    cfg = ckpt['cfg']
    
    # Instantiate policy
    policy = hydra.utils.instantiate(cfg.policy)
    
    # Load EMA weights if available
    if 'ema_model' in ckpt['state_dicts']:
        policy.load_state_dict(ckpt['state_dicts']['ema_model'])
        print("    Using EMA weights")
    else:
        policy.load_state_dict(ckpt['state_dicts']['model'])
        print("    Using model weights")
    
    policy.to(device)
    policy.eval()
    
    return policy, cfg


def create_env(env_name='Lift', render=False):
    """Create RoboMimic environment."""
    import robomimic.utils.env_utils as EnvUtils
    import robomimic.utils.file_utils as FileUtils
    
    # Get env meta from dataset
    dataset_path = f'$PROJECT_ROOT/data/robomimic/datasets/lift/ph/image.hdf5'
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
    
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=render,
        render_offscreen=True,
    )
    
    return env


def get_obs_dict(obs, shape_meta):
    """Convert environment observation to policy input format."""
    obs_dict = {}
    
    # Low-dim observations
    if 'robot0_eef_pos' in obs:
        obs_dict['robot0_eef_pos'] = torch.from_numpy(obs['robot0_eef_pos']).float()
    if 'robot0_eef_quat' in obs:
        obs_dict['robot0_eef_quat'] = torch.from_numpy(obs['robot0_eef_quat']).float()
    if 'robot0_gripper_qpos' in obs:
        obs_dict['robot0_gripper_qpos'] = torch.from_numpy(obs['robot0_gripper_qpos']).float()
    
    # Image observations
    if 'agentview_image' in obs:
        img = obs['agentview_image']
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0
        # HWC -> CHW
        if img.ndim == 3 and img.shape[-1] == 3:
            img = np.transpose(img, (2, 0, 1))
        obs_dict['agentview_image'] = torch.from_numpy(img).float()
    
    if 'robot0_eye_in_hand_image' in obs:
        img = obs['robot0_eye_in_hand_image']
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0
        if img.ndim == 3 and img.shape[-1] == 3:
            img = np.transpose(img, (2, 0, 1))
        obs_dict['robot0_eye_in_hand_image'] = torch.from_numpy(img).float()
    
    return obs_dict


def run_rollout(policy, env, max_steps=400, device='cuda'):
    """Run a single rollout and return success."""
    obs = env.reset()
    
    policy.reset()
    
    done = False
    success = False
    total_reward = 0
    step = 0
    
    # Observation history for temporal stacking
    obs_history = []
    n_obs_steps = 2  # Usually 2 observation steps
    
    while not done and step < max_steps:
        # Get observation dict
        obs_dict = get_obs_dict(obs, None)
        obs_history.append(obs_dict)
        
        # Keep only last n_obs_steps
        if len(obs_history) > n_obs_steps:
            obs_history = obs_history[-n_obs_steps:]
        
        # Pad if needed
        while len(obs_history) < n_obs_steps:
            obs_history.insert(0, obs_dict)
        
        # Stack observations
        stacked_obs = {}
        for key in obs_history[0].keys():
            stacked = torch.stack([o[key] for o in obs_history], dim=0)
            stacked_obs[key] = stacked.unsqueeze(0).to(device)  # Add batch dim
        
        # Get action from policy
        with torch.no_grad():
            try:
                action_dict = policy.predict_action(stacked_obs)
                action = action_dict['action'].cpu().numpy()[0, 0]  # First action in chunk
            except Exception as e:
                print(f"    Action prediction error at step {step}: {e}")
                action = np.zeros(7)
        
        # Execute action
        obs, reward, done, info = env.step(action)
        total_reward += reward
        step += 1
        
        # Check success
        if 'success' in info and info['success']:
            success = True
            break
    
    return {
        'success': success,
        'reward': total_reward,
        'steps': step,
    }


def evaluate_policy(policy, env, n_episodes=50, device='cuda'):
    """Evaluate policy over multiple episodes."""
    results = []
    
    for ep in range(n_episodes):
        result = run_rollout(policy, env, device=device)
        results.append(result)
        
        status = "✓" if result['success'] else "✗"
        if (ep + 1) % 10 == 0:
            successes = sum(r['success'] for r in results)
            print(f"    Episode {ep+1}/{n_episodes}: {status} | Running success rate: {successes}/{ep+1} ({100*successes/(ep+1):.1f}%)")
    
    # Aggregate results
    successes = sum(r['success'] for r in results)
    success_rate = successes / n_episodes
    avg_reward = np.mean([r['reward'] for r in results])
    avg_steps = np.mean([r['steps'] for r in results])
    
    return {
        'success_rate': success_rate,
        'successes': successes,
        'total_episodes': n_episodes,
        'avg_reward': avg_reward,
        'avg_steps': avg_steps,
        'success_steps': np.mean([r['steps'] for r in results if r['success']]) if successes > 0 else 0,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_episodes', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    print("=" * 60)
    print("ROLLOUT EVALUATION")
    print("=" * 60)
    
    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Episodes per policy: {args.n_episodes}")
    
    # Find all checkpoints
    outputs_dir = '$PROJECT_ROOT/outputs'
    checkpoints = {'bfn': [], 'diffusion': []}
    
    for exp_name in sorted(os.listdir(outputs_dir)):
        if not exp_name.startswith('thesis_'):
            continue
        ckpt_dir = os.path.join(outputs_dir, exp_name, 'checkpoints')
        if os.path.exists(ckpt_dir):
            for f in os.listdir(ckpt_dir):
                if f.endswith('.ckpt'):
                    ckpt_path = os.path.join(ckpt_dir, f)
                    model_type = 'bfn' if 'bfn' in exp_name else 'diffusion'
                    seed = exp_name.split('seed')[-1]
                    checkpoints[model_type].append({
                        'path': ckpt_path,
                        'name': exp_name,
                        'seed': seed,
                    })
    
    print(f"\nFound {len(checkpoints['bfn'])} BFN checkpoints")
    print(f"Found {len(checkpoints['diffusion'])} Diffusion checkpoints")
    
    # Create environment
    print("\nCreating environment...")
    env = create_env()
    print("  Environment ready")
    
    # Results storage
    all_results = {'bfn': [], 'diffusion': []}
    
    # Evaluate each checkpoint
    for model_type in ['bfn', 'diffusion']:
        print(f"\n{'='*60}")
        print(f"Evaluating {model_type.upper()} policies")
        print('='*60)
        
        for ckpt_info in checkpoints[model_type]:
            print(f"\n--- {ckpt_info['name']} ---")
            
            try:
                policy, cfg = load_policy_from_checkpoint(ckpt_info['path'], device)
                results = evaluate_policy(policy, env, n_episodes=args.n_episodes, device=device)
                results['name'] = ckpt_info['name']
                results['seed'] = ckpt_info['seed']
                all_results[model_type].append(results)
                
                print(f"  SUCCESS RATE: {results['success_rate']*100:.1f}% ({results['successes']}/{results['total_episodes']})")
                print(f"  Avg reward: {results['avg_reward']:.2f}")
                print(f"  Avg steps: {results['avg_steps']:.1f}")
                
                # Clean up
                del policy
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for model_type in ['bfn', 'diffusion']:
        if all_results[model_type]:
            rates = [r['success_rate'] for r in all_results[model_type]]
            mean_rate = np.mean(rates) * 100
            std_rate = np.std(rates) * 100
            print(f"\n{model_type.upper()}:")
            print(f"  Success Rate: {mean_rate:.1f}% ± {std_rate:.1f}%")
            for r in all_results[model_type]:
                print(f"    {r['name']}: {r['success_rate']*100:.1f}%")
    
    # Comparison
    if all_results['bfn'] and all_results['diffusion']:
        bfn_rate = np.mean([r['success_rate'] for r in all_results['bfn']]) * 100
        diff_rate = np.mean([r['success_rate'] for r in all_results['diffusion']]) * 100
        
        print("\n" + "-" * 40)
        print("COMPARISON")
        print("-" * 40)
        print(f"BFN Success Rate:       {bfn_rate:.1f}%")
        print(f"Diffusion Success Rate: {diff_rate:.1f}%")
        
        if bfn_rate > diff_rate:
            print(f"\n🏆 WINNER: BFN-Policy (+{bfn_rate - diff_rate:.1f}%)")
        elif diff_rate > bfn_rate:
            print(f"\n🏆 WINNER: Diffusion Policy (+{diff_rate - bfn_rate:.1f}%)")
        else:
            print(f"\n🏆 TIE")
    
    # Save results
    output_file = '$PROJECT_ROOT/thesis_figures/rollout_results.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_file}")
    
    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == '__main__':
    main()
