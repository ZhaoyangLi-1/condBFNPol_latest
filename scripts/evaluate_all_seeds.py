#!/usr/bin/env python3
"""Evaluate all BFN and Diffusion checkpoints across all seeds."""

import os
import sys
sys.path.insert(0, '$PROJECT_ROOT')
os.environ['MUJOCO_GL'] = 'osmesa'

import json
import torch
import numpy as np
import h5py
from pathlib import Path
import hydra
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("eval", eval, replace=True)

def load_policy(ckpt_path, device='cpu'):
    """Load policy from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location='cpu')
    cfg = ckpt['cfg']
    policy = hydra.utils.instantiate(cfg.policy)
    
    if 'ema_model' in ckpt['state_dicts']:
        policy.load_state_dict(ckpt['state_dicts']['ema_model'])
    else:
        policy.load_state_dict(ckpt['state_dicts']['model'])
    
    policy.to(device)
    policy.eval()
    return policy, cfg

def evaluate_policy(policy, dataset_path, device='cpu', n_demos=10, n_timesteps_per_demo=5):
    """Evaluate policy on dataset demos."""
    mse_list = []
    
    with h5py.File(dataset_path, 'r') as f:
        demos = list(f['data'].keys())[:n_demos]
        
        for demo_key in demos:
            demo = f['data'][demo_key]
            T = demo['actions'].shape[0]
            
            # Sample timesteps evenly
            timesteps = np.linspace(2, T-2, n_timesteps_per_demo, dtype=int)
            
            for t in timesteps:
                obs_dict = {}
                
                # Low-dim observations (2 timesteps for history)
                for key in ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos']:
                    if f'obs/{key}' in demo:
                        data = demo[f'obs/{key}'][t-1:t+1]
                        obs_dict[key] = torch.from_numpy(data).float().unsqueeze(0).to(device)
                
                # Image observations
                for key in ['agentview_image', 'robot0_eye_in_hand_image']:
                    if f'obs/{key}' in demo:
                        data = demo[f'obs/{key}'][t-1:t+1].astype(np.float32) / 255.0
                        data = np.transpose(data, (0, 3, 1, 2))
                        obs_dict[key] = torch.from_numpy(data).float().unsqueeze(0).to(device)
                
                gt_action = demo['actions'][t]
                
                try:
                    with torch.no_grad():
                        pred = policy.predict_action(obs_dict)
                        pred_action = pred['action'].cpu().numpy()[0, 0]  # First action in horizon
                    
                    mse = np.mean((pred_action - gt_action) ** 2)
                    mse_list.append(mse)
                except Exception as e:
                    print(f"    Error: {e}")
    
    if mse_list:
        return np.mean(mse_list), np.std(mse_list), len(mse_list)
    return None, None, 0

def main():
    print("=" * 60)
    print("COMPREHENSIVE POLICY EVALUATION")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    dataset_path = '$PROJECT_ROOT/data/robomimic/datasets/lift/ph/image.hdf5'
    outputs_dir = Path('$PROJECT_ROOT/outputs')
    
    results = {'bfn': {}, 'diffusion': {}}
    
    # Find all thesis checkpoints
    for exp_dir in sorted(outputs_dir.iterdir()):
        if not exp_dir.name.startswith('thesis_'):
            continue
        
        ckpt_dir = exp_dir / 'checkpoints'
        if not ckpt_dir.exists():
            continue
        
        ckpts = sorted(ckpt_dir.glob('*.ckpt'))
        if not ckpts:
            continue
        
        ckpt_path = str(ckpts[-1])  # Latest checkpoint
        exp_name = exp_dir.name
        
        print(f"\n{'=' * 60}")
        print(f"Evaluating: {exp_name}")
        print(f"Checkpoint: {ckpts[-1].name}")
        print(f"{'=' * 60}")
        
        try:
            policy, cfg = load_policy(ckpt_path, device)
            print("  Policy loaded successfully")
            
            mse_mean, mse_std, n_samples = evaluate_policy(
                policy, dataset_path, device, n_demos=20, n_timesteps_per_demo=10
            )
            
            if mse_mean is not None:
                print(f"  MSE: {mse_mean:.6f} ± {mse_std:.6f} (n={n_samples})")
                
                # Determine policy type
                if 'bfn' in exp_name.lower():
                    results['bfn'][exp_name] = {
                        'mse_mean': float(mse_mean),
                        'mse_std': float(mse_std),
                        'n_samples': n_samples
                    }
                elif 'diffusion' in exp_name.lower():
                    results['diffusion'][exp_name] = {
                        'mse_mean': float(mse_mean),
                        'mse_std': float(mse_std),
                        'n_samples': n_samples
                    }
        except Exception as e:
            print(f"  ERROR: {e}")
    
    # Compute aggregated stats
    print("\n" + "=" * 60)
    print("AGGREGATED RESULTS")
    print("=" * 60)
    
    for policy_type in ['bfn', 'diffusion']:
        if results[policy_type]:
            mses = [r['mse_mean'] for r in results[policy_type].values()]
            print(f"\n{policy_type.upper()}:")
            print(f"  Individual MSEs: {[f'{m:.6f}' for m in mses]}")
            print(f"  Mean MSE: {np.mean(mses):.6f} ± {np.std(mses):.6f}")
    
    # Save results
    output_file = '$PROJECT_ROOT/thesis_figures/all_seeds_evaluation.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")

if __name__ == '__main__':
    main()
