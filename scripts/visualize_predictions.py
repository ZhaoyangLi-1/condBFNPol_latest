#!/usr/bin/env python3
"""
Visualize model predictions vs ground truth actions.
Helps verify that trained models are making reasonable predictions.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import hydra
from omegaconf import OmegaConf

# Add project root to path
sys.path.insert(0, '$PROJECT_ROOT')

# Register eval resolver for hydra configs
OmegaConf.register_new_resolver("eval", eval, replace=True)

# Anthropic colors
TEAL = '#2D8B8B'
CORAL = '#D97757'
SLATE = '#4A4A4A'
TEAL_LT = '#B4D7D7'
CORAL_LT = '#F3C8B4'


def setup_style():
    """Configure matplotlib."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 9,
        'axes.titlesize': 10,
        'axes.labelsize': 9,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': 0.6,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })


def load_checkpoint_and_dataset(ckpt_path):
    """Load checkpoint, instantiate policy and dataset."""
    print(f"Loading: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu')
    cfg = ckpt['cfg']
    
    # Instantiate dataset from config
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    
    # Instantiate policy from config
    policy = hydra.utils.instantiate(cfg.policy)
    
    # Load weights (EMA if available)
    if 'ema_model' in ckpt['state_dicts']:
        policy.load_state_dict(ckpt['state_dicts']['ema_model'])
        print("  Loaded EMA weights")
    else:
        policy.load_state_dict(ckpt['state_dicts']['model'])
        print("  Loaded model weights")
    
    return policy, dataset, cfg


def run_inference(policy, dataset, num_samples=5, device='cuda'):
    """Run inference and collect predictions."""
    policy.to(device)
    policy.eval()
    
    results = []
    indices = np.random.RandomState(42).choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    print(f"  Running inference on {len(indices)} samples...")
    
    with torch.no_grad():
        for idx in indices:
            batch = dataset[idx]
            
            # Prepare observation dict
            obs_dict = {}
            for key, value in batch['obs'].items():
                if isinstance(value, torch.Tensor):
                    obs_dict[key] = value.unsqueeze(0).to(device)
                elif isinstance(value, np.ndarray):
                    obs_dict[key] = torch.from_numpy(value).unsqueeze(0).to(device)
            
            # Get ground truth
            gt_action = batch['action']
            if isinstance(gt_action, torch.Tensor):
                gt_action = gt_action.numpy()
            
            # Run policy
            try:
                pred = policy.predict_action(obs_dict)
                pred_action = pred['action'].cpu().numpy()[0]
                
                # Ensure same length
                min_len = min(gt_action.shape[0], pred_action.shape[0])
                gt_action = gt_action[:min_len]
                pred_action = pred_action[:min_len]
                
            except Exception as e:
                print(f"  Sample {idx} inference error: {e}")
                pred_action = np.zeros_like(gt_action)
            
            results.append({
                'idx': idx,
                'gt_action': gt_action,
                'pred_action': pred_action,
            })
    
    return results


def visualize_predictions_vs_gt(results, output_path, model_name="Model"):
    """Visualize predictions vs ground truth."""
    setup_style()
    
    n_samples = len(results)
    fig, axes = plt.subplots(n_samples, 2, figsize=(10, 2.5 * n_samples))
    
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i, result in enumerate(results):
        gt = result['gt_action']
        pred = result['pred_action']
        
        # Position (XYZ) - first 3 dims
        ax = axes[i, 0]
        timesteps = np.arange(gt.shape[0])
        
        for dim, (label, color) in enumerate(zip(['X', 'Y', 'Z'], [TEAL, TEAL, CORAL])):
            if dim < gt.shape[1]:
                ax.plot(timesteps, gt[:, dim], color=SLATE, linestyle='--', 
                       linewidth=1.5, alpha=0.7)
                ax.plot(timesteps, pred[:, dim], color=color, linewidth=2,
                       label=f'{label}' if i == 0 else None)
        
        ax.set_ylabel('Position')
        ax.set_xlabel('Timestep')
        ax.set_title(f'Sample {i+1}: Position (XYZ)')
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=7, loc='upper right')
        
        # Add MSE annotation
        pos_mse = np.mean((gt[:, :3] - pred[:, :3])**2)
        ax.text(0.02, 0.98, f'MSE: {pos_mse:.4f}', transform=ax.transAxes,
               fontsize=7, va='top', ha='left', 
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Gripper (dim 6 or last dim)
        ax = axes[i, 1]
        gripper_dim = 6 if gt.shape[1] > 6 else gt.shape[1] - 1
        
        ax.plot(timesteps, gt[:, gripper_dim], color=SLATE, linestyle='--',
               linewidth=2, label='Ground Truth' if i == 0 else None)
        ax.plot(timesteps, pred[:, gripper_dim], color=CORAL,
               linewidth=2, label='Prediction' if i == 0 else None)
        
        # Add threshold line for gripper
        ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
        
        ax.set_ylabel('Gripper')
        ax.set_xlabel('Timestep')
        ax.set_title(f'Sample {i+1}: Gripper State')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.1, 1.1)
        if i == 0:
            ax.legend(fontsize=7, loc='upper right')
    
    plt.suptitle(f'{model_name}: Predictions vs Ground Truth\n(dashed = GT, solid = pred)', 
                fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")


def compute_errors(results):
    """Compute prediction errors."""
    all_errors = []
    
    for result in results:
        gt = result['gt_action']
        pred = result['pred_action']
        
        min_dim = min(gt.shape[1], pred.shape[1])
        error = np.abs(gt[:, :min_dim] - pred[:, :min_dim])
        all_errors.append(error)
    
    return all_errors


def visualize_error_analysis(bfn_errors, diff_errors, output_path):
    """Visualize error comparison between models."""
    setup_style()
    
    fig, axes = plt.subplots(1, 3, figsize=(11, 3))
    
    # Panel A: Per-dimension mean error
    ax = axes[0]
    
    bfn_dim_mean = np.mean([e.mean(axis=0) for e in bfn_errors], axis=0)
    diff_dim_mean = np.mean([e.mean(axis=0) for e in diff_errors], axis=0)
    
    n_dims = min(len(bfn_dim_mean), len(diff_dim_mean), 7)
    x = np.arange(n_dims)
    width = 0.35
    
    labels = ['X', 'Y', 'Z', 'R1', 'R2', 'R3', 'Grip'][:n_dims]
    
    ax.bar(x - width/2, bfn_dim_mean[:n_dims], width, label='BFN (Ours)', color=TEAL)
    ax.bar(x + width/2, diff_dim_mean[:n_dims], width, label='Diffusion', color=CORAL)
    ax.set_xlabel('Action Dimension')
    ax.set_ylabel('Mean Absolute Error')
    ax.set_title('(a) Per-Dimension Error')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(fontsize=7)
    ax.grid(True, axis='y', alpha=0.3)
    
    # Panel B: Error over time
    ax = axes[1]
    
    bfn_time = np.mean([e.mean(axis=1) for e in bfn_errors], axis=0)
    diff_time = np.mean([e.mean(axis=1) for e in diff_errors], axis=0)
    
    ax.plot(bfn_time, color=TEAL, label='BFN (Ours)', linewidth=2)
    ax.plot(diff_time, color=CORAL, label='Diffusion', linewidth=2)
    ax.set_xlabel('Timestep in Action Chunk')
    ax.set_ylabel('Mean Absolute Error')
    ax.set_title('(b) Error Over Prediction Horizon')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    
    # Panel C: Error distribution
    ax = axes[2]
    
    bfn_flat = np.concatenate([e.flatten() for e in bfn_errors])
    diff_flat = np.concatenate([e.flatten() for e in diff_errors])
    
    ax.hist(bfn_flat, bins=50, alpha=0.6, color=TEAL, label='BFN (Ours)', density=True)
    ax.hist(diff_flat, bins=50, alpha=0.6, color=CORAL, label='Diffusion', density=True)
    ax.set_xlabel('Absolute Error')
    ax.set_ylabel('Density')
    ax.set_title('(c) Error Distribution')
    ax.legend(fontsize=7)
    ax.set_xlim(0, np.percentile(np.concatenate([bfn_flat, diff_flat]), 98))
    
    # Add statistics
    bfn_mae = np.mean(bfn_flat)
    diff_mae = np.mean(diff_flat)
    ax.axvline(bfn_mae, color=TEAL, linestyle='--', linewidth=1.5)
    ax.axvline(diff_mae, color=CORAL, linestyle='--', linewidth=1.5)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")


def visualize_overlay(bfn_results, diff_results, output_path):
    """Visualize BFN and Diffusion predictions overlaid."""
    setup_style()
    
    n_samples = min(3, len(bfn_results), len(diff_results))
    fig, axes = plt.subplots(n_samples, 2, figsize=(10, 2.5 * n_samples))
    
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_samples):
        bfn = bfn_results[i]
        diff = diff_results[i]
        gt = bfn['gt_action']
        
        # Position
        ax = axes[i, 0]
        timesteps = np.arange(gt.shape[0])
        
        for dim in range(min(3, gt.shape[1])):
            ax.plot(timesteps, gt[:, dim], color=SLATE, linestyle='--', 
                   linewidth=1.5, alpha=0.5)
        
        ax.plot(timesteps, bfn['pred_action'][:, 0], color=TEAL, linewidth=2, label='BFN')
        ax.plot(timesteps, diff['pred_action'][:, 0], color=CORAL, linewidth=2, label='Diffusion')
        
        ax.set_ylabel('X Position')
        ax.set_xlabel('Timestep')
        ax.set_title(f'Sample {i+1}: X Trajectory')
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=7)
        
        # Gripper
        ax = axes[i, 1]
        gripper_dim = 6 if gt.shape[1] > 6 else gt.shape[1] - 1
        
        ax.plot(timesteps, gt[:, gripper_dim], color=SLATE, linestyle='--',
               linewidth=2, alpha=0.5, label='GT')
        ax.plot(timesteps, bfn['pred_action'][:, gripper_dim], color=TEAL,
               linewidth=2, label='BFN')
        ax.plot(timesteps, diff['pred_action'][:, gripper_dim], color=CORAL,
               linewidth=2, label='Diffusion')
        
        ax.set_ylabel('Gripper')
        ax.set_xlabel('Timestep')
        ax.set_title(f'Sample {i+1}: Gripper')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.1, 1.1)
        if i == 0:
            ax.legend(fontsize=7)
    
    plt.suptitle('BFN vs Diffusion: Prediction Comparison', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    output_dir = '$PROJECT_ROOT/thesis_figures'
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 50)
    print("PREDICTION VISUALIZATION")
    print("=" * 50)
    
    # Find checkpoints
    outputs_dir = '$PROJECT_ROOT/outputs'
    
    bfn_ckpt = None
    diff_ckpt = None
    
    for exp_name in sorted(os.listdir(outputs_dir)):
        exp_path = os.path.join(outputs_dir, exp_name)
        if not exp_name.startswith('thesis_'):
            continue
        
        ckpt_dir = os.path.join(exp_path, 'checkpoints')
        if os.path.exists(ckpt_dir):
            for f in os.listdir(ckpt_dir):
                if f.endswith('.ckpt'):
                    ckpt_path = os.path.join(ckpt_dir, f)
                    if 'bfn' in exp_name and bfn_ckpt is None:
                        bfn_ckpt = ckpt_path
                    elif 'diffusion' in exp_name and diff_ckpt is None:
                        diff_ckpt = ckpt_path
    
    print(f"BFN checkpoint: {bfn_ckpt}")
    print(f"Diffusion checkpoint: {diff_ckpt}")
    
    if not bfn_ckpt or not diff_ckpt:
        print("ERROR: Could not find checkpoints!")
        return
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Process BFN
    print("\n--- BFN Policy ---")
    try:
        bfn_policy, bfn_dataset, _ = load_checkpoint_and_dataset(bfn_ckpt)
        bfn_results = run_inference(bfn_policy, bfn_dataset, num_samples=5, device=device)
        
        visualize_predictions_vs_gt(
            bfn_results, 
            os.path.join(output_dir, 'pred_bfn.png'),
            model_name="BFN-Policy"
        )
        bfn_errors = compute_errors(bfn_results)
        
        del bfn_policy
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"BFN error: {e}")
        import traceback
        traceback.print_exc()
        bfn_results = None
        bfn_errors = None
    
    # Process Diffusion
    print("\n--- Diffusion Policy ---")
    try:
        diff_policy, diff_dataset, _ = load_checkpoint_and_dataset(diff_ckpt)
        diff_results = run_inference(diff_policy, diff_dataset, num_samples=5, device=device)
        
        visualize_predictions_vs_gt(
            diff_results,
            os.path.join(output_dir, 'pred_diffusion.png'),
            model_name="Diffusion Policy"
        )
        diff_errors = compute_errors(diff_results)
        
        del diff_policy
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"Diffusion error: {e}")
        import traceback
        traceback.print_exc()
        diff_results = None
        diff_errors = None
    
    # Comparison figures
    if bfn_results and diff_results:
        print("\n--- Comparison ---")
        visualize_error_analysis(
            bfn_errors, diff_errors,
            os.path.join(output_dir, 'pred_error_comparison.png')
        )
        
        visualize_overlay(
            bfn_results, diff_results,
            os.path.join(output_dir, 'pred_overlay.png')
        )
    
    # Copy to figures_for_professor
    import shutil
    dest_dir = '$PROJECT_ROOT/figures_for_professor'
    for f in ['pred_bfn.png', 'pred_diffusion.png', 'pred_error_comparison.png', 'pred_overlay.png']:
        src = os.path.join(output_dir, f)
        if os.path.exists(src):
            shutil.copy(src, dest_dir)
    
    print("\n" + "=" * 50)
    print("Done! Figures saved to:")
    print(f"  {output_dir}/")
    print(f"  {dest_dir}/")
    print("=" * 50)


if __name__ == '__main__':
    main()
