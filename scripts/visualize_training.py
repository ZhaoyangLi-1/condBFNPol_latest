#!/usr/bin/env python3
"""
Visualize training results: BFN vs Diffusion Policy comparison.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

# Style setup
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (14, 5)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['legend.fontsize'] = 10

def load_logs(log_path):
    """Load training logs from JSON lines file."""
    logs = []
    with open(log_path, 'r') as f:
        for line in f:
            try:
                logs.append(json.loads(line.strip()))
            except:
                continue
    return logs

def aggregate_by_epoch(logs):
    """Aggregate per-step losses to per-epoch mean."""
    epoch_losses = defaultdict(list)
    for entry in logs:
        epoch_losses[entry['epoch']].append(entry['train_loss'])
    
    epochs = sorted(epoch_losses.keys())
    mean_losses = [np.mean(epoch_losses[e]) for e in epochs]
    return np.array(epochs), np.array(mean_losses)

def smooth(y, window=10):
    """Exponential moving average smoothing."""
    alpha = 2 / (window + 1)
    smoothed = np.zeros_like(y)
    smoothed[0] = y[0]
    for i in range(1, len(y)):
        smoothed[i] = alpha * y[i] + (1 - alpha) * smoothed[i-1]
    return smoothed

def main():
    outputs_dir = Path('$PROJECT_ROOT/outputs')
    results_dir = Path('$PROJECT_ROOT/results')
    results_dir.mkdir(exist_ok=True)
    
    # Load all logs
    bfn_data = {}
    diffusion_data = {}
    
    for run_dir in outputs_dir.iterdir():
        if not run_dir.is_dir():
            continue
        
        log_file = run_dir / 'logs.json.txt'
        if not log_file.exists():
            continue
        
        name = run_dir.name
        logs = load_logs(log_file)
        epochs, losses = aggregate_by_epoch(logs)
        
        if 'bfn' in name:
            seed = name.split('seed')[-1]
            bfn_data[seed] = (epochs, losses)
            print(f"Loaded BFN seed {seed}: {len(epochs)} epochs, final loss={losses[-1]:.6f}")
        elif 'diffusion' in name:
            seed = name.split('seed')[-1]
            diffusion_data[seed] = (epochs, losses)
            print(f"Loaded Diffusion seed {seed}: {len(epochs)} epochs, final loss={losses[-1]:.6f}")
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors_bfn = ['#2ecc71', '#27ae60', '#1e8449']  # Greens
    colors_diff = ['#e74c3c', '#c0392b', '#922b21']  # Reds
    
    # ===== Plot 1: Individual runs =====
    ax1 = axes[0]
    
    for i, (seed, (epochs, losses)) in enumerate(sorted(bfn_data.items())):
        ax1.plot(epochs, smooth(losses, 20), color=colors_bfn[i], 
                 alpha=0.8, linewidth=1.5, label=f'BFN seed {seed}')
    
    for i, (seed, (epochs, losses)) in enumerate(sorted(diffusion_data.items())):
        ax1.plot(epochs, smooth(losses, 20), color=colors_diff[i], 
                 alpha=0.8, linewidth=1.5, label=f'Diffusion seed {seed}')
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Training Loss Comparison: All Seeds')
    ax1.legend(loc='upper right', ncol=2)
    ax1.set_yscale('log')
    ax1.set_xlim(0, 300)
    ax1.grid(True, alpha=0.3)
    
    # ===== Plot 2: Mean ± std =====
    ax2 = axes[1]
    
    # Find common epoch range
    min_epochs = min(len(d[0]) for d in bfn_data.values())
    min_epochs_diff = min(len(d[0]) for d in diffusion_data.values())
    
    # BFN aggregate
    bfn_losses_aligned = np.array([smooth(d[1][:min_epochs], 20) for d in bfn_data.values()])
    bfn_mean = bfn_losses_aligned.mean(axis=0)
    bfn_std = bfn_losses_aligned.std(axis=0)
    bfn_epochs = list(bfn_data.values())[0][0][:min_epochs]
    
    ax2.plot(bfn_epochs, bfn_mean, color='#2ecc71', linewidth=2.5, label='BFN Policy (Ours)')
    ax2.fill_between(bfn_epochs, bfn_mean - bfn_std, bfn_mean + bfn_std, 
                     color='#2ecc71', alpha=0.2)
    
    # Diffusion aggregate
    diff_losses_aligned = np.array([smooth(d[1][:min_epochs_diff], 20) for d in diffusion_data.values()])
    diff_mean = diff_losses_aligned.mean(axis=0)
    diff_std = diff_losses_aligned.std(axis=0)
    diff_epochs = list(diffusion_data.values())[0][0][:min_epochs_diff]
    
    ax2.plot(diff_epochs, diff_mean, color='#e74c3c', linewidth=2.5, label='Diffusion Policy')
    ax2.fill_between(diff_epochs, diff_mean - diff_std, diff_mean + diff_std, 
                     color='#e74c3c', alpha=0.2)
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Training Loss')
    ax2.set_title('Training Loss: Mean ± Std (3 seeds)')
    ax2.legend(loc='upper right')
    ax2.set_yscale('log')
    ax2.set_xlim(0, 300)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(results_dir / 'training_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(results_dir / 'training_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved: {results_dir / 'training_comparison.pdf'}")
    
    # ===== Print summary statistics =====
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    
    print("\nBFN Policy:")
    for seed, (epochs, losses) in sorted(bfn_data.items()):
        print(f"  Seed {seed}: {len(epochs)} epochs, final loss = {losses[-1]:.6f}")
    print(f"  Mean final loss: {np.mean([d[1][-1] for d in bfn_data.values()]):.6f}")
    
    print("\nDiffusion Policy:")
    for seed, (epochs, losses) in sorted(diffusion_data.items()):
        print(f"  Seed {seed}: {len(epochs)} epochs, final loss = {losses[-1]:.6f}")
    print(f"  Mean final loss: {np.mean([d[1][-1] for d in diffusion_data.values()]):.6f}")
    
    # ===== Early convergence analysis =====
    print("\n" + "="*60)
    print("CONVERGENCE ANALYSIS (Loss at specific epochs)")
    print("="*60)
    
    checkpoints = [50, 100, 150, 200, 250]
    
    print("\nBFN Policy:")
    for ep in checkpoints:
        losses_at_ep = []
        for seed, (epochs, losses) in bfn_data.items():
            if ep < len(losses):
                losses_at_ep.append(losses[ep])
        if losses_at_ep:
            print(f"  Epoch {ep:3d}: {np.mean(losses_at_ep):.6f} ± {np.std(losses_at_ep):.6f}")
    
    print("\nDiffusion Policy:")
    for ep in checkpoints:
        losses_at_ep = []
        for seed, (epochs, losses) in diffusion_data.items():
            if ep < len(losses):
                losses_at_ep.append(losses[ep])
        if losses_at_ep:
            print(f"  Epoch {ep:3d}: {np.mean(losses_at_ep):.6f} ± {np.std(losses_at_ep):.6f}")

if __name__ == '__main__':
    main()
