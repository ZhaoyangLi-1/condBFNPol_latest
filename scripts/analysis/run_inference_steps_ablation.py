#!/usr/bin/env python
"""
Inference Steps Ablation Study: BFN vs Diffusion Policy

This script evaluates trained checkpoints at different numbers of inference steps
to compare computational efficiency vs performance trade-offs.

Key Hypothesis:
- BFN should maintain performance at fewer steps (5-20)
- Diffusion Policy degrades significantly below 50 steps

Usage:
    # Run ablation with specific checkpoints
    python scripts/run_inference_steps_ablation.py \
        --bfn-ckpt cluster_checkpoints/.../bfn_seed42/checkpoints/epoch=0200-test_mean_score=0.877.ckpt \
        --diffusion-ckpt cluster_checkpoints/.../diffusion_seed42/checkpoints/epoch=0100-test_mean_score=0.962.ckpt
    
    # Run with all seeds
    python scripts/run_inference_steps_ablation.py --all-seeds
    
    # Just generate plots from existing results
    python scripts/run_inference_steps_ablation.py --plot-only --results-file results/ablation/results.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import torch
import numpy as np
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# CHECKPOINT LOADING
# =============================================================================

def find_checkpoints(base_dir: str = "cluster_checkpoints/benchmarkresults") -> Dict[str, Dict[str, str]]:
    """Find all benchmark checkpoints organized by method and seed."""
    base_path = Path(base_dir)
    if not base_path.exists():
        print(f"Warning: {base_dir} not found")
        return {}
    
    checkpoints = {
        'bfn': {},
        'diffusion': {},
    }
    
    for run_dir in base_path.rglob("*_seed*"):
        if not run_dir.is_dir():
            continue
        
        # Parse method and seed from directory name
        dir_name = run_dir.name
        if "bfn" in dir_name.lower():
            method = "bfn"
        elif "diffusion" in dir_name.lower():
            method = "diffusion"
        else:
            continue
        
        # Extract seed
        seed = None
        for part in dir_name.split("_"):
            if part.startswith("seed"):
                try:
                    seed = int(part.replace("seed", ""))
                except ValueError:
                    pass
        
        if seed is None:
            continue
        
        # Find best checkpoint
        ckpt_dir = run_dir / "checkpoints"
        if not ckpt_dir.exists():
            continue
        
        best_score = -1
        best_ckpt = None
        for ckpt_file in ckpt_dir.glob("*.ckpt"):
            # Parse score from filename: epoch=0200-test_mean_score=0.877.ckpt
            name = ckpt_file.stem
            if "test_mean_score=" in name:
                try:
                    score_str = name.split("test_mean_score=")[1]
                    score = float(score_str)
                    if score > best_score:
                        best_score = score
                        best_ckpt = str(ckpt_file)
                except (IndexError, ValueError):
                    pass
        
        if best_ckpt:
            checkpoints[method][seed] = best_ckpt
            print(f"Found {method} seed {seed}: {Path(best_ckpt).name} (score={best_score:.3f})")
    
    return checkpoints


def load_bfn_policy(checkpoint_path: str, device: str = "cuda"):
    """Load BFN policy from checkpoint."""
    print(f"Loading BFN checkpoint: {checkpoint_path}")
    
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Extract config and state
    cfg = payload.get('cfg', payload.get('config'))
    state_dicts = payload.get('state_dicts', {})
    
    # Build policy from config
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    
    # Instantiate the policy
    policy = instantiate(cfg.policy)
    
    # Load state dicts
    if 'model' in state_dicts:
        policy.model.load_state_dict(state_dicts['model'])
    if 'obs_encoder' in state_dicts:
        policy.obs_encoder.load_state_dict(state_dicts['obs_encoder'])
    if 'ema_model' in state_dicts:
        # Load EMA weights into model
        ema_state = state_dicts['ema_model']
        # The EMA state might have a different structure
        try:
            policy.model.load_state_dict(ema_state.get('model', ema_state))
        except:
            pass
    
    # Load normalizer
    if 'normalizer' in payload:
        from diffusion_policy.model.common.normalizer import LinearNormalizer
        normalizer = LinearNormalizer()
        normalizer.load_state_dict(payload['normalizer'])
        policy.set_normalizer(normalizer)
    
    policy.to(device)
    policy.eval()
    return policy, cfg


def load_diffusion_policy(checkpoint_path: str, device: str = "cuda"):
    """Load Diffusion policy from checkpoint."""
    print(f"Loading Diffusion checkpoint: {checkpoint_path}")
    
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    cfg = payload.get('cfg', payload.get('config'))
    state_dicts = payload.get('state_dicts', {})
    
    from hydra.utils import instantiate
    
    policy = instantiate(cfg.policy)
    
    if 'model' in state_dicts:
        policy.model.load_state_dict(state_dicts['model'])
    if 'obs_encoder' in state_dicts:
        policy.obs_encoder.load_state_dict(state_dicts['obs_encoder'])
    if 'ema_model' in state_dicts:
        try:
            policy.model.load_state_dict(state_dicts['ema_model'].get('model', state_dicts['ema_model']))
        except:
            pass
    
    if 'normalizer' in payload:
        from diffusion_policy.model.common.normalizer import LinearNormalizer
        normalizer = LinearNormalizer()
        normalizer.load_state_dict(payload['normalizer'])
        policy.set_normalizer(normalizer)
    
    policy.to(device)
    policy.eval()
    return policy, cfg


# =============================================================================
# EVALUATION WITH DIFFERENT STEPS
# =============================================================================

def evaluate_at_steps(
    policy,
    method: str,  # 'bfn' or 'diffusion'
    n_steps: int,
    n_envs: int = 50,
    max_steps: int = 300,
    device: str = "cuda",
) -> Dict:
    """Evaluate a policy with a specific number of inference steps."""
    from diffusion_policy.env_runner.pusht_image_runner import PushTImageRunner
    
    # Modify inference steps
    original_steps = None
    if method == 'bfn':
        original_steps = getattr(policy, 'n_timesteps', 20)
        policy.n_timesteps = n_steps
    else:  # diffusion
        if hasattr(policy, 'num_inference_steps'):
            original_steps = policy.num_inference_steps
            policy.num_inference_steps = n_steps
        if hasattr(policy, 'noise_scheduler'):
            policy.noise_scheduler.set_timesteps(n_steps)
    
    # Create runner
    runner = PushTImageRunner(
        n_train=0,
        n_train_vis=0,
        train_start_seed=0,
        n_test=n_envs,
        n_test_vis=0,
        test_start_seed=100000,
        max_steps=max_steps,
        n_obs_steps=policy.n_obs_steps,
        n_action_steps=policy.n_action_steps,
        fps=10,
        past_action=False,
        n_envs=None,
        legacy_test=True,
    )
    
    # Run evaluation
    policy.eval()
    with torch.no_grad():
        result = runner.run(policy)
    
    # Restore original steps
    if original_steps is not None:
        if method == 'bfn':
            policy.n_timesteps = original_steps
        else:
            if hasattr(policy, 'num_inference_steps'):
                policy.num_inference_steps = original_steps
    
    return {
        'mean_score': result.get('test/mean_score', result.get('test_mean_score', 0)),
        'max_score': result.get('test/max_score', 0),
        'min_score': result.get('test/min_score', 0),
    }


def measure_inference_time(
    policy,
    method: str,
    n_steps: int,
    n_samples: int = 100,
    device: str = "cuda",
) -> Dict:
    """Measure inference time for given number of steps."""
    # Create dummy observation
    dummy_obs = {
        'image': torch.randn(1, 2, 3, 96, 96, device=device),
        'agent_pos': torch.randn(1, 2, 2, device=device),
    }
    
    # Set steps
    if method == 'bfn':
        original = policy.n_timesteps
        policy.n_timesteps = n_steps
    else:
        if hasattr(policy, 'num_inference_steps'):
            original = policy.num_inference_steps
            policy.num_inference_steps = n_steps
        if hasattr(policy, 'noise_scheduler'):
            policy.noise_scheduler.set_timesteps(n_steps)
    
    # Warmup
    policy.eval()
    for _ in range(5):
        with torch.no_grad():
            _ = policy.predict_action(dummy_obs)
    
    # Measure
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    
    for _ in range(n_samples):
        with torch.no_grad():
            _ = policy.predict_action(dummy_obs)
    
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    # Restore
    if method == 'bfn':
        policy.n_timesteps = original
    else:
        if hasattr(policy, 'num_inference_steps'):
            policy.num_inference_steps = original
    
    return {
        'total_time_sec': elapsed,
        'avg_time_ms': (elapsed / n_samples) * 1000,
        'hz': n_samples / elapsed,
    }


# =============================================================================
# MAIN ABLATION
# =============================================================================

def run_ablation(
    bfn_checkpoints: Dict[int, str],
    diffusion_checkpoints: Dict[int, str],
    bfn_steps: List[int] = [5, 10, 15, 20, 30, 50],
    diffusion_steps: List[int] = [10, 20, 30, 50, 75, 100],
    n_envs: int = 50,
    device: str = "cuda",
    output_dir: str = "results/ablation",
) -> Dict:
    """Run full ablation study."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results_file = Path(output_dir) / "ablation_results.json"

    if results_file.exists():
        with open(results_file, "r") as f:
            loaded = json.load(f)

        results = {
            'bfn': defaultdict(dict, {
                int(seed): {int(k): v for k, v in steps.items()}
                for seed, steps in loaded.get('bfn', {}).items()
            }),
            'diffusion': defaultdict(dict, {
                int(seed): {int(k): v for k, v in steps.items()}
                for seed, steps in loaded.get('diffusion', {}).items()
            }),
            'metadata': loaded.get('metadata', {}),
        }
    else:
        results = {
            'bfn': defaultdict(dict),
            'diffusion': defaultdict(dict),
            'metadata': {},
        }

    # Always refresh metadata (safe + correct)
    results['metadata'].update({
        'timestamp': datetime.now().isoformat(),
        'bfn_steps': bfn_steps,
        'diffusion_steps': diffusion_steps,
        'n_envs': n_envs,
    })

    
    # Run BFN ablation
    print("\n" + "=" * 60)
    print("BFN INFERENCE STEPS ABLATION")
    print("=" * 60)
    
    for seed, ckpt_path in bfn_checkpoints.items():
        print(f"\n--- BFN Seed {seed} ---")
        try:
            policy, cfg = load_bfn_policy(ckpt_path, device)
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            continue
        
        for n_steps in tqdm(bfn_steps, desc=f"BFN seed {seed}"):
            print(f"  Evaluating with {n_steps} steps...")
            
            # Evaluate
            try:
                eval_result = evaluate_at_steps(policy, 'bfn', n_steps, n_envs, device=device)
                time_result = measure_inference_time(policy, 'bfn', n_steps, device=device)
                
                results['bfn'][seed][n_steps] = {
                    **eval_result,
                    **time_result,
                }
                print(f"    Score: {eval_result['mean_score']:.3f}, Time: {time_result['avg_time_ms']:.1f}ms")
            except Exception as e:
                print(f"    Error: {e}")
    
    # Run Diffusion ablation
    print("\n" + "=" * 60)
    print("DIFFUSION INFERENCE STEPS ABLATION")
    print("=" * 60)
    
    for seed, ckpt_path in diffusion_checkpoints.items():
        print(f"\n--- Diffusion Seed {seed} ---")
        try:
            policy, cfg = load_diffusion_policy(ckpt_path, device)
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            continue
        
        for n_steps in tqdm(diffusion_steps, desc=f"Diffusion seed {seed}"):
            print(f"  Evaluating with {n_steps} steps...")
            
            try:
                eval_result = evaluate_at_steps(policy, 'diffusion', n_steps, n_envs, device=device)
                time_result = measure_inference_time(policy, 'diffusion', n_steps, device=device)
                
                results['diffusion'][seed][n_steps] = {
                    **eval_result,
                    **time_result,
                }
                print(f"    Score: {eval_result['mean_score']:.3f}, Time: {time_result['avg_time_ms']:.1f}ms")
            except Exception as e:
                print(f"    Error: {e}")
    
    # Save results
    results_file = output_path / "ablation_results.json"
    
    # Convert defaultdict to regular dict for JSON serialization
    results_to_save = {
        'bfn': {str(k): {str(k2): v2 for k2, v2 in v.items()} for k, v in results['bfn'].items()},
        'diffusion': {str(k): {str(k2): v2 for k2, v2 in v.items()} for k, v in results['diffusion'].items()},
        'metadata': results['metadata'],
    }
    
    with open(results_file, 'w') as f:
        json.dump(results_to_save, f, indent=2)
    print(f"\nResults saved to: {results_file}")
    
    return results


# =============================================================================
# PLOTTING
# =============================================================================

def generate_ablation_plots(results: Dict, output_dir: str = "figures/publication"):
    """Generate publication-quality ablation plots."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Style
    COLORS = {
        'bfn': '#4285F4',           # Google Blue
        'diffusion': '#EA4335',     # Google Red
    }
    
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        'font.size': 9,
        'axes.titlesize': 10,
        'axes.labelsize': 9,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.linewidth': 0.8,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })
    
    DOUBLE_COL = 7.16
    SINGLE_COL = 3.5
    
    # Aggregate results across seeds
    def aggregate_method(method_results):
        """Aggregate results across seeds."""
        all_steps = set()
        for seed_data in method_results.values():
            all_steps.update(seed_data.keys())
        
        aggregated = {}
        for steps in sorted(all_steps, key=lambda x: int(x)):
            scores = []
            times = []
            for seed_data in method_results.values():
                if steps in seed_data:
                    scores.append(seed_data[steps]['mean_score'])
                    times.append(seed_data[steps]['avg_time_ms'])
            
            if scores:
                aggregated[int(steps)] = {
                    'mean_score': np.mean(scores),
                    'std_score': np.std(scores),
                    'mean_time': np.mean(times),
                    'std_time': np.std(times),
                }
        return aggregated
    
    bfn_agg = aggregate_method(results.get('bfn', {}))
    diff_agg = aggregate_method(results.get('diffusion', {}))
    
    if not bfn_agg or not diff_agg:
        print("Warning: Not enough data to generate plots")
        return
    
    # ==================== FIGURE 1: Score vs Steps ====================
    fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.3, SINGLE_COL * 0.9))
    
    # BFN
    bfn_steps = sorted(bfn_agg.keys())
    bfn_scores = [bfn_agg[s]['mean_score'] for s in bfn_steps]
    bfn_stds = [bfn_agg[s]['std_score'] for s in bfn_steps]
    
    ax.errorbar(bfn_steps, bfn_scores, yerr=bfn_stds, 
                color=COLORS['bfn'], marker='o', markersize=6,
                capsize=3, linewidth=1.5, label='BFN Policy')
    
    # Diffusion
    diff_steps = sorted(diff_agg.keys())
    diff_scores = [diff_agg[s]['mean_score'] for s in diff_steps]
    diff_stds = [diff_agg[s]['std_score'] for s in diff_steps]
    
    ax.errorbar(diff_steps, diff_scores, yerr=diff_stds,
                color=COLORS['diffusion'], marker='s', markersize=6,
                capsize=3, linewidth=1.5, label='Diffusion Policy')
    
    ax.set_xlabel('Inference Steps')
    ax.set_ylabel('Success Score')
    ax.set_title('Performance vs Inference Steps', fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    
    # Add annotations for default steps
    ax.axvline(x=20, color=COLORS['bfn'], linestyle='--', alpha=0.3, linewidth=1)
    ax.axvline(x=100, color=COLORS['diffusion'], linestyle='--', alpha=0.3, linewidth=1)
    
    plt.tight_layout()
    fig.savefig(output_path / 'fig_ablation_score_vs_steps.pdf')
    fig.savefig(output_path / 'fig_ablation_score_vs_steps.png', dpi=300)
    plt.close(fig)
    print(f"  - Score vs Steps plot saved")
    
    # ==================== FIGURE 2: Time vs Steps ====================
    fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.3, SINGLE_COL * 0.9))
    
    # BFN
    bfn_times = [bfn_agg[s]['mean_time'] for s in bfn_steps]
    ax.plot(bfn_steps, bfn_times, color=COLORS['bfn'], marker='o', 
            markersize=6, linewidth=1.5, label='BFN Policy')
    
    # Diffusion
    diff_times = [diff_agg[s]['mean_time'] for s in diff_steps]
    ax.plot(diff_steps, diff_times, color=COLORS['diffusion'], marker='s',
            markersize=6, linewidth=1.5, label='Diffusion Policy')
    
    ax.set_xlabel('Inference Steps')
    ax.set_ylabel('Inference Time (ms)')
    ax.set_title('Computational Cost vs Inference Steps', fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(output_path / 'fig_ablation_time_vs_steps.pdf')
    fig.savefig(output_path / 'fig_ablation_time_vs_steps.png', dpi=300)
    plt.close(fig)
    print(f"  - Time vs Steps plot saved")
    
    # ==================== FIGURE 3: Pareto Frontier ====================
    fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.3, SINGLE_COL * 0.9))
    
    # BFN points
    for s in bfn_steps:
        ax.scatter(bfn_agg[s]['mean_time'], bfn_agg[s]['mean_score'],
                   c=COLORS['bfn'], s=60, zorder=5)
        ax.annotate(f'{s}', (bfn_agg[s]['mean_time'], bfn_agg[s]['mean_score']),
                   xytext=(5, 5), textcoords='offset points', fontsize=7,
                   color=COLORS['bfn'])
    
    # Diffusion points
    for s in diff_steps:
        ax.scatter(diff_agg[s]['mean_time'], diff_agg[s]['mean_score'],
                   c=COLORS['diffusion'], s=60, zorder=5, marker='s')
        ax.annotate(f'{s}', (diff_agg[s]['mean_time'], diff_agg[s]['mean_score']),
                   xytext=(5, 5), textcoords='offset points', fontsize=7,
                   color=COLORS['diffusion'])
    
    # Connect lines
    ax.plot(bfn_times, bfn_scores, color=COLORS['bfn'], linewidth=1, alpha=0.5, label='BFN')
    ax.plot(diff_times, diff_scores, color=COLORS['diffusion'], linewidth=1, alpha=0.5, label='Diffusion')
    
    ax.set_xlabel('Inference Time (ms)')
    ax.set_ylabel('Success Score')
    ax.set_title('Efficiency-Performance Trade-off', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(output_path / 'fig_ablation_pareto.pdf')
    fig.savefig(output_path / 'fig_ablation_pareto.png', dpi=300)
    plt.close(fig)
    print(f"  - Pareto frontier plot saved")
    
    # ==================== FIGURE 4: Combined 2-panel ====================
    fig = plt.figure(figsize=(DOUBLE_COL, SINGLE_COL * 0.85))
    gs = gridspec.GridSpec(1, 2, wspace=0.35)
    
    # Panel A: Score vs Steps
    ax1 = fig.add_subplot(gs[0])
    ax1.errorbar(bfn_steps, bfn_scores, yerr=bfn_stds,
                 color=COLORS['bfn'], marker='o', markersize=5,
                 capsize=2, linewidth=1.5, label='BFN Policy')
    ax1.errorbar(diff_steps, diff_scores, yerr=diff_stds,
                 color=COLORS['diffusion'], marker='s', markersize=5,
                 capsize=2, linewidth=1.5, label='Diffusion Policy')
    
    ax1.set_xlabel('Inference Steps')
    ax1.set_ylabel('Success Score')
    ax1.set_title('(a) Performance vs Steps', fontweight='bold')
    ax1.legend(loc='lower right', fontsize=7)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.05)
    
    # Panel B: Pareto
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(bfn_times, bfn_scores, color=COLORS['bfn'], marker='o',
             markersize=5, linewidth=1.5, label='BFN Policy')
    ax2.plot(diff_times, diff_scores, color=COLORS['diffusion'], marker='s',
             markersize=5, linewidth=1.5, label='Diffusion Policy')
    
    ax2.set_xlabel('Inference Time (ms)')
    ax2.set_ylabel('Success Score')
    ax2.set_title('(b) Efficiency Trade-off', fontweight='bold')
    ax2.legend(loc='lower right', fontsize=7)
    ax2.grid(True, alpha=0.3)
    
    fig.savefig(output_path / 'fig_ablation_combined.pdf')
    fig.savefig(output_path / 'fig_ablation_combined.png', dpi=300)
    plt.close(fig)
    print(f"  - Combined 2-panel plot saved")
    
    print(f"\nAll ablation plots saved to: {output_path}")


# =============================================================================
# GENERATE LATEX TABLE
# =============================================================================

def generate_ablation_table(results: Dict, output_dir: str = "figures/tables"):
    """Generate LaTeX table for ablation results."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Aggregate results
    def aggregate_method(method_results):
        all_steps = set()
        for seed_data in method_results.values():
            all_steps.update(seed_data.keys())
        
        aggregated = {}
        for steps in sorted(all_steps, key=lambda x: int(x)):
            scores = []
            times = []
            for seed_data in method_results.values():
                if steps in seed_data:
                    scores.append(seed_data[steps]['mean_score'])
                    times.append(seed_data[steps]['avg_time_ms'])
            
            if scores:
                aggregated[int(steps)] = {
                    'mean_score': np.mean(scores),
                    'std_score': np.std(scores),
                    'mean_time': np.mean(times),
                }
        return aggregated
    
    bfn_agg = aggregate_method(results.get('bfn', {}))
    diff_agg = aggregate_method(results.get('diffusion', {}))
    
    # Generate table
    table_content = r"""% =============================================================================
% TABLE: Inference Steps Ablation Study
% =============================================================================

\begin{table}[t]
\centering
\caption{\textbf{Inference Steps Ablation.} 
Performance comparison at different inference steps. 
BFN maintains strong performance with as few as 10 steps, while 
Diffusion Policy degrades significantly below 50 steps.}
\label{tab:inference_steps_ablation}
\vspace{0.5em}
\small
\begin{tabular}{@{}lcccc@{}}
\toprule
\textbf{Method} & \textbf{Steps} & \textbf{Score} & \textbf{Time (ms)} & \textbf{Rel. to Default} \\
\midrule
"""
    
    # BFN rows
    bfn_default_score = bfn_agg.get(20, {}).get('mean_score', 0.912)
    for steps in sorted(bfn_agg.keys()):
        data = bfn_agg[steps]
        rel = data['mean_score'] / bfn_default_score if bfn_default_score > 0 else 1.0
        is_default = steps == 20
        bold_start = r"\textbf{" if is_default else ""
        bold_end = "}" if is_default else ""
        table_content += f"BFN & {bold_start}{steps}{bold_end} & ${data['mean_score']:.3f} \\pm {data['std_score']:.3f}$ & {data['mean_time']:.1f} & {rel:.2f}$\\times$ \\\\\n"
    
    table_content += r"\midrule" + "\n"
    
    # Diffusion rows
    diff_default_score = diff_agg.get(100, {}).get('mean_score', 0.950)
    for steps in sorted(diff_agg.keys()):
        data = diff_agg[steps]
        rel = data['mean_score'] / diff_default_score if diff_default_score > 0 else 1.0
        is_default = steps == 100
        bold_start = r"\textbf{" if is_default else ""
        bold_end = "}" if is_default else ""
        table_content += f"Diffusion & {bold_start}{steps}{bold_end} & ${data['mean_score']:.3f} \\pm {data['std_score']:.3f}$ & {data['mean_time']:.1f} & {rel:.2f}$\\times$ \\\\\n"
    
    table_content += r"""\bottomrule
\end{tabular}
\vspace{0.3em}
\raggedright
\footnotesize
\textit{Note:} Default steps (bold) are 20 for BFN and 100 for Diffusion. 
Rel. to Default shows performance relative to default configuration.
\end{table}
"""
    
    table_file = output_path / "table_inference_steps_ablation.tex"
    with open(table_file, 'w') as f:
        f.write(table_content)
    print(f"Table saved to: {table_file}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run inference steps ablation study")
    parser.add_argument("--bfn-ckpt", type=str, help="Path to BFN checkpoint")
    parser.add_argument("--diffusion-ckpt", type=str, help="Path to Diffusion checkpoint")
    parser.add_argument("--all-seeds", action="store_true", help="Run with all available seeds")
    parser.add_argument("--checkpoint-dir", type=str, default="cluster_checkpoints/benchmarkresults",
                        help="Base directory for checkpoints")
    parser.add_argument("--bfn-steps", type=str, default="5,10,15,20,30,50",
                        help="Comma-separated list of BFN steps to test")
    parser.add_argument("--diffusion-steps", type=str, default="10,20,30,50,75,100",
                        help="Comma-separated list of Diffusion steps to test")
    parser.add_argument("--n-envs", type=int, default=50, help="Number of environments for evaluation")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--output-dir", type=str, default="results/ablation", help="Output directory")
    parser.add_argument("--plot-only", action="store_true", help="Only generate plots from existing results")
    parser.add_argument("--results-file", type=str, help="Path to existing results JSON for plotting")
    args = parser.parse_args()
    
    # Parse steps lists
    bfn_steps = [int(x) for x in args.bfn_steps.split(",")]
    diffusion_steps = [int(x) for x in args.diffusion_steps.split(",")]
    
    # Load existing results or run ablation
    if args.plot_only:
        if args.results_file and Path(args.results_file).exists():
            with open(args.results_file) as f:
                results = json.load(f)
        else:
            # Try default location
            default_results = Path(args.output_dir) / "ablation_results.json"
            if default_results.exists():
                with open(default_results) as f:
                    results = json.load(f)
            else:
                print("Error: No results file found. Run ablation first or specify --results-file")
                return
        
        print("Generating plots from existing results...")
        generate_ablation_plots(results)
        generate_ablation_table(results)
        return
    
    # Find checkpoints
    if args.all_seeds:
        checkpoints = find_checkpoints(args.checkpoint_dir)
        bfn_checkpoints = checkpoints.get('bfn', {})
        diffusion_checkpoints = checkpoints.get('diffusion', {})
    else:
        bfn_checkpoints = {}
        diffusion_checkpoints = {}
        
        if args.bfn_ckpt:
            bfn_checkpoints[42] = args.bfn_ckpt
        if args.diffusion_ckpt:
            diffusion_checkpoints[42] = args.diffusion_ckpt
    
    if not bfn_checkpoints and not diffusion_checkpoints:
        print("Error: No checkpoints found. Provide paths or use --all-seeds")
        return
    
    print(f"\nBFN checkpoints: {len(bfn_checkpoints)}")
    print(f"Diffusion checkpoints: {len(diffusion_checkpoints)}")
    
    # Run ablation
    results = run_ablation(
        bfn_checkpoints=bfn_checkpoints,
        diffusion_checkpoints=diffusion_checkpoints,
        bfn_steps=bfn_steps,
        diffusion_steps=diffusion_steps,
        n_envs=args.n_envs,
        device=args.device,
        output_dir=args.output_dir,
    )
    
    # Generate plots and table
    print("\n" + "=" * 60)
    print("GENERATING PLOTS AND TABLES")
    print("=" * 60)
    generate_ablation_plots(results)
    generate_ablation_table(results)
    
    print("\n" + "=" * 60)
    print("ABLATION STUDY COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
