#!/usr/bin/env python3
"""
Analyze hybrid discrete action experiments.

This script compares BFN-Policy (native discrete) vs Diffusion Policy 
(continuous relaxation) on RoboMimic tasks with gripper actions.

Key metrics:
1. Overall success rate (task completion)
2. Gripper accuracy (open/close at correct times)
3. Gripper decision confidence
4. Inference time

Usage:
    python scripts/analyze_hybrid_experiments.py --task lift
    python scripts/analyze_hybrid_experiments.py --task lift --compare bfn_hybrid diffusion
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Anthropic color palette
COLORS = {
    'teal': '#2D8B8B',
    'coral': '#D97757',
    'slate': '#3D4F5F',
}


def load_results(output_dir: str, method: str, task: str) -> Dict:
    """Load experiment results from wandb logs or checkpoint directories."""
    results = {
        'success_rates': [],
        'gripper_accuracy': [],
        'inference_time_ms': [],
        'seeds': []
    }
    
    # Pattern: outputs/YYYY.MM.DD/HH.MM.SS_method_task/
    pattern = f"*_{method}*_{task}*"
    
    for run_dir in Path(output_dir).glob(f"**/{pattern}"):
        logs_file = run_dir / "logs.json.txt"
        if logs_file.exists():
            # Parse the last line for final metrics
            with open(logs_file) as f:
                lines = f.readlines()
                if lines:
                    try:
                        final_metrics = json.loads(lines[-1])
                        if 'test_mean_score' in final_metrics:
                            results['success_rates'].append(final_metrics['test_mean_score'])
                        # Extract seed from directory name
                        seed = run_dir.name.split('seed')[-1].split('_')[0] if 'seed' in run_dir.name else '?'
                        results['seeds'].append(seed)
                    except json.JSONDecodeError:
                        continue
    
    return results


def compute_statistics(values: List[float]) -> Tuple[float, float]:
    """Compute mean and standard error."""
    if not values:
        return 0.0, 0.0
    mean = np.mean(values)
    std = np.std(values, ddof=1) if len(values) > 1 else 0.0
    return mean, std


def welch_t_test(group1: List[float], group2: List[float]) -> Tuple[float, float]:
    """Perform Welch's t-test for two groups with unequal variance."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0, 1.0
    
    m1, m2 = np.mean(group1), np.mean(group2)
    v1, v2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    t_stat = (m1 - m2) / np.sqrt(v1/n1 + v2/n2)
    
    # Approximate p-value using normal distribution for simplicity
    from scipy import stats
    df = ((v1/n1 + v2/n2)**2) / ((v1/n1)**2/(n1-1) + (v2/n2)**2/(n2-1))
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
    
    return t_stat, p_value


def print_comparison_table(bfn_results: Dict, diff_results: Dict, task: str):
    """Print a comparison table in the style of the thesis."""
    bfn_mean, bfn_std = compute_statistics(bfn_results['success_rates'])
    diff_mean, diff_std = compute_statistics(diff_results['success_rates'])
    
    print("\n" + "=" * 70)
    print(f"HYBRID ACTION SPACE COMPARISON: {task.upper()}")
    print("=" * 70)
    print(f"\n{'Method':<25} {'Success Rate':<20} {'n_seeds':<10}")
    print("-" * 55)
    print(f"{'BFN-Policy (Hybrid)':<25} {bfn_mean*100:.1f}% ± {bfn_std*100:.1f}%    {len(bfn_results['success_rates'])}")
    print(f"{'Diffusion (Relaxation)':<25} {diff_mean*100:.1f}% ± {diff_std*100:.1f}%    {len(diff_results['success_rates'])}")
    print("-" * 55)
    
    # Statistical test
    if len(bfn_results['success_rates']) >= 2 and len(diff_results['success_rates']) >= 2:
        t_stat, p_value = welch_t_test(
            bfn_results['success_rates'], 
            diff_results['success_rates']
        )
        diff = (bfn_mean - diff_mean) * 100
        print(f"\nDifference: {diff:+.1f} percentage points")
        print(f"Welch's t-test: t={t_stat:.2f}, p={p_value:.3f}")
        
        if p_value < 0.05:
            winner = "BFN-Policy" if diff > 0 else "Diffusion"
            print(f"Result: {winner} significantly better (p < 0.05)")
        else:
            print("Result: No significant difference (p ≥ 0.05)")
    
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Analyze hybrid action experiments")
    parser.add_argument("--task", type=str, default="lift", 
                        choices=["lift", "can", "square"])
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--compare", nargs="+", default=["bfn_hybrid", "diffusion"])
    args = parser.parse_args()
    
    print(f"\nAnalyzing experiments for task: {args.task}")
    print(f"Output directory: {args.output_dir}")
    
    # Load results for each method
    results = {}
    for method in args.compare:
        results[method] = load_results(args.output_dir, method, args.task)
        print(f"\nLoaded {len(results[method]['success_rates'])} runs for {method}")
    
    # Print comparison
    if "bfn_hybrid" in results and "diffusion" in results:
        print_comparison_table(
            results["bfn_hybrid"], 
            results["diffusion"], 
            args.task
        )
    else:
        # Print individual results
        for method, data in results.items():
            mean, std = compute_statistics(data['success_rates'])
            print(f"\n{method}: {mean*100:.1f}% ± {std*100:.1f}%")
    
    print("\n[Note: If no results found, ensure experiments have completed]")
    print("[Run: sbatch jobs/run_hybrid_discrete_experiments.sh]")


if __name__ == "__main__":
    main()

