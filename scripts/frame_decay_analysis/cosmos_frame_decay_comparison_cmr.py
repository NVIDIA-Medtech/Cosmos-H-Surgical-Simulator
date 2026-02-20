# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Checkpoint comparison script for CMR Versius frame decay analysis.

This script compares two checkpoints' frame decay characteristics and determines
which checkpoint produces better quality video generation over time.

Features:
1. QUALITATIVE ANALYSIS:
   - Side-by-side L1 and SSIM decay curves
   - Difference plots showing frame-by-frame improvement
   - Shaded regions indicating which checkpoint is better

2. QUANTITATIVE ANALYSIS:
   - Mean L1/SSIM across all frames
   - Decay rate (linear regression slope)
   - Area Under Curve (AUC)
   - Final frame quality
   - Quality retention ratio (last frame / first frame)

3. STATISTICAL ANALYSIS:
   - Paired t-test for per-frame differences
   - Wilcoxon signed-rank test (non-parametric alternative)
   - Effect size (Cohen's d)
   - Bootstrap confidence intervals for mean differences
   - Per-frame significance indicators

Example usage:
    # Compare two checkpoints from the same JSON file
    python scripts/frame_decay_analysis/cosmos_frame_decay_comparison_cmr.py \
        --results_path output/frame_decay_cmr/run_*/frame_decay_results_*.json \
        --checkpoint_a cmr-exp1-10k \
        --checkpoint_b cmr-exp2-10k \
        --output_dir output/comparison_results

    # Compare checkpoints from different JSON files
    python scripts/frame_decay_analysis/cosmos_frame_decay_comparison_cmr.py \
        --results_path_a output/run1/frame_decay_results.json \
        --results_path_b output/run2/frame_decay_results.json \
        --checkpoint_a cmr-exp1-10k \
        --checkpoint_b cmr-exp2-10k \
        --output_dir output/comparison_results
"""

import argparse
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from scipy import stats


@dataclass
class CheckpointMetrics:
    """Container for checkpoint frame decay metrics."""
    label: str
    l1_means: List[float]
    l1_stds: List[float]
    ssim_means: List[float]
    ssim_stds: List[float]
    num_episodes: int
    num_frames: int
    
    # Derived metrics (computed after initialization)
    l1_auc: float = 0.0
    ssim_auc: float = 0.0
    l1_slope: float = 0.0
    ssim_slope: float = 0.0
    l1_mean_all: float = 0.0
    ssim_mean_all: float = 0.0
    l1_retention: float = 0.0  # last/first ratio for L1 (lower is better for L1)
    ssim_retention: float = 0.0  # last/first ratio for SSIM (higher is better)


@dataclass
class ComparisonResults:
    """Container for comparison results between two checkpoints."""
    checkpoint_a: CheckpointMetrics
    checkpoint_b: CheckpointMetrics
    
    # Statistical test results
    l1_ttest_pvalue: float = 0.0
    l1_ttest_statistic: float = 0.0
    l1_wilcoxon_pvalue: float = 0.0
    l1_cohens_d: float = 0.0
    l1_mean_diff: float = 0.0
    l1_mean_diff_ci: Tuple[float, float] = (0.0, 0.0)
    
    ssim_ttest_pvalue: float = 0.0
    ssim_ttest_statistic: float = 0.0
    ssim_wilcoxon_pvalue: float = 0.0
    ssim_cohens_d: float = 0.0
    ssim_mean_diff: float = 0.0
    ssim_mean_diff_ci: Tuple[float, float] = (0.0, 0.0)
    
    # Per-frame significance
    l1_per_frame_pvalues: List[float] = None
    ssim_per_frame_pvalues: List[float] = None
    
    # Winner determination
    l1_winner: str = ""
    ssim_winner: str = ""
    overall_winner: str = ""
    confidence_level: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare frame decay between two CMR checkpoints with statistical analysis."
    )
    
    # Input options - either single file with two checkpoints, or two separate files
    parser.add_argument(
        "--results_path",
        type=str,
        default=None,
        help="Path to JSON file containing both checkpoints (use with --checkpoint_a and --checkpoint_b)",
    )
    parser.add_argument(
        "--results_path_a",
        type=str,
        default=None,
        help="Path to JSON file for checkpoint A (alternative to --results_path)",
    )
    parser.add_argument(
        "--results_path_b",
        type=str,
        default=None,
        help="Path to JSON file for checkpoint B (alternative to --results_path)",
    )
    parser.add_argument(
        "--checkpoint_a",
        type=str,
        required=True,
        help="Label of first checkpoint to compare",
    )
    parser.add_argument(
        "--checkpoint_b",
        type=str,
        required=True,
        help="Label of second checkpoint to compare",
    )
    
    # Output options
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/frame_decay_comparison",
        help="Directory to save comparison results and plots",
    )
    parser.add_argument(
        "--significance_level",
        type=float,
        default=0.05,
        help="Significance level for statistical tests (default: 0.05)",
    )
    parser.add_argument(
        "--bootstrap_samples",
        type=int,
        default=10000,
        help="Number of bootstrap samples for confidence intervals (default: 10000)",
    )
    
    return parser.parse_args()


def load_results(results_path: str) -> Tuple[List[Dict], Dict[str, Any]]:
    """Load results from JSON file."""
    with open(results_path, "r") as f:
        data = json.load(f)
    
    if isinstance(data, dict) and 'results' in data:
        return data['results'], data.get('metadata', {})
    
    if isinstance(data, list):
        return data, {}
    
    raise ValueError("Unexpected JSON format")


def extract_checkpoint_data(
    results: List[Dict],
    checkpoint_label: str
) -> Dict:
    """Extract data for a specific checkpoint from results."""
    for entry in results:
        if entry.get("checkpoint_label") == "model_ema_bf16":
            return entry
    
    available = ", ".join(entry.get("checkpoint_label", "<unknown>") for entry in results)
    raise ValueError(f"Checkpoint '{checkpoint_label}' not found. Available: {available}")


def extract_metrics(checkpoint_data: Dict, metric_name: str) -> Tuple[List[float], List[float]]:
    """Extract means and stds for a metric from checkpoint data."""
    # Try new CMR format
    if metric_name in checkpoint_data:
        metric_data = checkpoint_data[metric_name]
        if isinstance(metric_data, dict) and 'means' in metric_data:
            return (
                [float(m) for m in metric_data.get("means", [])],
                [float(s) for s in metric_data.get("stds", [])]
            )
    
    # Try legacy format
    legacy_key = f"aggregated_{metric_name}"
    if legacy_key in checkpoint_data:
        metric_data = checkpoint_data[legacy_key]
        if isinstance(metric_data, list):
            means = [float(entry.get("mean", 0.0)) for entry in metric_data]
            stds = [float(entry.get("std", 0.0)) for entry in metric_data]
            return means, stds
    
    return [], []


def compute_derived_metrics(metrics: CheckpointMetrics) -> None:
    """Compute derived metrics like AUC, slope, etc."""
    if not metrics.l1_means or not metrics.ssim_means:
        return
    
    frames = np.arange(len(metrics.l1_means))
    
    # Area Under Curve (using trapezoidal rule)
    metrics.l1_auc = float(np.trapz(metrics.l1_means, frames))
    metrics.ssim_auc = float(np.trapz(metrics.ssim_means, frames))
    
    # Linear regression slope (decay rate)
    if len(frames) > 1:
        l1_slope, _ = np.polyfit(frames, metrics.l1_means, 1)
        ssim_slope, _ = np.polyfit(frames, metrics.ssim_means, 1)
        metrics.l1_slope = float(l1_slope)
        metrics.ssim_slope = float(ssim_slope)
    
    # Mean across all frames
    metrics.l1_mean_all = float(np.mean(metrics.l1_means))
    metrics.ssim_mean_all = float(np.mean(metrics.ssim_means))
    
    # Quality retention (last / first)
    if metrics.l1_means[0] > 0:
        metrics.l1_retention = float(metrics.l1_means[-1] / metrics.l1_means[0])
    if metrics.ssim_means[0] > 0:
        metrics.ssim_retention = float(metrics.ssim_means[-1] / metrics.ssim_means[0])


def load_checkpoint_metrics(
    results_path: str,
    checkpoint_label: str
) -> CheckpointMetrics:
    """Load and compute metrics for a checkpoint."""
    results, _ = load_results(results_path)
    checkpoint_data = extract_checkpoint_data(results, checkpoint_label)
    
    l1_means, l1_stds = extract_metrics(checkpoint_data, 'l1_per_frame')
    ssim_means, ssim_stds = extract_metrics(checkpoint_data, 'ssim_per_frame')
    
    # Ensure same length
    min_len = min(len(l1_means), len(ssim_means))
    l1_means = l1_means[:min_len]
    l1_stds = l1_stds[:min_len]
    ssim_means = ssim_means[:min_len]
    ssim_stds = ssim_stds[:min_len]
    
    metrics = CheckpointMetrics(
        label=checkpoint_label,
        l1_means=l1_means,
        l1_stds=l1_stds,
        ssim_means=ssim_means,
        ssim_stds=ssim_stds,
        num_episodes=checkpoint_data.get('num_episodes', 0),
        num_frames=min_len,
    )
    
    compute_derived_metrics(metrics)
    return metrics


def compute_cohens_d(group1: List[float], group2: List[float]) -> float:
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return 0.0
    
    return float((np.mean(group1) - np.mean(group2)) / pooled_std)


def bootstrap_mean_diff_ci(
    group1: List[float],
    group2: List[float],
    n_bootstrap: int = 10000,
    confidence: float = 0.95
) -> Tuple[float, float]:
    """Compute bootstrap confidence interval for mean difference."""
    np.random.seed(42)  # For reproducibility
    
    group1 = np.array(group1)
    group2 = np.array(group2)
    
    diffs = []
    n1, n2 = len(group1), len(group2)
    
    for _ in range(n_bootstrap):
        sample1 = np.random.choice(group1, size=n1, replace=True)
        sample2 = np.random.choice(group2, size=n2, replace=True)
        diffs.append(np.mean(sample1) - np.mean(sample2))
    
    alpha = 1 - confidence
    lower = np.percentile(diffs, 100 * alpha / 2)
    upper = np.percentile(diffs, 100 * (1 - alpha / 2))
    
    return (float(lower), float(upper))


def interpret_effect_size(d: float) -> str:
    """Interpret Cohen's d effect size."""
    d = abs(d)
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"


def perform_statistical_comparison(
    metrics_a: CheckpointMetrics,
    metrics_b: CheckpointMetrics,
    significance_level: float = 0.05,
    n_bootstrap: int = 10000,
) -> ComparisonResults:
    """Perform comprehensive statistical comparison between two checkpoints."""
    results = ComparisonResults(
        checkpoint_a=metrics_a,
        checkpoint_b=metrics_b,
    )
    
    # Ensure same number of frames
    n_frames = min(metrics_a.num_frames, metrics_b.num_frames)
    
    l1_a = metrics_a.l1_means[:n_frames]
    l1_b = metrics_b.l1_means[:n_frames]
    ssim_a = metrics_a.ssim_means[:n_frames]
    ssim_b = metrics_b.ssim_means[:n_frames]
    
    # === L1 Statistical Tests ===
    # Paired t-test (per-frame comparison)
    t_stat, p_val = stats.ttest_rel(l1_a, l1_b)
    results.l1_ttest_statistic = float(t_stat)
    results.l1_ttest_pvalue = float(p_val)
    
    # Wilcoxon signed-rank test (non-parametric)
    try:
        w_stat, w_pval = stats.wilcoxon(l1_a, l1_b)
        results.l1_wilcoxon_pvalue = float(w_pval)
    except ValueError:
        # Can fail if all differences are zero
        results.l1_wilcoxon_pvalue = 1.0
    
    # Cohen's d effect size
    results.l1_cohens_d = compute_cohens_d(l1_a, l1_b)
    
    # Mean difference and bootstrap CI
    results.l1_mean_diff = float(np.mean(l1_a) - np.mean(l1_b))
    results.l1_mean_diff_ci = bootstrap_mean_diff_ci(l1_a, l1_b, n_bootstrap)
    
    # === SSIM Statistical Tests ===
    t_stat, p_val = stats.ttest_rel(ssim_a, ssim_b)
    results.ssim_ttest_statistic = float(t_stat)
    results.ssim_ttest_pvalue = float(p_val)
    
    try:
        w_stat, w_pval = stats.wilcoxon(ssim_a, ssim_b)
        results.ssim_wilcoxon_pvalue = float(w_pval)
    except ValueError:
        results.ssim_wilcoxon_pvalue = 1.0
    
    results.ssim_cohens_d = compute_cohens_d(ssim_a, ssim_b)
    results.ssim_mean_diff = float(np.mean(ssim_a) - np.mean(ssim_b))
    results.ssim_mean_diff_ci = bootstrap_mean_diff_ci(ssim_a, ssim_b, n_bootstrap)
    
    # === Per-frame significance (using z-test approximation) ===
    # This is a simplified approach - ideally we'd have per-episode data
    results.l1_per_frame_pvalues = []
    results.ssim_per_frame_pvalues = []
    
    for i in range(n_frames):
        # Approximate z-test using pooled standard error
        l1_se = np.sqrt(metrics_a.l1_stds[i]**2 + metrics_b.l1_stds[i]**2)
        if l1_se > 0:
            z = (l1_a[i] - l1_b[i]) / l1_se
            p = 2 * (1 - stats.norm.cdf(abs(z)))
            results.l1_per_frame_pvalues.append(float(p))
        else:
            results.l1_per_frame_pvalues.append(1.0)
        
        ssim_se = np.sqrt(metrics_a.ssim_stds[i]**2 + metrics_b.ssim_stds[i]**2)
        if ssim_se > 0:
            z = (ssim_a[i] - ssim_b[i]) / ssim_se
            p = 2 * (1 - stats.norm.cdf(abs(z)))
            results.ssim_per_frame_pvalues.append(float(p))
        else:
            results.ssim_per_frame_pvalues.append(1.0)
    
    # === Determine Winners ===
    # For L1: lower is better, so negative mean_diff means A is better
    if results.l1_ttest_pvalue < significance_level:
        results.l1_winner = metrics_a.label if results.l1_mean_diff < 0 else metrics_b.label
    else:
        results.l1_winner = "No significant difference"
    
    # For SSIM: higher is better, so positive mean_diff means A is better
    if results.ssim_ttest_pvalue < significance_level:
        results.ssim_winner = metrics_a.label if results.ssim_mean_diff > 0 else metrics_b.label
    else:
        results.ssim_winner = "No significant difference"
    
    # Overall winner (considering both metrics)
    l1_better = None
    ssim_better = None
    
    if results.l1_ttest_pvalue < significance_level:
        l1_better = 'A' if results.l1_mean_diff < 0 else 'B'
    if results.ssim_ttest_pvalue < significance_level:
        ssim_better = 'A' if results.ssim_mean_diff > 0 else 'B'
    
    if l1_better == ssim_better and l1_better is not None:
        results.overall_winner = metrics_a.label if l1_better == 'A' else metrics_b.label
        results.confidence_level = "High (both metrics agree)"
    elif l1_better is None and ssim_better is None:
        results.overall_winner = "No significant difference"
        results.confidence_level = "N/A"
    elif l1_better is None:
        results.overall_winner = metrics_a.label if ssim_better == 'A' else metrics_b.label
        results.confidence_level = "Medium (SSIM only)"
    elif ssim_better is None:
        results.overall_winner = metrics_a.label if l1_better == 'A' else metrics_b.label
        results.confidence_level = "Medium (L1 only)"
    else:
        results.overall_winner = "Inconclusive (metrics disagree)"
        results.confidence_level = "Low"
    
    return results


def print_comparison_report(results: ComparisonResults, significance_level: float = 0.05):
    """Print a comprehensive comparison report."""
    a = results.checkpoint_a
    b = results.checkpoint_b
    
    print("\n" + "=" * 80)
    print("FRAME DECAY COMPARISON REPORT")
    print("=" * 80)
    
    print(f"\nCheckpoint A: {a.label}")
    print(f"Checkpoint B: {b.label}")
    print(f"Frames analyzed: {min(a.num_frames, b.num_frames)}")
    print(f"Significance level: α = {significance_level}")
    
    # === Quantitative Metrics ===
    print("\n" + "-" * 80)
    print("QUANTITATIVE METRICS")
    print("-" * 80)
    
    print(f"\n{'Metric':<25} {'Checkpoint A':>15} {'Checkpoint B':>15} {'Difference':>15} {'Better':<15}")
    print("-" * 85)
    
    # L1 metrics (lower is better)
    l1_diff = a.l1_mean_all - b.l1_mean_all
    l1_better = a.label if l1_diff < 0 else b.label
    print(f"{'L1 Mean (all frames)':<25} {a.l1_mean_all:>15.6f} {b.l1_mean_all:>15.6f} {l1_diff:>+15.6f} {l1_better:<15}")
    
    l1_auc_diff = a.l1_auc - b.l1_auc
    l1_auc_better = a.label if l1_auc_diff < 0 else b.label
    print(f"{'L1 AUC':<25} {a.l1_auc:>15.2f} {b.l1_auc:>15.2f} {l1_auc_diff:>+15.2f} {l1_auc_better:<15}")
    
    l1_slope_diff = a.l1_slope - b.l1_slope
    l1_slope_better = a.label if l1_slope_diff < 0 else b.label
    print(f"{'L1 Decay Rate (slope)':<25} {a.l1_slope:>15.6f} {b.l1_slope:>15.6f} {l1_slope_diff:>+15.6f} {l1_slope_better:<15}")
    
    l1_ret_diff = a.l1_retention - b.l1_retention
    l1_ret_better = a.label if l1_ret_diff < 0 else b.label
    print(f"{'L1 Retention (last/first)':<25} {a.l1_retention:>15.4f} {b.l1_retention:>15.4f} {l1_ret_diff:>+15.4f} {l1_ret_better:<15}")
    
    print()
    
    # SSIM metrics (higher is better)
    ssim_diff = a.ssim_mean_all - b.ssim_mean_all
    ssim_better = a.label if ssim_diff > 0 else b.label
    print(f"{'SSIM Mean (all frames)':<25} {a.ssim_mean_all:>15.6f} {b.ssim_mean_all:>15.6f} {ssim_diff:>+15.6f} {ssim_better:<15}")
    
    ssim_auc_diff = a.ssim_auc - b.ssim_auc
    ssim_auc_better = a.label if ssim_auc_diff > 0 else b.label
    print(f"{'SSIM AUC':<25} {a.ssim_auc:>15.2f} {b.ssim_auc:>15.2f} {ssim_auc_diff:>+15.2f} {ssim_auc_better:<15}")
    
    ssim_slope_diff = a.ssim_slope - b.ssim_slope
    ssim_slope_better = a.label if ssim_slope_diff > 0 else b.label  # Less negative slope is better
    print(f"{'SSIM Decay Rate (slope)':<25} {a.ssim_slope:>15.6f} {b.ssim_slope:>15.6f} {ssim_slope_diff:>+15.6f} {ssim_slope_better:<15}")
    
    ssim_ret_diff = a.ssim_retention - b.ssim_retention
    ssim_ret_better = a.label if ssim_ret_diff > 0 else b.label
    print(f"{'SSIM Retention (last/first)':<25} {a.ssim_retention:>15.4f} {b.ssim_retention:>15.4f} {ssim_ret_diff:>+15.4f} {ssim_ret_better:<15}")
    
    # === Statistical Analysis ===
    print("\n" + "-" * 80)
    print("STATISTICAL ANALYSIS")
    print("-" * 80)
    
    print("\n--- L1 Distance Analysis ---")
    print(f"Mean difference (A - B):     {results.l1_mean_diff:+.6f}")
    print(f"95% Bootstrap CI:            [{results.l1_mean_diff_ci[0]:+.6f}, {results.l1_mean_diff_ci[1]:+.6f}]")
    print(f"Paired t-test p-value:       {results.l1_ttest_pvalue:.6f} {'***' if results.l1_ttest_pvalue < 0.001 else '**' if results.l1_ttest_pvalue < 0.01 else '*' if results.l1_ttest_pvalue < 0.05 else ''}")
    print(f"Wilcoxon signed-rank p:      {results.l1_wilcoxon_pvalue:.6f}")
    print(f"Cohen's d effect size:       {results.l1_cohens_d:+.4f} ({interpret_effect_size(results.l1_cohens_d)})")
    print(f"Winner (L1):                 {results.l1_winner}")
    
    print("\n--- SSIM Analysis ---")
    print(f"Mean difference (A - B):     {results.ssim_mean_diff:+.6f}")
    print(f"95% Bootstrap CI:            [{results.ssim_mean_diff_ci[0]:+.6f}, {results.ssim_mean_diff_ci[1]:+.6f}]")
    print(f"Paired t-test p-value:       {results.ssim_ttest_pvalue:.6f} {'***' if results.ssim_ttest_pvalue < 0.001 else '**' if results.ssim_ttest_pvalue < 0.01 else '*' if results.ssim_ttest_pvalue < 0.05 else ''}")
    print(f"Wilcoxon signed-rank p:      {results.ssim_wilcoxon_pvalue:.6f}")
    print(f"Cohen's d effect size:       {results.ssim_cohens_d:+.4f} ({interpret_effect_size(results.ssim_cohens_d)})")
    print(f"Winner (SSIM):               {results.ssim_winner}")
    
    # === Conclusion ===
    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print(f"\nOverall Winner: {results.overall_winner}")
    print(f"Confidence: {results.confidence_level}")
    
    if results.overall_winner not in ["No significant difference", "Inconclusive (metrics disagree)"]:
        winner_label = results.overall_winner
        if winner_label == a.label:
            l1_improvement = (1 - a.l1_mean_all / b.l1_mean_all) * 100
            ssim_improvement = (a.ssim_mean_all / b.ssim_mean_all - 1) * 100
        else:
            l1_improvement = (1 - b.l1_mean_all / a.l1_mean_all) * 100
            ssim_improvement = (b.ssim_mean_all / a.ssim_mean_all - 1) * 100
        
        print(f"\n{winner_label} shows:")
        print(f"  - {abs(l1_improvement):.2f}% {'lower' if l1_improvement > 0 else 'higher'} L1 distance (average across frames)")
        print(f"  - {abs(ssim_improvement):.2f}% {'higher' if ssim_improvement > 0 else 'lower'} SSIM score (average across frames)")
    
    print("\nNote: * p<0.05, ** p<0.01, *** p<0.001")
    print("=" * 80)


def create_comparison_plots(
    results: ComparisonResults,
    output_dir: str,
    significance_level: float = 0.05,
):
    """Create comprehensive comparison plots."""
    a = results.checkpoint_a
    b = results.checkpoint_b
    n_frames = min(a.num_frames, b.num_frames)
    frames = list(range(1, n_frames + 1))
    
    # Colors
    color_a = "#E69F00"  # Orange
    color_b = "#0072B2"  # Blue
    color_diff = "#009E73"  # Green
    color_sig = "#D55E00"  # Red-orange
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # === Figure 1: Side-by-side decay curves ===
    fig1, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # L1 plot
    ax1 = axes[0]
    ax1.plot(frames, a.l1_means[:n_frames], color=color_a, linewidth=2, label=a.label)
    ax1.fill_between(frames, 
                     [m - s for m, s in zip(a.l1_means[:n_frames], a.l1_stds[:n_frames])],
                     [m + s for m, s in zip(a.l1_means[:n_frames], a.l1_stds[:n_frames])],
                     color=mcolors.to_rgba(color_a, alpha=0.3), edgecolor='none')
    
    ax1.plot(frames, b.l1_means[:n_frames], color=color_b, linewidth=2, label=b.label)
    ax1.fill_between(frames,
                     [m - s for m, s in zip(b.l1_means[:n_frames], b.l1_stds[:n_frames])],
                     [m + s for m, s in zip(b.l1_means[:n_frames], b.l1_stds[:n_frames])],
                     color=mcolors.to_rgba(color_b, alpha=0.3), edgecolor='none')
    
    ax1.set_xlabel('Generated Frame', fontsize=12)
    ax1.set_ylabel('L1 Distance', fontsize=12)
    ax1.set_title('L1 Distance Comparison', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Add p-value annotation
    pval_text = f"p = {results.l1_ttest_pvalue:.4f}"
    if results.l1_ttest_pvalue < significance_level:
        pval_text += f" (sig.)"
    ax1.text(0.98, 0.02, pval_text, transform=ax1.transAxes, ha='right', va='bottom',
             fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # SSIM plot
    ax2 = axes[1]
    ax2.plot(frames, a.ssim_means[:n_frames], color=color_a, linewidth=2, label=a.label)
    ax2.fill_between(frames,
                     [m - s for m, s in zip(a.ssim_means[:n_frames], a.ssim_stds[:n_frames])],
                     [m + s for m, s in zip(a.ssim_means[:n_frames], a.ssim_stds[:n_frames])],
                     color=mcolors.to_rgba(color_a, alpha=0.3), edgecolor='none')
    
    ax2.plot(frames, b.ssim_means[:n_frames], color=color_b, linewidth=2, label=b.label)
    ax2.fill_between(frames,
                     [m - s for m, s in zip(b.ssim_means[:n_frames], b.ssim_stds[:n_frames])],
                     [m + s for m, s in zip(b.ssim_means[:n_frames], b.ssim_stds[:n_frames])],
                     color=mcolors.to_rgba(color_b, alpha=0.3), edgecolor='none')
    
    ax2.set_xlabel('Generated Frame', fontsize=12)
    ax2.set_ylabel('SSIM', fontsize=12)
    ax2.set_title('SSIM Comparison', fontsize=14, fontweight='bold')
    ax2.legend(loc='lower left')
    ax2.grid(True, alpha=0.3)
    
    pval_text = f"p = {results.ssim_ttest_pvalue:.4f}"
    if results.ssim_ttest_pvalue < significance_level:
        pval_text += f" (sig.)"
    ax2.text(0.98, 0.02, pval_text, transform=ax2.transAxes, ha='right', va='bottom',
             fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    fig1.suptitle(f'Frame Decay Comparison: {a.label} vs {b.label}', fontsize=16, fontweight='bold')
    fig1.tight_layout()
    
    plot1_path = os.path.join(output_dir, f"comparison_curves_{timestamp}.png")
    fig1.savefig(plot1_path, dpi=300, bbox_inches='tight')
    plt.close(fig1)
    print(f"Saved decay curves plot to: {plot1_path}")
    
    # === Figure 2: Difference plots with significance markers ===
    fig2, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # L1 difference
    ax1 = axes[0]
    l1_diff = [a.l1_means[i] - b.l1_means[i] for i in range(n_frames)]
    
    # Color bars by which checkpoint is better (lower L1 is better)
    colors = [color_a if d < 0 else color_b for d in l1_diff]
    ax1.bar(frames, l1_diff, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=1)
    
    # Mark significant differences
    for i, p in enumerate(results.l1_per_frame_pvalues[:n_frames]):
        if p < significance_level:
            ax1.scatter(frames[i], l1_diff[i] + np.sign(l1_diff[i]) * 0.005, 
                       marker='*', color=color_sig, s=100, zorder=5)
    
    ax1.set_xlabel('Generated Frame', fontsize=12)
    ax1.set_ylabel(f'L1 Difference ({a.label} - {b.label})', fontsize=11)
    ax1.set_title('L1 Difference per Frame (negative = A better, positive = B better)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=color_a, alpha=0.7, edgecolor='black', label=f'{a.label} better'),
        Patch(facecolor=color_b, alpha=0.7, edgecolor='black', label=f'{b.label} better'),
        plt.Line2D([0], [0], marker='*', color='w', markerfacecolor=color_sig, markersize=15, label=f'p < {significance_level}')
    ]
    ax1.legend(handles=legend_elements, loc='upper right')
    
    # SSIM difference
    ax2 = axes[1]
    ssim_diff = [a.ssim_means[i] - b.ssim_means[i] for i in range(n_frames)]
    
    # Color bars by which checkpoint is better (higher SSIM is better)
    colors = [color_a if d > 0 else color_b for d in ssim_diff]
    ax2.bar(frames, ssim_diff, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    
    # Mark significant differences
    for i, p in enumerate(results.ssim_per_frame_pvalues[:n_frames]):
        if p < significance_level:
            ax2.scatter(frames[i], ssim_diff[i] + np.sign(ssim_diff[i]) * 0.005,
                       marker='*', color=color_sig, s=100, zorder=5)
    
    ax2.set_xlabel('Generated Frame', fontsize=12)
    ax2.set_ylabel(f'SSIM Difference ({a.label} - {b.label})', fontsize=11)
    ax2.set_title('SSIM Difference per Frame (positive = A better, negative = B better)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend(handles=legend_elements, loc='lower right')
    
    fig2.suptitle(f'Per-Frame Difference Analysis', fontsize=14, fontweight='bold')
    fig2.tight_layout()
    
    plot2_path = os.path.join(output_dir, f"comparison_differences_{timestamp}.png")
    fig2.savefig(plot2_path, dpi=300, bbox_inches='tight')
    plt.close(fig2)
    print(f"Saved difference plot to: {plot2_path}")
    
    # === Figure 3: Summary bar chart ===
    fig3, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # L1 metrics bar chart
    ax1 = axes[0]
    metrics_names = ['Mean L1', 'AUC', 'Decay Rate\n(×100)', 'Retention']
    a_values = [a.l1_mean_all, a.l1_auc / 100, a.l1_slope * 100, a.l1_retention]
    b_values = [b.l1_mean_all, b.l1_auc / 100, b.l1_slope * 100, b.l1_retention]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, a_values, width, label=a.label, color=color_a, alpha=0.8)
    bars2 = ax1.bar(x + width/2, b_values, width, label=b.label, color=color_b, alpha=0.8)
    
    ax1.set_ylabel('Value', fontsize=12)
    ax1.set_title('L1 Metrics Comparison\n(lower is better)', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics_names)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # SSIM metrics bar chart
    ax2 = axes[1]
    metrics_names = ['Mean SSIM', 'AUC', 'Decay Rate\n(×100)', 'Retention']
    a_values = [a.ssim_mean_all, a.ssim_auc / 100, a.ssim_slope * 100, a.ssim_retention]
    b_values = [b.ssim_mean_all, b.ssim_auc / 100, b.ssim_slope * 100, b.ssim_retention]
    
    bars1 = ax2.bar(x - width/2, a_values, width, label=a.label, color=color_a, alpha=0.8)
    bars2 = ax2.bar(x + width/2, b_values, width, label=b.label, color=color_b, alpha=0.8)
    
    ax2.set_ylabel('Value', fontsize=12)
    ax2.set_title('SSIM Metrics Comparison\n(higher is better)', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics_names)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    fig3.suptitle(f'Summary Metrics: {a.label} vs {b.label}', fontsize=14, fontweight='bold')
    fig3.tight_layout()
    
    plot3_path = os.path.join(output_dir, f"comparison_summary_{timestamp}.png")
    fig3.savefig(plot3_path, dpi=300, bbox_inches='tight')
    plt.close(fig3)
    print(f"Saved summary plot to: {plot3_path}")
    
    return [plot1_path, plot2_path, plot3_path]


def save_comparison_results(
    results: ComparisonResults,
    output_dir: str,
) -> str:
    """Save comparison results to JSON."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(output_dir, f"comparison_results_{timestamp}.json")
    
    data = {
        'metadata': {
            'timestamp': timestamp,
            'checkpoint_a': results.checkpoint_a.label,
            'checkpoint_b': results.checkpoint_b.label,
            'num_frames': min(results.checkpoint_a.num_frames, results.checkpoint_b.num_frames),
        },
        'checkpoint_a_metrics': {
            'label': results.checkpoint_a.label,
            'l1_mean': results.checkpoint_a.l1_mean_all,
            'l1_auc': results.checkpoint_a.l1_auc,
            'l1_slope': results.checkpoint_a.l1_slope,
            'l1_retention': results.checkpoint_a.l1_retention,
            'ssim_mean': results.checkpoint_a.ssim_mean_all,
            'ssim_auc': results.checkpoint_a.ssim_auc,
            'ssim_slope': results.checkpoint_a.ssim_slope,
            'ssim_retention': results.checkpoint_a.ssim_retention,
        },
        'checkpoint_b_metrics': {
            'label': results.checkpoint_b.label,
            'l1_mean': results.checkpoint_b.l1_mean_all,
            'l1_auc': results.checkpoint_b.l1_auc,
            'l1_slope': results.checkpoint_b.l1_slope,
            'l1_retention': results.checkpoint_b.l1_retention,
            'ssim_mean': results.checkpoint_b.ssim_mean_all,
            'ssim_auc': results.checkpoint_b.ssim_auc,
            'ssim_slope': results.checkpoint_b.ssim_slope,
            'ssim_retention': results.checkpoint_b.ssim_retention,
        },
        'statistical_analysis': {
            'l1': {
                'mean_diff': results.l1_mean_diff,
                'mean_diff_ci_lower': results.l1_mean_diff_ci[0],
                'mean_diff_ci_upper': results.l1_mean_diff_ci[1],
                'ttest_pvalue': results.l1_ttest_pvalue,
                'ttest_statistic': results.l1_ttest_statistic,
                'wilcoxon_pvalue': results.l1_wilcoxon_pvalue,
                'cohens_d': results.l1_cohens_d,
                'effect_size_interpretation': interpret_effect_size(results.l1_cohens_d),
                'winner': results.l1_winner,
            },
            'ssim': {
                'mean_diff': results.ssim_mean_diff,
                'mean_diff_ci_lower': results.ssim_mean_diff_ci[0],
                'mean_diff_ci_upper': results.ssim_mean_diff_ci[1],
                'ttest_pvalue': results.ssim_ttest_pvalue,
                'ttest_statistic': results.ssim_ttest_statistic,
                'wilcoxon_pvalue': results.ssim_wilcoxon_pvalue,
                'cohens_d': results.ssim_cohens_d,
                'effect_size_interpretation': interpret_effect_size(results.ssim_cohens_d),
                'winner': results.ssim_winner,
            },
        },
        'conclusion': {
            'overall_winner': results.overall_winner,
            'confidence_level': results.confidence_level,
        },
        'per_frame_data': {
            'l1_a_means': results.checkpoint_a.l1_means,
            'l1_b_means': results.checkpoint_b.l1_means,
            'ssim_a_means': results.checkpoint_a.ssim_means,
            'ssim_b_means': results.checkpoint_b.ssim_means,
            'l1_per_frame_pvalues': results.l1_per_frame_pvalues,
            'ssim_per_frame_pvalues': results.ssim_per_frame_pvalues,
        },
    }
    
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Saved comparison results to: {json_path}")
    return json_path


def main():
    args = parse_args()
    
    # Validate input arguments
    if args.results_path is None and (args.results_path_a is None or args.results_path_b is None):
        raise ValueError("Either --results_path or both --results_path_a and --results_path_b must be provided")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load checkpoint metrics
    if args.results_path:
        # Both checkpoints from the same file
        print(f"Loading checkpoints from: {args.results_path}")
        metrics_a = load_checkpoint_metrics(args.results_path, args.checkpoint_a)
        metrics_b = load_checkpoint_metrics(args.results_path, args.checkpoint_b)
    else:
        # Checkpoints from different files
        print(f"Loading checkpoint A from: {args.results_path_a}")
        print(f"Loading checkpoint B from: {args.results_path_b}")
        metrics_a = load_checkpoint_metrics(args.results_path_a, args.checkpoint_a)
        metrics_b = load_checkpoint_metrics(args.results_path_b, args.checkpoint_b)
    
    print(f"\nCheckpoint A: {metrics_a.label} ({metrics_a.num_frames} frames)")
    print(f"Checkpoint B: {metrics_b.label} ({metrics_b.num_frames} frames)")
    
    # Perform statistical comparison
    print("\nPerforming statistical analysis...")
    comparison_results = perform_statistical_comparison(
        metrics_a,
        metrics_b,
        significance_level=args.significance_level,
        n_bootstrap=args.bootstrap_samples,
    )
    
    # Print comparison report
    print_comparison_report(comparison_results, args.significance_level)
    
    # Create plots
    print("\nGenerating comparison plots...")
    create_comparison_plots(comparison_results, args.output_dir, args.significance_level)
    
    # Save results to JSON
    save_comparison_results(comparison_results, args.output_dir)
    
    print("\nComparison analysis complete!")


if __name__ == "__main__":
    main()
