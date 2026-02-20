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
Plot script for CMR Versius frame decay analysis results.

This script loads the JSON output from cosmos_frame_decay_analysis_cmr.py and creates
a dual-axis plot showing L1 distance and SSIM metrics over generated frames.

Example usage:
    python scripts/frame_decay_analysis/cosmos_frame_decay_analysis_cmr_plot.py \
        --results_path output/frame_decay_cmr/run_20250204_120000/frame_decay_results_20250204_120000.json \
        --checkpoint_label cmr-exp-10k \
        --error-band
"""

import argparse
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot CMR Versius frame decay metrics with dual y-axes."
    )
    parser.add_argument(
        "--results_path",
        type=str,
        required=True,
        help="Path to frame_decay_results_*.json produced by cosmos_frame_decay_analysis_cmr.py",
    )
    parser.add_argument(
        "--checkpoint_label",
        type=str,
        default=None,
        help="Checkpoint label to plot (defaults to the first entry in the JSON).",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Optional path to save the plot. Defaults to alongside the JSON file with .png extension.",
    )
    parser.add_argument(
        "--error-band",
        action="store_true",
        help="If set, render metrics with shaded error bands instead of error bars.",
    )
    parser.add_argument(
        "--figsize",
        type=str,
        default="12,3",
        help="Figure size as 'width,height' (default: '12,3')",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional title for the plot",
    )
    return parser.parse_args()


def load_results(results_path: str) -> tuple[List[Dict], Dict[str, Any]]:
    """
    Load results from JSON file.
    
    Handles both the new CMR format (with 'metadata' and 'results' keys) and 
    the legacy chole format (flat list of checkpoint entries).
    
    Returns:
        Tuple of (results_list, metadata_dict)
    """
    with open(results_path, "r") as f:
        data = json.load(f)
    
    # Check if this is the new CMR format with metadata
    if isinstance(data, dict) and 'results' in data:
        metadata = data.get('metadata', {})
        results = data['results']
        if not isinstance(results, list):
            raise ValueError("Expected 'results' key to contain a list of checkpoint entries.")
        return results, metadata
    
    # Legacy format: flat list of checkpoint entries
    if isinstance(data, list):
        return data, {}
    
    raise ValueError("Unexpected JSON format. Expected either a dict with 'results' key or a list.")


def select_checkpoint(data: List[Dict], checkpoint_label: Optional[str]) -> Dict:
    """Select a specific checkpoint entry from the results."""
    if checkpoint_label is None:
        if not data:
            raise ValueError("Results JSON is empty; nothing to plot.")
        return data[0]

    for entry in data:
        if entry.get("checkpoint_label") == checkpoint_label:
            return entry

    available = ", ".join(entry.get("checkpoint_label", "<unknown>") for entry in data)
    raise ValueError(
        f"Checkpoint label '{checkpoint_label}' not found. Available labels: {available}"
    )


def extract_metric_series(checkpoint_data: Dict, metric_name: str) -> Dict[str, List[float]]:
    """
    Extract metric means and stds from checkpoint data.
    
    Handles both:
    - New CMR format: {'l1_per_frame': {'means': [...], 'stds': [...]}}
    - Legacy chole format: {'aggregated_l1_per_frame': [{'mean': x, 'std': y}, ...]}
    """
    # Try new CMR format first
    if metric_name in checkpoint_data:
        metric_data = checkpoint_data[metric_name]
        if isinstance(metric_data, dict) and 'means' in metric_data:
            return {
                "means": [float(m) for m in metric_data.get("means", [])],
                "stds": [float(s) for s in metric_data.get("stds", [])]
            }
    
    # Try legacy format (aggregated_l1_per_frame or aggregated_ssim_per_frame)
    legacy_key = f"aggregated_{metric_name}"
    if legacy_key in checkpoint_data:
        metric_data = checkpoint_data[legacy_key]
        if isinstance(metric_data, list):
            means = []
            stds = []
            for entry in metric_data:
                means.append(float(entry.get("mean", 0.0)))
                stds.append(float(entry.get("std", 0.0)))
            return {"means": means, "stds": stds}
    
    # Return empty if metric not found
    return {"means": [], "stds": []}


def determine_output_path(results_path: str, checkpoint_label: str, user_output_path: Optional[str]) -> str:
    """Determine the output path for the plot."""
    if user_output_path:
        return user_output_path

    base_dir = os.path.dirname(os.path.abspath(results_path))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"frame_decay_cmr_{checkpoint_label}_{timestamp}.png"
    return os.path.join(base_dir, filename)


def plot_dual_axis(
    frames: List[int],
    l1_means: List[float],
    l1_stds: List[float],
    ssim_means: List[float],
    ssim_stds: List[float],
    checkpoint_label: str,
    output_path: str,
    use_error_band: bool,
    figsize: tuple[float, float] = (12, 3),
    title: Optional[str] = None,
    metadata: Optional[Dict] = None,
):
    """Create a dual-axis plot with L1 on left y-axis and SSIM on right y-axis."""
    if not frames:
        raise ValueError("No frames to plot. Check the input data for aggregated metrics.")

    fig, ax_l1 = plt.subplots(figsize=figsize)

    # Color-blind friendly colors
    color_l1 = "#E69F00"   # Orange
    color_ssim = "#0072B2"  # Blue

    # Plot L1 on the left y-axis
    if use_error_band:
        line_l1, = ax_l1.plot(frames, l1_means, color=color_l1, linewidth=1.5, label="L1 distance")
        lower = [m - s for m, s in zip(l1_means, l1_stds)]
        upper = [m + s for m, s in zip(l1_means, l1_stds)]
        ax_l1.fill_between(
            frames,
            lower,
            upper,
            color=mcolors.to_rgba(color_l1, alpha=0.35),
            edgecolor="none",
            zorder=1,
            label="_nolegend_",
        )
        ax_l1.plot(frames, lower, color=color_l1, linestyle="--", linewidth=0.8, alpha=0.9, label="_nolegend_")
        ax_l1.plot(frames, upper, color=color_l1, linestyle="--", linewidth=0.8, alpha=0.9, label="_nolegend_")
    else:
        ax_l1.errorbar(
            frames,
            l1_means,
            yerr=l1_stds,
            fmt="o",
            markersize=4,
            color=color_l1,
            ecolor=color_l1,
            elinewidth=1,
            capsize=3,
        )
        line_l1, = ax_l1.plot(frames, l1_means, color=color_l1, linewidth=1.5, label="L1 distance")
    
    ax_l1.set_xlabel("Generated Frame", fontsize=12)
    ax_l1.set_ylabel("L1 Distance", color=color_l1, fontsize=11)
    ax_l1.tick_params(axis="y", colors=color_l1, left=True, right=False, labelcolor=color_l1, labelleft=True, labelsize=9)
    if "left" in ax_l1.spines:
        ax_l1.spines["left"].set_visible(True)
        ax_l1.spines["left"].set_color(color_l1)

    # Plot SSIM on the right y-axis
    ax_ssim = ax_l1.twinx()
    if use_error_band:
        line_ssim, = ax_ssim.plot(frames, ssim_means, color=color_ssim, linewidth=1.5, label="SSIM")
        lower_ssim = [m - s for m, s in zip(ssim_means, ssim_stds)]
        upper_ssim = [m + s for m, s in zip(ssim_means, ssim_stds)]
        ax_ssim.fill_between(
            frames,
            lower_ssim,
            upper_ssim,
            color=mcolors.to_rgba(color_ssim, alpha=0.35),
            edgecolor="none",
            zorder=1,
            label="_nolegend_",
        )
        ax_ssim.plot(frames, lower_ssim, color=color_ssim, linestyle="--", linewidth=0.8, alpha=0.9, label="_nolegend_")
        ax_ssim.plot(frames, upper_ssim, color=color_ssim, linestyle="--", linewidth=0.8, alpha=0.9, label="_nolegend_")
    else:
        ax_ssim.errorbar(
            frames,
            ssim_means,
            yerr=ssim_stds,
            fmt="o",
            markersize=4,
            color=color_ssim,
            ecolor=color_ssim,
            elinewidth=1,
            capsize=3,
        )
        line_ssim, = ax_ssim.plot(frames, ssim_means, color=color_ssim, linewidth=1.5, label="SSIM")

    ax_ssim.set_ylabel("SSIM", color=color_ssim, fontsize=11)
    ax_ssim.tick_params(axis="y", colors=color_ssim, left=False, right=True, labelcolor=color_ssim, labelleft=False, labelright=True, labelsize=9)
    if "right" in ax_ssim.spines:
        ax_ssim.spines["right"].set_visible(True)
        ax_ssim.spines["right"].set_color(color_ssim)

    # Build combined legend
    legend_handles = [line_l1, line_ssim]
    legend_labels = ["L1 distance", "SSIM"]
    ax_l1.legend(legend_handles, legend_labels, loc="center right")

    # Add title if specified
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')
    elif metadata:
        # Auto-generate title from metadata
        total_frames = metadata.get('total_frames_analyzed', len(frames))
        seeds = metadata.get('seeds', [])
        if seeds:
            auto_title = f"CMR Frame Decay Analysis: {checkpoint_label} ({total_frames} frames, {len(seeds)} seeds)"
        else:
            auto_title = f"CMR Frame Decay Analysis: {checkpoint_label} ({total_frames} frames)"
        fig.suptitle(auto_title, fontsize=12)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def print_summary(checkpoint_data: Dict, metadata: Dict):
    """Print a summary of the frame decay metrics."""
    print("\n" + "=" * 60)
    print("FRAME DECAY SUMMARY")
    print("=" * 60)
    
    label = checkpoint_data.get('checkpoint_label', 'Unknown')
    print(f"Checkpoint: {label}")
    print(f"Episodes: {checkpoint_data.get('num_episodes', 'N/A')}")
    print(f"Max frames: {checkpoint_data.get('max_frames', 'N/A')}")
    
    if metadata:
        print(f"Seeds: {metadata.get('seeds', 'N/A')}")
        print(f"Total frames analyzed: {metadata.get('total_frames_analyzed', 'N/A')}")
    
    # L1 summary
    l1_data = extract_metric_series(checkpoint_data, 'l1_per_frame')
    if l1_data['means']:
        print(f"\nL1 Distance:")
        print(f"  First frame: {l1_data['means'][0]:.6f}")
        print(f"  Last frame:  {l1_data['means'][-1]:.6f}")
        increase = l1_data['means'][-1] - l1_data['means'][0]
        pct_increase = (l1_data['means'][-1] / l1_data['means'][0] - 1) * 100 if l1_data['means'][0] > 0 else 0
        print(f"  Increase:    {increase:.6f} ({pct_increase:.1f}%)")
    
    # SSIM summary
    ssim_data = extract_metric_series(checkpoint_data, 'ssim_per_frame')
    if ssim_data['means']:
        print(f"\nSSIM:")
        print(f"  First frame: {ssim_data['means'][0]:.6f}")
        print(f"  Last frame:  {ssim_data['means'][-1]:.6f}")
        decrease = ssim_data['means'][0] - ssim_data['means'][-1]
        pct_decrease = (1 - ssim_data['means'][-1] / ssim_data['means'][0]) * 100 if ssim_data['means'][0] > 0 else 0
        print(f"  Decrease:    {decrease:.6f} ({pct_decrease:.1f}%)")
    
    print("=" * 60)


def main() -> None:
    args = parse_args()
    
    # Parse figsize
    try:
        width, height = map(float, args.figsize.split(','))
        figsize = (width, height)
    except ValueError:
        print(f"Warning: Invalid figsize '{args.figsize}', using default (12, 3)")
        figsize = (12, 3)
    
    # Load results
    data, metadata = load_results(args.results_path)
    checkpoint_data = select_checkpoint(data, args.checkpoint_label)

    # Extract metrics
    l1_series = extract_metric_series(checkpoint_data, 'l1_per_frame')
    ssim_series = extract_metric_series(checkpoint_data, 'ssim_per_frame')

    # Determine frame numbers
    frames = list(range(1, len(l1_series["means"]) + 1))
    
    # Ensure L1 and SSIM have the same number of frames
    if len(ssim_series["means"]) != len(frames):
        min_len = min(len(frames), len(ssim_series["means"]))
        frames = frames[:min_len]
        l1_series["means"] = l1_series["means"][:min_len]
        l1_series["stds"] = l1_series["stds"][:min_len]
        ssim_series["means"] = ssim_series["means"][:min_len]
        ssim_series["stds"] = ssim_series["stds"][:min_len]

    # Determine output path
    output_path = determine_output_path(
        args.results_path,
        checkpoint_data.get("checkpoint_label", "checkpoint"),
        args.output_path,
    )

    # Print summary
    print_summary(checkpoint_data, metadata)

    # Create plot
    plot_dual_axis(
        frames,
        l1_series["means"],
        l1_series["stds"],
        ssim_series["means"],
        ssim_series["stds"],
        checkpoint_data.get("checkpoint_label", "checkpoint"),
        output_path,
        args.error_band,
        figsize=figsize,
        title=args.title,
        metadata=metadata,
    )

    print(f"\nSaved plot to: {output_path}")


if __name__ == "__main__":
    main()
