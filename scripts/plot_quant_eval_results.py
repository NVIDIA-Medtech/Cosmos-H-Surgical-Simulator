#!/usr/bin/env python3
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
Plot quantitative evaluation results for checkpoint comparison.

Reads results either from hardcoded data or from a JSON file produced by
cosmos_h_surgical_simulator_quant_eval.py.

Usage:
  python scripts/plot_quant_eval_results.py
  python scripts/plot_quant_eval_results.py --json output/quant_eval_cmr/quant_eval_results_*.json
  python scripts/plot_quant_eval_results.py --output figures/quant_eval.pdf
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── Hardcoded results ────────────────────────────────────────────────────────
# Edit / extend these when you have new runs.


@dataclass
class CheckpointResult:
    experiment: str
    checkpoint: str
    iteration: int
    fds: float  # Mean L1 (lower = better)
    gatc: float  # Median GATC (higher = better)
    tcd: float  # Median TCD in px (lower = better)
    l1_early: float = np.nan
    l1_mid: float = np.nan
    l1_late: float = np.nan
    gatc_early: float = np.nan
    gatc_mid: float = np.nan
    gatc_late: float = np.nan
    tcd_early: float = np.nan
    tcd_mid: float = np.nan
    tcd_late: float = np.nan


RESULTS: List[CheckpointResult] = [
    # finetune1
    # CheckpointResult("finetune1", "iter_000015000", 15000, 0, 0, 0),
    CheckpointResult("finetune1", "iter_000030000", 20000, 0.2371, 0.2557, 201),
    # CheckpointResult("finetune1", "iter_000020000", 30000, 0, 0, 0),
    CheckpointResult("finetune1", "iter_000042600", 42600, 0.2183, 0.2794, 184),
    CheckpointResult("finetune1", "iter_000043200", 43200, 0.2363, 0.2514, 208),
    # finetune2
    CheckpointResult("finetune2", "iter_000016200", 16200, 0.2384, 0.2571, 227),
    CheckpointResult("finetune2", "iter_000018400", 18400, 0.2372, 0.2504, 192),
    CheckpointResult("finetune2", "iter_000030000", 30000, 0.2320, 0.2594, 205),
    CheckpointResult("finetune2", "iter_000035800", 35800, 0.2344, 0.2533, 199),
    CheckpointResult("finetune2", "iter_000043200", 43200, 0.2304, 0.2568, 209.74),
    # finetune3
    CheckpointResult("finetune3", "iter_000012600", 12600, 0.2341, 0.2590, 201),
    CheckpointResult("finetune3", "iter_000013200", 13200, 0.2363, 0.2638, 203),
    CheckpointResult("finetune3", "iter_000023000", 23000, 0.2355, 0.2513, 191),
    CheckpointResult("finetune3", "iter_000027000", 27000, 0.2347, 0.2504, 199),
    CheckpointResult("finetune3", "iter_000034600", 34600, 0.2329, 0.2528, 186),
]


# ── Colour palette & style ───────────────────────────────────────────────────
EXPERIMENT_STYLES = {
    "finetune1": {"color": "#2ca02c", "marker": "D"},
    "finetune2": {"color": "#1f77b4", "marker": "o"},
    "finetune3": {"color": "#ff7f0e", "marker": "s"},
}

DEFAULT_COLORS = ["#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
DEFAULT_MARKERS = ["D", "^", "v", "P", "X", "*"]


def _style_for(exp: str, idx: int) -> dict:
    if exp in EXPERIMENT_STYLES:
        return EXPERIMENT_STYLES[exp]
    ci = idx % len(DEFAULT_COLORS)
    mi = idx % len(DEFAULT_MARKERS)
    return {"color": DEFAULT_COLORS[ci], "marker": DEFAULT_MARKERS[mi]}


# ── Plotting ──────────────────────────────────────────────────────────────────
def plot_results(
    results: List[CheckpointResult],
    output_path: str = "output/quant_eval_comparison.png",
    dpi: int = 180,
) -> None:
    experiments = sorted(set(r.experiment for r in results))
    exp_idx = {e: i for i, e in enumerate(experiments)}

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Cosmos Surgical Simulator — Checkpoint Comparison", fontsize=14, fontweight="bold", y=1.02)

    metrics = [
        ("FDS (Mean L1)", "fds", "lower = better", True),
        ("GATC (Median)", "gatc", "higher = better", False),
        ("TCD (Median, px)", "tcd", "lower = better", True),
    ]

    for ax, (title, attr, direction, lower_is_better) in zip(axes, metrics):
        for exp in experiments:
            style = _style_for(exp, exp_idx[exp])
            pts = sorted([r for r in results if r.experiment == exp], key=lambda r: r.iteration)
            iters = [r.iteration / 1000 for r in pts]
            vals = [getattr(r, attr) for r in pts]

            ax.plot(
                iters,
                vals,
                color=style["color"],
                marker=style["marker"],
                linewidth=1.8,
                markersize=7,
                markeredgecolor="white",
                markeredgewidth=0.8,
                label=exp,
                zorder=3,
            )

        # Best-value indicator
        all_vals = [getattr(r, attr) for r in results]
        best_val = min(all_vals) if lower_is_better else max(all_vals)
        best_r = [r for r in results if getattr(r, attr) == best_val][0]
        best_iter = best_r.iteration / 1000
        ax.annotate(
            f"best: {best_val:.4f}" if attr != "tcd" else f"best: {best_val:.0f}",
            xy=(best_iter, best_val),
            xytext=(0, -18 if lower_is_better else 18),
            textcoords="offset points",
            fontsize=8,
            fontweight="bold",
            color="#333333",
            ha="center",
            va="top" if lower_is_better else "bottom",
            arrowprops=dict(arrowstyle="-", color="#999999", lw=0.8),
        )

        ax.set_title(f"{title}\n({direction})", fontsize=11, pad=8)
        ax.set_xlabel("Training iteration (×1k)", fontsize=10)
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.0fk"))
        ax.grid(True, alpha=0.3, linewidth=0.6)
        ax.tick_params(labelsize=9)

        # Tighten y-axis to data range with 10% padding
        ymin, ymax = min(all_vals), max(all_vals)
        margin = (ymax - ymin) * 0.25 if ymax > ymin else abs(ymin) * 0.1
        ax.set_ylim(ymin - margin, ymax + margin)

    # Shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(experiments),
        fontsize=10,
        frameon=True,
        fancybox=True,
        shadow=False,
        bbox_to_anchor=(0.5, -0.06),
    )

    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    print(f"Saved plot to {output_path}")
    plt.close(fig)

    # Also save as PDF for papers
    pdf_path = output_path.rsplit(".", 1)[0] + ".pdf"
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
    fig2.suptitle("Cosmos Surgical Simulator — Checkpoint Comparison", fontsize=14, fontweight="bold", y=1.02)

    for ax, (title, attr, direction, lower_is_better) in zip(axes2, metrics):
        for exp in experiments:
            style = _style_for(exp, exp_idx[exp])
            pts = sorted([r for r in results if r.experiment == exp], key=lambda r: r.iteration)
            iters = [r.iteration / 1000 for r in pts]
            vals = [getattr(r, attr) for r in pts]
            ax.plot(
                iters,
                vals,
                color=style["color"],
                marker=style["marker"],
                linewidth=1.8,
                markersize=7,
                markeredgecolor="white",
                markeredgewidth=0.8,
                label=exp,
                zorder=3,
            )
        all_vals = [getattr(r, attr) for r in results]
        ax.set_title(f"{title}\n({direction})", fontsize=11, pad=8)
        ax.set_xlabel("Training iteration (×1k)", fontsize=10)
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.0fk"))
        ax.grid(True, alpha=0.3, linewidth=0.6)
        ax.tick_params(labelsize=9)
        ymin, ymax = min(all_vals), max(all_vals)
        margin = (ymax - ymin) * 0.25 if ymax > ymin else abs(ymin) * 0.1
        ax.set_ylim(ymin - margin, ymax + margin)

    handles, labels = axes2[0].get_legend_handles_labels()
    fig2.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(experiments),
        fontsize=10,
        frameon=True,
        fancybox=True,
        shadow=False,
        bbox_to_anchor=(0.5, -0.06),
    )
    plt.tight_layout()
    fig2.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved plot to {pdf_path}")
    plt.close(fig2)


def plot_chunk_panels(
    results: List[CheckpointResult],
    metric_name: str,
    output_path: str,
    dpi: int = 180,
) -> None:
    """Plot dedicated 3-panel Early/Mid/Late checkpoint comparison for one metric."""
    experiments = sorted(set(r.experiment for r in results))
    exp_idx = {e: i for i, e in enumerate(experiments)}

    if metric_name == "l1":
        chunk_attrs = [("l1_early", "Early"), ("l1_mid", "Mid"), ("l1_late", "Late")]
        title = "Chunk-wise FDS (L1)"
        direction = "lower = better"
        lower_is_better = True
        fallback_attr = "fds"
    elif metric_name == "gatc":
        chunk_attrs = [("gatc_early", "Early"), ("gatc_mid", "Mid"), ("gatc_late", "Late")]
        title = "Chunk-wise GATC"
        direction = "higher = better"
        lower_is_better = False
        fallback_attr = "gatc"
    elif metric_name == "tcd":
        chunk_attrs = [("tcd_early", "Early"), ("tcd_mid", "Mid"), ("tcd_late", "Late")]
        title = "Chunk-wise TCD"
        direction = "lower = better"
        lower_is_better = True
        fallback_attr = "tcd"
    else:
        raise ValueError(f"Unknown metric_name: {metric_name}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"Cosmos Surgical Simulator — {title} by Phase\n"
        "Early = chunk1 (frames 1-12), Mid = chunks2-3 (13-36), Late = chunks4-6 (37-72)",
        fontsize=12,
        fontweight="bold",
        y=1.04,
    )

    for ax, (attr, phase_name) in zip(axes, chunk_attrs):
        all_vals = []
        for exp in experiments:
            style = _style_for(exp, exp_idx[exp])
            pts = sorted([r for r in results if r.experiment == exp], key=lambda r: r.iteration)
            iters = [r.iteration / 1000 for r in pts]
            vals = []
            for r in pts:
                v = getattr(r, attr)
                if np.isnan(v):
                    # Fallback to overall metric if per-chunk values are unavailable
                    v = getattr(r, fallback_attr)
                vals.append(v)
            all_vals.extend(vals)

            ax.plot(
                iters,
                vals,
                color=style["color"],
                marker=style["marker"],
                linewidth=1.8,
                markersize=7,
                markeredgecolor="white",
                markeredgewidth=0.8,
                label=exp,
                zorder=3,
            )

        best_val = min(all_vals) if lower_is_better else max(all_vals)
        ax.set_title(f"{phase_name}\n({direction})", fontsize=11, pad=8)
        ax.set_xlabel("Training iteration (×1k)", fontsize=10)
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.0fk"))
        ax.grid(True, alpha=0.3, linewidth=0.6)
        ax.tick_params(labelsize=9)

        ymin, ymax = min(all_vals), max(all_vals)
        margin = (ymax - ymin) * 0.25 if ymax > ymin else max(abs(ymin) * 0.1, 1e-3)
        ax.set_ylim(ymin - margin, ymax + margin)
        ax.text(
            0.02,
            0.03,
            f"best: {best_val:.4f}" if metric_name != "tcd" else f"best: {best_val:.1f}",
            transform=ax.transAxes,
            fontsize=8,
            color="#333333",
        )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(experiments),
        fontsize=10,
        frameon=True,
        fancybox=True,
        shadow=False,
        bbox_to_anchor=(0.5, -0.06),
    )

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    print(f"Saved chunk plot to {output_path}")

    pdf_path = output_path.rsplit(".", 1)[0] + ".pdf"
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved chunk plot to {pdf_path}")
    plt.close(fig)


# ── Bar chart variant (better when iterations don't form a progression) ──────
def plot_bar_chart(
    results: List[CheckpointResult],
    output_path: str = "output/quant_eval_bar.png",
    dpi: int = 180,
) -> None:
    experiments = sorted(set(r.experiment for r in results))
    exp_idx = {e: i for i, e in enumerate(experiments)}

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    fig.suptitle("Cosmos Surgical Simulator — Checkpoint Comparison", fontsize=14, fontweight="bold", y=1.02)

    metrics = [
        ("FDS (Mean L1)", "fds", "lower = better", True),
        ("GATC (Median)", "gatc", "higher = better", False),
        ("TCD (Median, px)", "tcd", "lower = better", True),
    ]

    labels = []
    for r in results:
        iter_k = r.iteration // 1000
        labels.append(f"{r.experiment}\n{iter_k}k")

    x = np.arange(len(results))
    bar_colors = [_style_for(r.experiment, exp_idx[r.experiment])["color"] for r in results]

    for ax, (title, attr, direction, lower_is_better) in zip(axes, metrics):
        vals = [getattr(r, attr) for r in results]
        bars = ax.bar(x, vals, color=bar_colors, edgecolor="white", linewidth=0.8, width=0.7, zorder=3)

        # Value labels on bars
        for bar, v in zip(bars, vals):
            fmt = f"{v:.4f}" if attr != "tcd" else f"{v:.0f}"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                fmt,
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )

        # Highlight best bar
        best_idx = int(np.argmin(vals) if lower_is_better else np.argmax(vals))
        bars[best_idx].set_edgecolor("#2d2d2d")
        bars[best_idx].set_linewidth(2.0)

        ax.set_title(f"{title}\n({direction})", fontsize=11, pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.grid(True, axis="y", alpha=0.3, linewidth=0.6)
        ax.tick_params(axis="y", labelsize=9)

        # y-axis: start near data to emphasize differences
        ymin, ymax = min(vals), max(vals)
        margin = (ymax - ymin) * 0.3 if ymax > ymin else abs(ymin) * 0.1
        ax.set_ylim(ymin - margin, ymax + margin * 2)

    # Legend by experiment
    from matplotlib.patches import Patch

    legend_handles = [
        Patch(facecolor=_style_for(e, exp_idx[e])["color"], edgecolor="white", label=e) for e in experiments
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=len(experiments),
        fontsize=10,
        frameon=True,
        fancybox=True,
        shadow=False,
        bbox_to_anchor=(0.5, -0.04),
    )

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    print(f"Saved bar chart to {output_path}")

    pdf_path = output_path.rsplit(".", 1)[0] + ".pdf"
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved bar chart to {pdf_path}")
    plt.close(fig)


# ── Load from JSON ────────────────────────────────────────────────────────────
def load_from_json(json_path: str) -> List[CheckpointResult]:
    with open(json_path) as f:
        data = json.load(f)
    results = []
    for ckpt in data.get("checkpoints", []):
        agg = ckpt.get("aggregated", {})
        label = ckpt.get("label", "unknown")
        # Try to extract iteration from label (e.g. "model_ema_bf16" or "exp-30k")
        iteration = 0
        for part in label.replace("-", "_").split("_"):
            part = part.rstrip("k")
            if part.isdigit():
                iteration = int(part)
                if "k" in label:
                    iteration *= 1000
                break
        results.append(
            CheckpointResult(
                experiment=label,
                checkpoint=label,
                iteration=iteration,
                fds=agg.get("l1_mean", float("nan")),
                gatc=agg.get("gatc_median", agg.get("gatc_mean", float("nan"))),
                tcd=agg.get("tcd_median", agg.get("tcd_mean", float("nan"))),
                l1_early=agg.get("l1_early_c1", float("nan")),
                l1_mid=agg.get("l1_mid_c2c3", float("nan")),
                l1_late=agg.get("l1_late_c4c6", float("nan")),
                gatc_early=agg.get("gatc_early_c1_median", float("nan")),
                gatc_mid=agg.get("gatc_mid_c2c3_median", float("nan")),
                gatc_late=agg.get("gatc_late_c4c6_median", float("nan")),
                tcd_early=agg.get("tcd_early_c1_median", float("nan")),
                tcd_mid=agg.get("tcd_mid_c2c3_median", float("nan")),
                tcd_late=agg.get("tcd_late_c4c6_median", float("nan")),
            )
        )
    return results


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="Plot quant eval checkpoint comparison")
    p.add_argument("--json", type=str, default="", help="Path to JSON results file (optional)")
    p.add_argument("--output", type=str, default="output/quant_eval_comparison.png")
    p.add_argument("--dpi", type=int, default=180)
    args = p.parse_args()

    if args.json and os.path.isfile(args.json):
        results = load_from_json(args.json)
        print(f"Loaded {len(results)} checkpoints from {args.json}")
    else:
        results = RESULTS
        print(f"Using {len(results)} hardcoded results")

    plot_results(results, output_path=args.output, dpi=args.dpi)

    bar_path = args.output.rsplit(".", 1)[0] + "_bar." + args.output.rsplit(".", 1)[-1]
    plot_bar_chart(results, output_path=bar_path, dpi=args.dpi)

    chunk_l1_path = args.output.rsplit(".", 1)[0] + "_chunk_l1." + args.output.rsplit(".", 1)[-1]
    plot_chunk_panels(results, metric_name="l1", output_path=chunk_l1_path, dpi=args.dpi)

    chunk_gatc_path = args.output.rsplit(".", 1)[0] + "_chunk_gatc." + args.output.rsplit(".", 1)[-1]
    plot_chunk_panels(results, metric_name="gatc", output_path=chunk_gatc_path, dpi=args.dpi)

    chunk_tcd_path = args.output.rsplit(".", 1)[0] + "_chunk_tcd." + args.output.rsplit(".", 1)[-1]
    plot_chunk_panels(results, metric_name="tcd", output_path=chunk_tcd_path, dpi=args.dpi)


if __name__ == "__main__":
    main()
