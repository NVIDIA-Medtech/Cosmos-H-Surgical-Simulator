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
Quantitative evaluation script for Cosmos surgical simulator world model.

For a given checkpoint (or list of checkpoints), runs evaluation across 4 CMR Versius
surgical datasets, multiple episodes per dataset, and multiple seeds.  Computes three
complementary metrics per episode and reports aggregated results at the end.

METRICS:
  1. Frame Decay Score (FDS)
     - Mean per-frame L1 distance (lower = better) and Mean per-frame SSIM (higher = better)
       over all generated frames, summarised as scalar scores with std / 95% CI.
  2. GT-Anchored Tool Consistency (GATC)
     - Measures whether surgical tools remain consistent with GT tool locations.
     - Score in [-1, 1], higher = better.  Requires Medical-SAM3 segmentation.
  3. Tool Centroid Distance (TCD)
     - Hungarian-matched centroid distance between GT and generated tool instances.
     - In pixels, lower = better.  Requires Medical-SAM3 segmentation.

EVALUATION MATRIX (defaults):
  4 datasets × 5 episodes × 3 seeds = 60 episode evaluations per checkpoint.

TWO-PHASE EXECUTION:
  Phase 1 — Cosmos model: generate videos, compute Frame Decay (L1 / SSIM).
  Phase 2 — SAM3 model:  compute GATC and TCD on stored video pairs.
  This avoids loading both large models on the GPU simultaneously.

EXAMPLE USAGE:

  # Single checkpoint:
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/cosmos_h_surgical_simulator_quant_eval.py \\
    --ckpt_path /path/to/model_ema_bf16.pt \\
    --sam3_checkpoint /path/to/sam3_checkpoint.pt

  # Multiple checkpoints:
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/cosmos_h_surgical_simulator_quant_eval.py \\
    --ckpt_path /path/to/ckpt_10k.pt /path/to/ckpt_20k.pt \\
    --sam3_checkpoint /path/to/sam3_checkpoint.pt

  # Fewer episodes / seeds for quick debugging:
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/cosmos_h_surgical_simulator_quant_eval.py \\
    --ckpt_path /path/to/model_ema_bf16.pt \\
    --sam3_checkpoint /path/to/sam3_checkpoint.pt \\
    --num_episodes 1 --num_seeds 1

MEMORY NOTE:
  Video arrays are held in RAM between Phase 1 and Phase 2.  Peak RAM usage is
  approximately (num_seeds × num_datasets × num_episodes × 2 × 73 × 480 × 854 × 3)
  bytes.  For the default 60-episode config this is ~11 GB.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import traceback
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from loguru import logger

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------
try:
    import mediapy
except ImportError:
    mediapy = None

try:
    import cv2
except ImportError:
    cv2 = None
    logger.warning("OpenCV (cv2) not installed — GATC / TCD metrics will be unavailable")

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None  # type: ignore[assignment]
    logger.warning("scipy not installed — TCD metric will be unavailable")

try:
    from skimage.metrics import structural_similarity as skimage_ssim

    def compute_ssim(im1: np.ndarray, im2: np.ndarray, data_range: float = 1.0) -> float:
        if im1.shape[0] == 3:
            im1 = np.transpose(im1, (1, 2, 0))
            im2 = np.transpose(im2, (1, 2, 0))
        return float(skimage_ssim(im1, im2, data_range=data_range, channel_axis=2))

except ImportError:
    try:
        from pytorch_msssim import ssim as pt_ssim

        def compute_ssim(im1: np.ndarray, im2: np.ndarray, data_range: float = 1.0) -> float:
            t1 = torch.from_numpy(im1).float()
            t2 = torch.from_numpy(im2).float()
            if t1.dim() == 3:
                if t1.shape[-1] == 3:
                    t1 = t1.permute(2, 0, 1)
                    t2 = t2.permute(2, 0, 1)
                t1 = t1.unsqueeze(0)
                t2 = t2.unsqueeze(0)
            elif t1.dim() == 2:
                t1 = t1.unsqueeze(0).unsqueeze(0)
                t2 = t2.unsqueeze(0).unsqueeze(0)
            return float(pt_ssim(t1, t2, data_range=data_range).item())

    except ImportError:
        logger.warning("Neither scikit-image nor pytorch_msssim installed — SSIM unavailable")
        compute_ssim = None  # type: ignore[assignment]

try:
    from sam3_inference import SAM3Model
    SAM3_AVAILABLE = True
except ImportError:
    SAM3_AVAILABLE = False
    logger.warning(
        "sam3_inference not found at import time. "
        "Provide --sam3_checkpoint and ensure sam3_inference is on PYTHONPATH."
    )

from cosmos_predict2._src.predict2.action.inference.inference_pipeline import (
    ActionVideo2WorldInference,
)
from cosmos_predict2._src.predict2.action.datasets.gr00t_dreams.data.dataset import (
    LeRobotDataset,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════
NUM_FRAMES = 13
TIMESTEP_INTERVAL = 6
CHUNK_SIZE = 12
MAX_CHUNKS = 6

# Verify / adjust episode IDs for your data.
DATASET_CONFIGS = [
    {
        "path": "/CMR_Versius/prostatectomy_360p",
        "name": "prostatectomy",
        "episode_ids": [10155, 10156, 10157, 10158, 10159],
    },
    {
        "path": "/CMR_Versius/inguinal_hernia_360p",
        "name": "inguinal_hernia",
        "episode_ids": [7186, 7187, 7188, 7189, 7190],
    },
    {
        "path": "/CMR_Versius/hysterectomy_360p",
        "name": "hysterectomy",
        "episode_ids": [7369, 7370, 7371, 7372, 7373],
    },
    {
        "path": "/CMR_Versius/cholecystectomy_360p",
        "name": "cholecystectomy",
        "episode_ids": [4646, 4647, 4648, 4649, 4650],
    },
]

CHECKPOINT_CONFIGS: dict = {}


# ═══════════════════════════════════════════════════════════════════════════════
# SAM3 Tool Segmenter
# ═══════════════════════════════════════════════════════════════════════════════
def _split_mask_to_instances(
    unified: np.ndarray, erode_radius: int = 3,
) -> List[np.ndarray]:
    """Split a unified binary mask into per-instance masks.

    Uses morphological erosion → connected-components → dilation so that
    tools whose masks are barely touching are still separated into distinct
    instances.  Each returned mask covers the *original* (pre-erosion) pixels
    that belong to that component.
    """
    assert cv2 is not None
    if unified.sum() == 0:
        return []

    src = unified.astype(np.uint8)

    # Erode to break thin bridges between touching tools
    if erode_radius > 0:
        k = 2 * erode_radius + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        eroded = cv2.erode(src, kernel, iterations=1)
    else:
        eroded = src

    # If erosion wiped everything, fall back to un-eroded mask
    if eroded.sum() == 0:
        eroded = src

    num_labels, labels_eroded = cv2.connectedComponents(eroded)
    if num_labels <= 1:
        return [src]

    # Assign each original-mask pixel to its nearest eroded component via
    # watershed-style dilation of the eroded labels into the original mask.
    labels_full = labels_eroded.copy()
    # Dilate each label iteratively until the full mask is covered
    remaining = (src > 0) & (labels_full == 0)
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    for _ in range(max(unified.shape)):
        if not remaining.any():
            break
        dilated = cv2.dilate(labels_full.astype(np.uint8), dilate_kernel, iterations=1)
        # Only fill pixels that belong to the original mask and are still unassigned
        fill_mask = remaining & (dilated > 0)
        labels_full[fill_mask] = dilated[fill_mask]
        remaining = (src > 0) & (labels_full == 0)

    instances: List[np.ndarray] = []
    for i in range(1, num_labels):
        inst = (labels_full == i).astype(np.uint8)
        if inst.sum() > 0:
            instances.append(inst)
    return instances


class SAM3ToolSegmenter:
    """Wraps Medical-SAM3 for unified tool mask (GATC) and instance masks (TCD).

    Instance separation strategy (in order of preference):
      1. **Native SAM3 per-instance masks** — access the SAM3 processor's
         ``set_text_prompt`` which returns ``state["masks"]`` of shape
         ``(N, H, W)`` with one binary mask per detected instance, plus
         ``state["scores"]`` for confidence filtering.
      2. **Morphology-enhanced connected components** (fallback) — when the
         processor is unavailable, get the unified mask and split via
         erode → connected-components → label expansion.

    The SAM3Model wrapper's ``predict_text`` only returns the single best mask
    and ``predict_text_union`` unions all masks into one.  Neither exposes
    individual instances, so strategy 1 bypasses the wrapper.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
        prompt: str = "surgical tool",
        score_threshold: float = 0.3,
        erode_radius: int = 3,
    ):
        if not SAM3_AVAILABLE:
            raise RuntimeError(
                "sam3_inference not importable. Install Medical-SAM3 and ensure "
                "sam3_inference is on PYTHONPATH."
            )
        self.sam3 = SAM3Model(checkpoint_path=checkpoint_path, confidence_threshold=0.1)
        self.sam3.load_model()  # force eager load so processor is available
        self.prompt = prompt
        self.score_threshold = score_threshold
        self.device = device
        self.erode_radius = erode_radius

        # Verify the internal processor is available (requires load_model() first)
        if not hasattr(self.sam3, "processor") or self.sam3.processor is None:
            raise RuntimeError(
                "Medical-SAM3 processor is None after load_model(). "
                "Check SAM3Model checkpoint and version."
            )
        if not hasattr(self.sam3.processor, "set_text_prompt"):
            raise RuntimeError(
                "Medical-SAM3 processor.set_text_prompt is required for this evaluation. "
                "Found processor without set_text_prompt method."
            )

        mode = "native (processor.set_text_prompt)"
        logger.info(
            f"SAM3ToolSegmenter initialised (prompt='{prompt}', score_thr={score_threshold}, "
            f"instance_mode={mode})"
        )

    def _get_all_masks_and_unified(
        self, frame_rgb: np.ndarray,
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        """Run SAM3 once and return (instance_masks, unified_mask).

        Tries the native processor first for true per-instance output,
        then falls back to unified mask + morphological splitting.
        """
        state = self.sam3.encode_image(frame_rgb)
        H, W = frame_rgb.shape[:2]

        # Native per-instance masks via processor (required path)
        text_state = self.sam3.processor.set_text_prompt(self.prompt, state)
        raw_masks = text_state.get("masks")
        raw_scores = text_state.get("scores")

        if raw_masks is None or len(raw_masks) == 0:
            return [], np.zeros((H, W), dtype=np.uint8)

        if isinstance(raw_masks, torch.Tensor):
            masks_np = (raw_masks.float().cpu().numpy() > 0).astype(np.uint8)
        else:
            masks_np = (np.asarray(raw_masks) > 0).astype(np.uint8)

        # Ensure shape (N, H, W)
        if masks_np.ndim == 2:
            masks_np = masks_np[None, ...]

        # Filter by score
        if raw_scores is not None and len(raw_scores) == masks_np.shape[0]:
            if isinstance(raw_scores, torch.Tensor):
                scores_np = raw_scores.float().cpu().numpy()
            else:
                scores_np = np.asarray(raw_scores, dtype=np.float32)
            keep = scores_np >= self.score_threshold
            if keep.any():
                masks_np = masks_np[keep]
            else:
                masks_np = masks_np[:0]

        instances = [
            np.squeeze(masks_np[i])
            for i in range(masks_np.shape[0])
            if np.squeeze(masks_np[i]).sum() > 0
        ]
        if not instances:
            return [], np.zeros((H, W), dtype=np.uint8)

        unified = np.zeros((H, W), dtype=np.uint8)
        for m in instances:
            unified = np.maximum(unified, m)
        return instances, unified

    # ---- unified mask (used by GATC) ----
    def predict_mask(self, frame_rgb: np.ndarray) -> np.ndarray:
        """Return a binary tool mask (H, W) uint8 — union of all tool instances."""
        _, unified = self._get_all_masks_and_unified(frame_rgb)
        return unified

    # ---- per-instance masks (used by TCD) ----
    def predict_instances(self, frame_rgb: np.ndarray) -> List[np.ndarray]:
        """Return a list of binary masks, one per tool instance."""
        assert cv2 is not None, "cv2 required for instance segmentation"
        instances, _ = self._get_all_masks_and_unified(frame_rgb)
        return instances


def segment_video_frames(
    segmenter: SAM3ToolSegmenter, frames: np.ndarray,
) -> Tuple[List[np.ndarray], List[List[np.ndarray]]]:
    """Segment every frame, returning (unified_masks, instance_masks_per_frame).

    The unified mask is the union of all tool pixels (for GATC).
    The instance masks are per-tool (for TCD).  Both are derived from a single
    ``_get_all_masks_and_unified`` call per frame — one SAM3 inference each.
    """
    unified_masks: List[np.ndarray] = []
    instance_masks: List[List[np.ndarray]] = []
    for t in range(frames.shape[0]):
        instances, unified = segmenter._get_all_masks_and_unified(frames[t])
        unified_masks.append(unified)
        instance_masks.append(instances)
    return unified_masks, instance_masks


# ═══════════════════════════════════════════════════════════════════════════════
# Metric 1 — Frame Decay Score (FDS)
# ═══════════════════════════════════════════════════════════════════════════════
# Chunk boundaries for per-chunk breakdown (frame indices within generated frames,
# i.e. after skipping the conditioning frame).
# Chunk 1 = frames 0..11 (early), Chunks 2-3 = frames 12..35 (mid), Chunks 4-6 = frames 36..71 (late)
CHUNK_RANGES = {
    "early_c1":  (0, CHUNK_SIZE),                          # frames 0-11
    "mid_c2c3":  (CHUNK_SIZE, CHUNK_SIZE * 3),             # frames 12-35
    "late_c4c6": (CHUNK_SIZE * 3, CHUNK_SIZE * MAX_CHUNKS),  # frames 36-71
}


def _chunk_mean(values: List[float], start: int, end: int) -> float:
    sl = values[start:end]
    return float(np.mean(sl)) if sl else float("nan")


def _chunk_nanmedian(values: np.ndarray, start: int, end: int) -> float:
    sl = values[start:end]
    if sl.size == 0:
        return float("nan")
    return float(np.nanmedian(sl))


def compute_frame_decay(
    gt_video: np.ndarray, gen_video: np.ndarray,
) -> Dict[str, object]:
    """Compute per-frame L1 and SSIM, returning scalar summaries.

    Skips the first frame (conditioning frame).
    Images normalised to [-1, 1] for L1 and SSIM (data_range=2.0).
    Also returns per-chunk breakdowns (early / mid / late).
    """
    T = min(gt_video.shape[0], gen_video.shape[0])
    gt_norm = gt_video[:T].astype(np.float32) / 127.5 - 1.0
    gen_norm = gen_video[:T].astype(np.float32) / 127.5 - 1.0

    l1_list: List[float] = []
    ssim_list: List[float] = []

    for t in range(1, T):
        l1_list.append(float(np.mean(np.abs(gen_norm[t] - gt_norm[t]))))
        if compute_ssim is not None:
            ssim_list.append(compute_ssim(gen_norm[t], gt_norm[t], data_range=2.0))

    mean_l1 = float(np.mean(l1_list)) if l1_list else float("nan")
    std_l1 = float(np.std(l1_list)) if l1_list else float("nan")
    mean_ssim = float(np.mean(ssim_list)) if ssim_list else float("nan")
    std_ssim = float(np.std(ssim_list)) if ssim_list else float("nan")

    l1_slope = float("nan")
    if len(l1_list) > 1:
        l1_slope = float(np.polyfit(np.arange(len(l1_list)), l1_list, 1)[0])

    # Per-chunk breakdown
    per_chunk: Dict[str, Dict[str, float]] = {}
    for cname, (s, e) in CHUNK_RANGES.items():
        per_chunk[cname] = {
            "l1": _chunk_mean(l1_list, s, e),
            "ssim": _chunk_mean(ssim_list, s, e) if ssim_list else float("nan"),
        }

    return {
        "mean_l1": mean_l1,
        "std_l1": std_l1,
        "mean_ssim": mean_ssim,
        "std_ssim": std_ssim,
        "l1_slope": l1_slope,
        "l1_per_frame": l1_list,
        "ssim_per_frame": ssim_list,
        "num_frames": len(l1_list),
        "per_chunk": per_chunk,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Metric 2 — GT-Anchored Tool Consistency (GATC)
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class GATCConfig:
    k_translation_px: int = 3
    dilate_radius_px: int = 10
    time_tolerance: Tuple[int, ...] = (-1, 0, 1)
    eps: float = 1e-6
    min_mask_pixels: int = 50
    use_gradient_mag: bool = False  # grayscale ZNCC by default (more sensitive than gradient mag)


def _to_gray_float(frame_rgb: np.ndarray) -> np.ndarray:
    assert cv2 is not None
    return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0


def _grad_mag(gray01: np.ndarray) -> np.ndarray:
    assert cv2 is not None
    gx = cv2.Sobel(gray01, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray01, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


def _dilate_mask(mask: np.ndarray, radius_px: int) -> np.ndarray:
    assert cv2 is not None
    if radius_px <= 0:
        return mask.astype(bool)
    k = 2 * radius_px + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.dilate(mask.astype(np.uint8) * 255, kernel, iterations=1) > 0


def _masked_zncc(a: np.ndarray, b: np.ndarray, mask: np.ndarray, eps: float = 1e-6) -> float:
    idx = mask
    if idx.sum() < 10:
        return float("nan")
    x = a[idx].astype(np.float32)
    y = b[idx].astype(np.float32)
    x = x - x.mean()
    y = y - y.mean()
    denom = (np.sqrt((x * x).mean()) * np.sqrt((y * y).mean())) + eps
    return float((x * y).mean() / denom)


def _shift_image(img: np.ndarray, dx: int, dy: int) -> np.ndarray:
    assert cv2 is not None
    h, w = img.shape[:2]
    M = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def _tool_presence_penalty(
    grad_gt: np.ndarray, grad_gen: np.ndarray, mask: np.ndarray, eps: float = 1e-6,
) -> float:
    idx = mask
    if idx.sum() < 10:
        return float("nan")
    num = float(np.mean(grad_gen[idx]))
    den = float(np.mean(grad_gt[idx])) + eps
    if den <= 0:
        return 1.0
    return float(min(1.0, num / den))


def compute_gatc(
    gt_frames: np.ndarray,
    gen_frames: np.ndarray,
    gt_masks: List[np.ndarray],
    cfg: GATCConfig = GATCConfig(),
) -> Dict[str, object]:
    """Compute GATC score over provided frames.

    Args:
        gt_frames:  (T, H, W, 3) uint8
        gen_frames: (T, H, W, 3) uint8
        gt_masks:   list of T binary masks (H, W) from SAM3 on GT
        cfg:        GATC hyper-parameters

    Returns:
        dict with 'score', 'coverage', and per-frame details.
    """
    if cv2 is None:
        return {"score": float("nan"), "coverage": 0.0}

    T = min(gt_frames.shape[0], gen_frames.shape[0], len(gt_masks))
    gt_frames = gt_frames[:T]
    gen_frames = gen_frames[:T]
    gt_masks = gt_masks[:T]

    gt_repr, gen_repr, gt_grad, gen_grad = [], [], [], []
    for t in range(T):
        g_gt = _to_gray_float(gt_frames[t])
        g_gen = _to_gray_float(gen_frames[t])
        if cfg.use_gradient_mag:
            gt_repr.append(_grad_mag(g_gt))
            gen_repr.append(_grad_mag(g_gen))
        else:
            gt_repr.append(g_gt)
            gen_repr.append(g_gen)
        gt_grad.append(_grad_mag(g_gt))
        gen_grad.append(_grad_mag(g_gen))

    spt_list = np.full(T, np.nan, dtype=np.float32)
    valid = np.zeros(T, dtype=bool)

    for t in range(T):
        m = _dilate_mask(gt_masks[t].astype(bool), cfg.dilate_radius_px)
        if int(m.sum()) < cfg.min_mask_pixels:
            continue

        pt = _tool_presence_penalty(gt_grad[t], gen_grad[t], m, eps=cfg.eps)

        best = -np.inf
        for dt in cfg.time_tolerance:
            tt = t + dt
            if tt < 0 or tt >= T:
                continue
            Y = gen_repr[tt]
            k = cfg.k_translation_px
            for dy in range(-k, k + 1):
                for dx in range(-k, k + 1):
                    Y_shift = _shift_image(Y, dx=dx, dy=dy)
                    s = _masked_zncc(gt_repr[t], Y_shift, m, eps=cfg.eps)
                    if not math.isnan(s) and s > best:
                        best = s

        if best == -np.inf:
            continue

        spt_list[t] = float(best) * pt if not math.isnan(pt) else float("nan")
        valid[t] = True

    vals = spt_list[valid]
    score = float(np.nanmedian(vals)) if vals.size > 0 else float("nan")
    coverage = float(valid.mean()) * 100.0 if valid.size > 0 else 0.0

    per_chunk = {
        cname: _chunk_nanmedian(spt_list, s, e)
        for cname, (s, e) in CHUNK_RANGES.items()
    }

    return {"score": score, "coverage": coverage, "per_chunk": per_chunk}


# ═══════════════════════════════════════════════════════════════════════════════
# Metric 3 — Tool Centroid Distance (TCD)
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class TCDConfig:
    min_instance_area_px: int = 50
    D_miss_mode: str = "half_diag"  # less harsh than full diagonal
    D_miss_custom: float = 0.0


def _mask_centroid_xy(mask: np.ndarray) -> Optional[Tuple[float, float]]:
    assert cv2 is not None
    m = (mask.astype(np.uint8) > 0).astype(np.uint8)
    if m.sum() == 0:
        return None
    M = cv2.moments(m, binaryImage=True)
    if M["m00"] == 0:
        return None
    return (float(M["m10"] / M["m00"]), float(M["m01"] / M["m00"]))


def _extract_centroids(
    instance_masks: Sequence[np.ndarray], min_area_px: int = 50,
) -> np.ndarray:
    centroids: List[Tuple[float, float]] = []
    for m in instance_masks:
        if int((m > 0).sum()) < min_area_px:
            continue
        c = _mask_centroid_xy(m)
        if c is not None:
            centroids.append(c)
    if not centroids:
        return np.zeros((0, 2), dtype=np.float32)
    return np.asarray(centroids, dtype=np.float32)


def _tcd_frame(C_gt: np.ndarray, C_gen: np.ndarray, D_miss: float) -> float:
    N = C_gt.shape[0]
    if N == 0:
        return float("nan")
    if C_gen.shape[0] == 0:
        return float(D_miss)
    diff = C_gt[:, None, :] - C_gen[None, :, :]
    D = np.sqrt(np.sum(diff * diff, axis=2)).astype(np.float32)
    assert linear_sum_assignment is not None
    row_ind, col_ind = linear_sum_assignment(D)
    matched_cost = float(D[row_ind, col_ind].sum())
    unmatched_gt = N - len(row_ind)
    return float((matched_cost + unmatched_gt * D_miss) / N)


def compute_tcd(
    gt_instance_masks: List[List[np.ndarray]],
    gen_instance_masks: List[List[np.ndarray]],
    H: int,
    W: int,
    cfg: TCDConfig = TCDConfig(),
) -> Dict[str, object]:
    """Compute TCD from pre-computed per-frame instance masks.

    Args:
        gt_instance_masks:  list (length T) of lists-of-binary-masks
        gen_instance_masks: same shape
        H, W: frame dimensions (for D_miss computation)
        cfg: TCD config

    Returns:
        dict with 'score', 'coverage'.
    """
    if cv2 is None or linear_sum_assignment is None:
        return {"score": float("nan"), "coverage": 0.0}

    T = min(len(gt_instance_masks), len(gen_instance_masks))
    diag = float(math.sqrt(H * H + W * W))
    if cfg.D_miss_mode == "half_diag":
        D_miss = 0.5 * diag
    elif cfg.D_miss_mode == "diag":
        D_miss = diag
    elif cfg.D_miss_mode == "custom":
        D_miss = float(cfg.D_miss_custom) if cfg.D_miss_custom > 0 else 0.5 * diag
    else:
        D_miss = 0.5 * diag

    per_frame = np.full(T, np.nan, dtype=np.float32)
    for t in range(T):
        C_gt = _extract_centroids(gt_instance_masks[t], min_area_px=cfg.min_instance_area_px)
        C_gen = _extract_centroids(gen_instance_masks[t], min_area_px=cfg.min_instance_area_px)
        per_frame[t] = _tcd_frame(C_gt, C_gen, D_miss)

    valid = ~np.isnan(per_frame)
    score = float(np.nanmedian(per_frame[valid])) if valid.any() else float("nan")
    coverage = float(valid.mean()) * 100.0 if valid.size > 0 else 0.0

    per_chunk = {
        cname: _chunk_nanmedian(per_frame, s, e)
        for cname, (s, e) in CHUNK_RANGES.items()
    }

    return {"score": score, "coverage": coverage, "per_chunk": per_chunk}


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset / model utilities (from original frame_decay_analysis_cmr.py)
# ═══════════════════════════════════════════════════════════════════════════════
def build_episode_index_map(dataset: LeRobotDataset) -> dict:
    inner_dataset = dataset.lerobot_datasets[0]
    all_steps = inner_dataset._all_steps
    episode_map: dict[int, list[tuple[int, int]]] = {}
    for dataset_idx, (episode_id, base_index) in enumerate(all_steps):
        episode_map.setdefault(episode_id, []).append((dataset_idx, base_index))
    for episode_id in episode_map:
        episode_map[episode_id].sort(key=lambda x: x[1])
    return episode_map


def find_chunk_indices(
    episode_map: dict,
    episode_id: int,
    chunk_size: int = CHUNK_SIZE,
    timestep_interval: int = TIMESTEP_INTERVAL,
    max_chunks: int = MAX_CHUNKS,
) -> Optional[List[int]]:
    if episode_id not in episode_map:
        return None
    entries = episode_map[episode_id]
    bi2di = {base_idx: ds_idx for ds_idx, base_idx in entries}
    if 0 not in bi2di:
        return None
    stride = chunk_size * timestep_interval
    indices: List[int] = []
    base = 0
    while base in bi2di and len(indices) < max_chunks:
        indices.append(bi2di[base])
        base += stride
    return indices


def load_dataset(dataset_path: str, data_split: str) -> LeRobotDataset:
    logger.info(f"Loading LeRobotDataset from {dataset_path} (split={data_split})")
    return LeRobotDataset(
        num_frames=NUM_FRAMES,
        time_division_factor=4,
        time_division_remainder=1,
        max_pixels=1920 * 1080,
        data_file_keys=("video",),
        image_file_extension=("jpg", "jpeg", "png", "webp"),
        video_file_extension=("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm"),
        repeat=1,
        args=None,
        dataset_path=dataset_path,
        data_split=data_split,
        embodiment="cmr_versius",
        downscaled_res=False,
    )


def setup_inference_pipeline(
    experiment: str, ckpt_path: str, s3_cred: str, context_parallel_size: int,
) -> ActionVideo2WorldInference:
    logger.info(f"Loading Cosmos model from {ckpt_path}")
    v2w = ActionVideo2WorldInference(
        experiment, ckpt_path, s3_cred, context_parallel_size=context_parallel_size,
    )
    mem_gb = torch.cuda.memory_allocated() / (1024 ** 3)
    logger.info(f"GPU memory after model load: {mem_gb:.2f} GB")
    return v2w


# ═══════════════════════════════════════════════════════════════════════════════
# Video generation
# ═══════════════════════════════════════════════════════════════════════════════
def generate_episode_video(
    video2world: ActionVideo2WorldInference,
    dataset: LeRobotDataset,
    episode_map: dict,
    episode_id: int,
    seed: int,
    guidance: float = 0,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Autoregressively generate a full video for one episode.

    Returns (gt_video, gen_video) each of shape (T, H, W, 3) uint8,
    or (None, None) on failure.
    """
    chunk_indices = find_chunk_indices(episode_map, episode_id)
    if chunk_indices is None:
        logger.warning(f"Episode {episode_id}: no base_index=0 in split, skipping")
        return None, None
    if len(chunk_indices) < MAX_CHUNKS:
        logger.warning(
            f"Episode {episode_id}: only {len(chunk_indices)} chunks (need {MAX_CHUNKS}), skipping"
        )
        return None, None

    predicted_chunks: List[np.ndarray] = []
    gt_chunks: List[np.ndarray] = []
    current_frame: Optional[np.ndarray] = None

    for ci, ds_idx in enumerate(chunk_indices):
        data = dataset[ds_idx]
        video = data["video"].permute(1, 2, 3, 0).numpy()  # (T, H, W, C)
        actions = data["action"].numpy().astype(np.float32)

        if ci == 0:
            current_frame = video[0]

        gt_chunks.append(video)
        next_frame, video_chunk = video2world.step_inference(
            img_array=current_frame,
            action=actions,
            guidance=guidance,
            seed=seed + ci,
            num_latent_conditional_frames=1,
        )
        predicted_chunks.append(video_chunk)
        current_frame = next_frame

    if not predicted_chunks:
        return None, None

    gen_video = np.concatenate(
        [predicted_chunks[0]] + [c[1:] for c in predicted_chunks[1:]], axis=0,
    )
    gt_video = np.concatenate(
        [gt_chunks[0]] + [c[1:] for c in gt_chunks[1:]], axis=0,
    )
    min_len = min(len(gt_video), len(gen_video))
    return gt_video[:min_len], gen_video[:min_len]


# ═══════════════════════════════════════════════════════════════════════════════
# Checkpoint evaluation (two-phase)
# ═══════════════════════════════════════════════════════════════════════════════
def evaluate_checkpoint(
    args: argparse.Namespace,
    ckpt_path: str,
    label: str,
    seeds: List[int],
) -> List[dict]:
    """Evaluate one checkpoint.  Returns a list of per-episode result dicts."""

    # Pre-load datasets (reused across seeds)
    datasets: dict[str, dict] = {}
    for cfg in DATASET_CONFIGS:
        ep_ids = cfg["episode_ids"][: args.num_episodes]
        ds = load_dataset(cfg["path"], args.data_split)
        datasets[cfg["name"]] = {
            "dataset": ds,
            "episode_map": build_episode_index_map(ds),
            "episode_ids": ep_ids,
        }

    # ------------------------------------------------------------------
    # Phase 1: generate videos + frame decay
    # ------------------------------------------------------------------
    logger.info(f"\n{'='*80}")
    logger.info(f"PHASE 1 — Video generation: {label}")
    logger.info(f"  Path:    {ckpt_path}")
    logger.info(f"  Seeds:   {seeds}")
    logger.info(f"  Datasets: {len(DATASET_CONFIGS)} × {args.num_episodes} episodes = "
                f"{len(DATASET_CONFIGS) * args.num_episodes} per seed, "
                f"{len(DATASET_CONFIGS) * args.num_episodes * len(seeds)} total")
    logger.info(f"{'='*80}\n")

    video2world = setup_inference_pipeline(
        args.experiment, ckpt_path, args.s3_cred, args.context_parallel_size,
    )

    episode_results: List[dict] = []
    total_evals = len(seeds) * len(DATASET_CONFIGS) * args.num_episodes
    eval_idx = 0
    phase1_start = time.time()

    for seed in seeds:
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        for dcfg in DATASET_CONFIGS:
            ds_name = dcfg["name"]
            ds_info = datasets[ds_name]

            for ep_id in ds_info["episode_ids"]:
                eval_idx += 1
                logger.info(
                    f"[{eval_idx}/{total_evals}] seed={seed}  {ds_name}  ep={ep_id}"
                )

                try:
                    gt_video, gen_video = generate_episode_video(
                        video2world, ds_info["dataset"], ds_info["episode_map"],
                        ep_id, seed, args.guidance,
                    )
                except Exception:
                    logger.error(f"  Generation failed:\n{traceback.format_exc()}")
                    continue

                if gt_video is None:
                    continue

                fd = compute_frame_decay(gt_video, gen_video)

                # Sanity check: verify generated video actually differs from GT
                gt_f1 = gt_video[1].astype(np.float32)
                gen_f1 = gen_video[1].astype(np.float32)
                div_f1 = float(np.mean(np.abs(gen_f1 - gt_f1)))
                gt_fl = gt_video[-1].astype(np.float32)
                gen_fl = gen_video[-1].astype(np.float32)
                div_fl = float(np.mean(np.abs(gen_fl - gt_fl)))

                pc = fd["per_chunk"]
                logger.info(
                    f"  FDS: L1={fd['mean_l1']:.4f}  SSIM={fd['mean_ssim']:.4f}  "
                    f"({fd['num_frames']} frames)  "
                    f"[early={pc['early_c1']['l1']:.4f}  mid={pc['mid_c2c3']['l1']:.4f}  "
                    f"late={pc['late_c4c6']['l1']:.4f}]"
                )
                logger.info(
                    f"  Sanity: |gen-gt| frame1={div_f1:.1f}  frameLast={div_fl:.1f}  "
                    f"(uint8 scale, expect >0 if model is generating)"
                )

                entry: dict = {
                    "checkpoint_label": label,
                    "checkpoint_path": ckpt_path,
                    "dataset_name": ds_name,
                    "episode_id": int(ep_id),
                    "seed": int(seed),
                    "num_frames": int(gt_video.shape[0]),
                    "frame_decay": fd,
                }

                entry["_gt_video"] = gt_video
                entry["_gen_video"] = gen_video

                if args.save_videos and mediapy is not None:
                    vid_dir = os.path.join(args.save_path, label, ds_name)
                    os.makedirs(os.path.join(vid_dir, "generated"), exist_ok=True)
                    os.makedirs(os.path.join(vid_dir, "comparison"), exist_ok=True)
                    tag = f"ep{ep_id:05d}_seed{seed}"
                    mediapy.write_video(
                        os.path.join(vid_dir, "generated", f"{tag}.mp4"),
                        gen_video, fps=args.save_fps,
                    )
                    cmp = np.concatenate([gt_video, gen_video], axis=2)
                    mediapy.write_video(
                        os.path.join(vid_dir, "comparison", f"{tag}.mp4"),
                        cmp, fps=args.save_fps,
                    )

                episode_results.append(entry)

    video2world.cleanup()
    del video2world
    torch.cuda.empty_cache()
    phase1_dur = time.time() - phase1_start
    logger.info(f"Phase 1 complete in {phase1_dur / 60:.1f} min — {len(episode_results)} episodes")

    # ------------------------------------------------------------------
    # Phase 2: SAM3-based metrics (GATC + TCD)
    # ------------------------------------------------------------------
    if not episode_results:
        return episode_results

    logger.info(f"\n{'='*80}")
    logger.info(f"PHASE 2 — SAM3 metrics (GATC + TCD): {label}")
    logger.info(f"{'='*80}\n")

    phase2_start = time.time()
    segmenter = SAM3ToolSegmenter(
        args.sam3_checkpoint,
        device="cuda",
        prompt=args.sam3_prompt,
        score_threshold=args.sam3_score_threshold,
    )

    gatc_cfg = GATCConfig(
        k_translation_px=args.gatc_k,
        use_gradient_mag=args.gatc_use_grad,
    )
    tcd_cfg = TCDConfig()

    for i, entry in enumerate(episode_results):
        gt_video = entry.pop("_gt_video", None)
        gen_video = entry.pop("_gen_video", None)
        if gt_video is None or gen_video is None:
            continue

        gt_eval = gt_video[1:]
        gen_eval = gen_video[1:]

        logger.info(
            f"  SAM3 [{i + 1}/{len(episode_results)}] "
            f"seed={entry['seed']}  {entry['dataset_name']}  ep={entry['episode_id']}"
        )

        try:
            gt_masks, gt_instances = segment_video_frames(segmenter, gt_eval)
            _, gen_instances = segment_video_frames(segmenter, gen_eval)

            gatc_result = compute_gatc(gt_eval, gen_eval, gt_masks, gatc_cfg)
            tcd_result = compute_tcd(
                gt_instances, gen_instances, gt_eval.shape[1], gt_eval.shape[2], tcd_cfg,
            )

            entry["gatc"] = gatc_result
            entry["tcd"] = tcd_result
            gpc = gatc_result.get("per_chunk", {})
            tpc = tcd_result.get("per_chunk", {})
            logger.info(
                f"    GATC={gatc_result['score']:.4f} (cov {gatc_result['coverage']:.1f}%)  "
                f"TCD={tcd_result['score']:.1f}px (cov {tcd_result['coverage']:.1f}%)"
            )
            logger.info(
                "    Per-chunk: "
                f"GATC[e/m/l]={_fmt(gpc.get('early_c1', float('nan')))} / "
                f"{_fmt(gpc.get('mid_c2c3', float('nan')))} / "
                f"{_fmt(gpc.get('late_c4c6', float('nan')))}   "
                f"TCD[e/m/l]={_fmt(tpc.get('early_c1', float('nan')), 2)} / "
                f"{_fmt(tpc.get('mid_c2c3', float('nan')), 2)} / "
                f"{_fmt(tpc.get('late_c4c6', float('nan')), 2)}"
            )
        except Exception:
            logger.error(f"    SAM3 metric failed:\n{traceback.format_exc()}")

        del gt_video, gen_video

    del segmenter
    torch.cuda.empty_cache()
    phase2_dur = time.time() - phase2_start
    logger.info(f"Phase 2 complete in {phase2_dur / 60:.1f} min")

    return episode_results


# ═══════════════════════════════════════════════════════════════════════════════
# Aggregation helpers
# ═══════════════════════════════════════════════════════════════════════════════
def _ci95(values: np.ndarray) -> Tuple[float, float]:
    """Mean ± 1.96 * SE."""
    if len(values) < 2:
        m = float(np.mean(values)) if len(values) else float("nan")
        return (m, m)
    m = float(np.mean(values))
    se = float(np.std(values, ddof=1) / np.sqrt(len(values)))
    return (m - 1.96 * se, m + 1.96 * se)


def aggregate_checkpoint_results(episodes: List[dict]) -> dict:
    """Aggregate per-episode results into checkpoint-level statistics."""
    l1_vals = np.array([e["frame_decay"]["mean_l1"] for e in episodes])
    ssim_vals = np.array([e["frame_decay"]["mean_ssim"] for e in episodes])
    slope_vals = np.array([e["frame_decay"]["l1_slope"] for e in episodes])

    agg: dict = {
        "num_episodes": len(episodes),
        "l1_mean": float(np.mean(l1_vals)),
        "l1_std": float(np.std(l1_vals)),
        "l1_ci95": _ci95(l1_vals),
        "ssim_mean": float(np.mean(ssim_vals)),
        "ssim_std": float(np.std(ssim_vals)),
        "ssim_ci95": _ci95(ssim_vals),
        "l1_slope_mean": float(np.mean(slope_vals)),
    }

    # Per-chunk L1 breakdown
    for cname in CHUNK_RANGES:
        c_vals = np.array([
            e["frame_decay"]["per_chunk"][cname]["l1"]
            for e in episodes
            if "per_chunk" in e["frame_decay"] and cname in e["frame_decay"]["per_chunk"]
        ])
        if c_vals.size > 0:
            agg[f"l1_{cname}"] = float(np.mean(c_vals))

    gatc_vals = np.array([
        e["gatc"]["score"] for e in episodes
        if "gatc" in e and not math.isnan(e["gatc"]["score"])
    ])
    if gatc_vals.size > 0:
        agg["gatc_median"] = float(np.median(gatc_vals))
        agg["gatc_mean"] = float(np.mean(gatc_vals))
        agg["gatc_std"] = float(np.std(gatc_vals))
        agg["gatc_ci95"] = _ci95(gatc_vals)
        gatc_cov = np.array([e["gatc"]["coverage"] for e in episodes if "gatc" in e])
        agg["gatc_coverage_mean"] = float(np.mean(gatc_cov))
        for cname in CHUNK_RANGES:
            c_vals = np.array([
                e["gatc"]["per_chunk"].get(cname, float("nan"))
                for e in episodes
                if "gatc" in e and "per_chunk" in e["gatc"]
            ])
            c_vals = c_vals[~np.isnan(c_vals)]
            if c_vals.size > 0:
                agg[f"gatc_{cname}_median"] = float(np.median(c_vals))

    tcd_vals = np.array([
        e["tcd"]["score"] for e in episodes
        if "tcd" in e and not math.isnan(e["tcd"]["score"])
    ])
    if tcd_vals.size > 0:
        agg["tcd_median"] = float(np.median(tcd_vals))
        agg["tcd_mean"] = float(np.mean(tcd_vals))
        agg["tcd_std"] = float(np.std(tcd_vals))
        agg["tcd_ci95"] = _ci95(tcd_vals)
        tcd_cov = np.array([e["tcd"]["coverage"] for e in episodes if "tcd" in e])
        agg["tcd_coverage_mean"] = float(np.mean(tcd_cov))
        for cname in CHUNK_RANGES:
            c_vals = np.array([
                e["tcd"]["per_chunk"].get(cname, float("nan"))
                for e in episodes
                if "tcd" in e and "per_chunk" in e["tcd"]
            ])
            c_vals = c_vals[~np.isnan(c_vals)]
            if c_vals.size > 0:
                agg[f"tcd_{cname}_median"] = float(np.median(c_vals))

    # Per-dataset breakdown
    ds_names = sorted(set(e["dataset_name"] for e in episodes))
    per_ds: dict[str, dict] = {}
    for ds in ds_names:
        ds_eps = [e for e in episodes if e["dataset_name"] == ds]
        ds_l1 = np.array([e["frame_decay"]["mean_l1"] for e in ds_eps])
        ds_ssim = np.array([e["frame_decay"]["mean_ssim"] for e in ds_eps])
        d: dict = {
            "count": len(ds_eps),
            "l1_mean": float(np.mean(ds_l1)),
            "ssim_mean": float(np.mean(ds_ssim)),
        }
        ds_gatc = np.array([
            e["gatc"]["score"] for e in ds_eps
            if "gatc" in e and not math.isnan(e["gatc"]["score"])
        ])
        if ds_gatc.size > 0:
            d["gatc_median"] = float(np.median(ds_gatc))
        ds_tcd = np.array([
            e["tcd"]["score"] for e in ds_eps
            if "tcd" in e and not math.isnan(e["tcd"]["score"])
        ])
        if ds_tcd.size > 0:
            d["tcd_median"] = float(np.median(ds_tcd))
        per_ds[ds] = d
    agg["per_dataset"] = per_ds

    return agg


# ═══════════════════════════════════════════════════════════════════════════════
# Report generation
# ═══════════════════════════════════════════════════════════════════════════════
def _fmt(v: float, prec: int = 4) -> str:
    if math.isnan(v):
        return "  N/A  "
    return f"{v:.{prec}f}"


def _fmt_ci(ci: Tuple[float, float], prec: int = 4) -> str:
    return f"[{ci[0]:.{prec}f} – {ci[1]:.{prec}f}]"


def print_quantitative_report(
    all_checkpoint_results: List[Tuple[str, str, dict, List[dict]]],
    seeds: List[int],
    args: argparse.Namespace,
) -> None:
    """Print the full benchmarking report to the log.

    all_checkpoint_results: list of (label, path, aggregated_dict, episode_list)
    """
    logger.info("")
    logger.info("=" * 90)
    logger.info("  COSMOS SURGICAL SIMULATOR — QUANTITATIVE EVALUATION REPORT")
    logger.info("=" * 90)
    logger.info(f"  Date:           {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"  Datasets:       {len(DATASET_CONFIGS)} "
                f"({', '.join(c['name'] for c in DATASET_CONFIGS)})")
    logger.info(f"  Episodes/DS:    {args.num_episodes}")
    logger.info(f"  Seeds:          {seeds}")
    logger.info(f"  Evals/Ckpt:     {len(DATASET_CONFIGS) * args.num_episodes * len(seeds)}")
    logger.info(f"  SAM3 ckpt:      {args.sam3_checkpoint}")
    gatc_mode = "gradient-magnitude" if args.gatc_use_grad else "grayscale"
    logger.info(f"  Metrics:        FDS + GATC({gatc_mode}, k={args.gatc_k}) + TCD(D_miss=0.5*diag)")
    logger.info(f"  Checkpoints:    {len(all_checkpoint_results)}")
    logger.info("=" * 90)

    for label, path, agg, eps in all_checkpoint_results:
        logger.info("")
        logger.info("-" * 90)
        logger.info(f"  CHECKPOINT: {label}")
        logger.info(f"  Path:       {path}")
        logger.info(f"  Episodes:   {agg['num_episodes']}")
        logger.info("-" * 90)

        # Frame Decay
        logger.info("")
        logger.info("  Frame Decay Score (FDS):")
        logger.info(f"    Mean L1  (lower=better):   {_fmt(agg['l1_mean'])} +/- {_fmt(agg['l1_std'])}")
        logger.info(f"    Mean SSIM (higher=better):  {_fmt(agg['ssim_mean'])} +/- {_fmt(agg['ssim_std'])}")
        logger.info(f"    L1 decay rate (slope):      {_fmt(agg['l1_slope_mean'], 6)}/frame")

        # Per-chunk L1 breakdown
        early = agg.get("l1_early_c1", float("nan"))
        mid = agg.get("l1_mid_c2c3", float("nan"))
        late = agg.get("l1_late_c4c6", float("nan"))
        logger.info(f"    Per-chunk L1:  early(c1)={_fmt(early)}  mid(c2-3)={_fmt(mid)}  late(c4-6)={_fmt(late)}")

        # GATC
        if "gatc_median" in agg:
            logger.info("")
            logger.info("  GT-Anchored Tool Consistency (GATC, higher=better):")
            logger.info(f"    Median:    {_fmt(agg['gatc_median'])} "
                         f"+/- {_fmt(agg['gatc_std'])}   95% CI: {_fmt_ci(agg['gatc_ci95'])}")
            logger.info(f"    Coverage:  {agg['gatc_coverage_mean']:.1f}% valid frames")
            logger.info(
                f"    Per-chunk median:  early(c1)={_fmt(agg.get('gatc_early_c1_median', float('nan')))}  "
                f"mid(c2-3)={_fmt(agg.get('gatc_mid_c2c3_median', float('nan')))}  "
                f"late(c4-6)={_fmt(agg.get('gatc_late_c4c6_median', float('nan')))}"
            )

        # TCD
        if "tcd_median" in agg:
            logger.info("")
            logger.info("  Tool Centroid Distance (TCD, lower=better):")
            logger.info(f"    Median:    {_fmt(agg['tcd_median'], 2)} px "
                         f"+/- {_fmt(agg['tcd_std'], 2)}   "
                         f"95% CI: {_fmt_ci(agg['tcd_ci95'], 2)}")
            logger.info(f"    Coverage:  {agg['tcd_coverage_mean']:.1f}% valid frames")
            logger.info(
                f"    Per-chunk median:  early(c1)={_fmt(agg.get('tcd_early_c1_median', float('nan')), 2)}  "
                f"mid(c2-3)={_fmt(agg.get('tcd_mid_c2c3_median', float('nan')), 2)}  "
                f"late(c4-6)={_fmt(agg.get('tcd_late_c4c6_median', float('nan')), 2)}"
            )

        # Per-dataset breakdown
        per_ds = agg.get("per_dataset", {})
        if per_ds:
            logger.info("")
            header = (f"    {'Dataset':<22s} | {'L1':>7s} | {'SSIM':>7s}"
                      f" | {'GATC':>7s} | {'TCD(px)':>8s}")
            sep = (f"    {'-'*22}-+-{'-'*7}-+-{'-'*7}"
                   f"-+-{'-'*7}-+-{'-'*8}")
            logger.info(header)
            logger.info(sep)
            for ds_name, d in sorted(per_ds.items()):
                g = _fmt(d.get("gatc_median", float("nan")))
                t = _fmt(d.get("tcd_median", float("nan")), 2)
                row = (f"    {ds_name:<22s} | {_fmt(d['l1_mean']):>7s} | {_fmt(d['ssim_mean']):>7s}"
                       f" | {g:>7s} | {t:>8s}")
                logger.info(row)

    # ---- Comparison table (when multiple checkpoints) ----
    if len(all_checkpoint_results) > 1:
        logger.info("")
        logger.info("=" * 90)
        logger.info("  CHECKPOINT COMPARISON TABLE")
        logger.info("=" * 90)

        hdr = (f"  {'Checkpoint':<30s} | {'L1 (↓)':>10s} | {'SSIM (↑)':>10s}"
               f" | {'GATC (↑)':>10s} | {'TCD px(↓)':>10s}")
        div = (f"  {'-'*30}-+-{'-'*10}-+-{'-'*10}"
               f"-+-{'-'*10}-+-{'-'*10}")
        logger.info(hdr)
        logger.info(div)

        rows = []
        for label, _, agg, _ in all_checkpoint_results:
            r = {
                "label": label,
                "l1": agg["l1_mean"],
                "ssim": agg["ssim_mean"],
                "gatc": agg.get("gatc_median", float("nan")),
                "tcd": agg.get("tcd_median", float("nan")),
            }
            rows.append(r)

            line = (f"  {label:<30s} | {_fmt(r['l1']):>10s} | {_fmt(r['ssim']):>10s}"
                    f" | {_fmt(r['gatc']):>10s} | {_fmt(r['tcd'], 2):>10s}")
            logger.info(line)

        logger.info("")
        logger.info("  Ranking (by Mean L1, ascending):")
        sorted_by_l1 = sorted(rows, key=lambda x: x["l1"])
        for rank, r in enumerate(sorted_by_l1, 1):
            logger.info(f"    #{rank}  {r['label']}  (L1={_fmt(r['l1'])})")

        valid_gatc = [r for r in rows if not math.isnan(r["gatc"])]
        if valid_gatc:
            logger.info("")
            logger.info("  Ranking (by GATC, descending):")
            sorted_by_gatc = sorted(valid_gatc, key=lambda x: -x["gatc"])
            for rank, r in enumerate(sorted_by_gatc, 1):
                logger.info(f"    #{rank}  {r['label']}  (GATC={_fmt(r['gatc'])})")

        valid_tcd = [r for r in rows if not math.isnan(r["tcd"])]
        if valid_tcd:
            logger.info("")
            logger.info("  Ranking (by TCD, ascending):")
            sorted_by_tcd = sorted(valid_tcd, key=lambda x: x["tcd"])
            for rank, r in enumerate(sorted_by_tcd, 1):
                logger.info(f"    #{rank}  {r['label']}  (TCD={_fmt(r['tcd'], 2)}px)")

        logger.info("")
        logger.info("  CHECKPOINT COMPARISON BY CHUNK")
        logger.info("  Early = chunk 1 (frames 1-12), Mid = chunks 2-3 (frames 13-36), Late = chunks 4-6 (frames 37-72)")
        logger.info("")
        hdr_chunk = (
            f"  {'Checkpoint':<28s} | "
            f"{'L1 E/M/L (↓)':<25s} | "
            f"{'GATC E/M/L (↑)':<25s} | "
            f"{'TCD E/M/L px (↓)':<25s}"
        )
        div_chunk = (
            f"  {'-'*28}-+-{'-'*25}-+-{'-'*25}-+-{'-'*25}"
        )
        logger.info(hdr_chunk)
        logger.info(div_chunk)
        for label, _, agg, _ in all_checkpoint_results:
            l1_triplet = (
                f"{_fmt(agg.get('l1_early_c1', float('nan')))} / "
                f"{_fmt(agg.get('l1_mid_c2c3', float('nan')))} / "
                f"{_fmt(agg.get('l1_late_c4c6', float('nan')))}"
            )
            gatc_triplet = (
                f"{_fmt(agg.get('gatc_early_c1_median', float('nan')))} / "
                f"{_fmt(agg.get('gatc_mid_c2c3_median', float('nan')))} / "
                f"{_fmt(agg.get('gatc_late_c4c6_median', float('nan')))}"
            )
            tcd_triplet = (
                f"{_fmt(agg.get('tcd_early_c1_median', float('nan')), 2)} / "
                f"{_fmt(agg.get('tcd_mid_c2c3_median', float('nan')), 2)} / "
                f"{_fmt(agg.get('tcd_late_c4c6_median', float('nan')), 2)}"
            )
            logger.info(
                f"  {label:<28s} | {l1_triplet:<25s} | {gatc_triplet:<25s} | {tcd_triplet:<25s}"
            )

    # ---- CSV section for spreadsheet copy-paste ----
    logger.info("")
    logger.info("=" * 90)
    logger.info("  CSV (copy-paste to spreadsheet)")
    logger.info("=" * 90)

    for label, path, agg, eps in all_checkpoint_results:
        logger.info("")
        logger.info(f"  Checkpoint: {label}")
        logger.info("")

        # Per-dataset CSV
        per_ds = agg.get("per_dataset", {})
        logger.info("Dataset;L1;SSIM;GATC;TCD(px)")
        for ds_name, d in sorted(per_ds.items()):
            l1 = d.get("l1_mean", float("nan"))
            ssim = d.get("ssim_mean", float("nan"))
            gatc = d.get("gatc_median", float("nan"))
            tcd = d.get("tcd_median", float("nan"))
            logger.info(f"{ds_name};{l1:.4f};{ssim:.4f};{gatc:.4f};{tcd:.2f}")

        # Summary scores CSV
        logger.info("")
        logger.info(
            "FDS (Mean L1);GATC (Median);TCD (Median, px);"
            "L1 early(c1);L1 mid(c2-3);L1 late(c4-6);"
            "GATC early(c1);GATC mid(c2-3);GATC late(c4-6);"
            "TCD early(c1);TCD mid(c2-3);TCD late(c4-6)"
        )
        fds = agg.get("l1_mean", float("nan"))
        gatc = agg.get("gatc_median", float("nan"))
        tcd = agg.get("tcd_median", float("nan"))
        early = agg.get("l1_early_c1", float("nan"))
        mid = agg.get("l1_mid_c2c3", float("nan"))
        late = agg.get("l1_late_c4c6", float("nan"))
        g_early = agg.get("gatc_early_c1_median", float("nan"))
        g_mid = agg.get("gatc_mid_c2c3_median", float("nan"))
        g_late = agg.get("gatc_late_c4c6_median", float("nan"))
        t_early = agg.get("tcd_early_c1_median", float("nan"))
        t_mid = agg.get("tcd_mid_c2c3_median", float("nan"))
        t_late = agg.get("tcd_late_c4c6_median", float("nan"))
        logger.info(
            f"{fds:.4f};{gatc:.4f};{tcd:.2f};{early:.4f};{mid:.4f};{late:.4f};"
            f"{g_early:.4f};{g_mid:.4f};{g_late:.4f};"
            f"{t_early:.2f};{t_mid:.2f};{t_late:.2f}"
        )

    # Multi-checkpoint summary CSV
    if len(all_checkpoint_results) > 1:
        logger.info("")
        logger.info(
            "Checkpoint;FDS (Mean L1);GATC (Median);TCD (Median, px);"
            "L1 early(c1);L1 mid(c2-3);L1 late(c4-6);"
            "GATC early(c1);GATC mid(c2-3);GATC late(c4-6);"
            "TCD early(c1);TCD mid(c2-3);TCD late(c4-6)"
        )
        for label, _, agg, _ in all_checkpoint_results:
            fds = agg.get("l1_mean", float("nan"))
            gatc = agg.get("gatc_median", float("nan"))
            tcd = agg.get("tcd_median", float("nan"))
            early = agg.get("l1_early_c1", float("nan"))
            mid = agg.get("l1_mid_c2c3", float("nan"))
            late = agg.get("l1_late_c4c6", float("nan"))
            g_early = agg.get("gatc_early_c1_median", float("nan"))
            g_mid = agg.get("gatc_mid_c2c3_median", float("nan"))
            g_late = agg.get("gatc_late_c4c6_median", float("nan"))
            t_early = agg.get("tcd_early_c1_median", float("nan"))
            t_mid = agg.get("tcd_mid_c2c3_median", float("nan"))
            t_late = agg.get("tcd_late_c4c6_median", float("nan"))
            logger.info(
                f"{label};{fds:.4f};{gatc:.4f};{tcd:.2f};{early:.4f};{mid:.4f};{late:.4f};"
                f"{g_early:.4f};{g_mid:.4f};{g_late:.4f};"
                f"{t_early:.2f};{t_mid:.2f};{t_late:.2f}"
            )

        logger.info("")
        logger.info("Checkpoint;L1 Early;L1 Mid;L1 Late;GATC Early;GATC Mid;GATC Late;TCD Early;TCD Mid;TCD Late")
        for label, _, agg, _ in all_checkpoint_results:
            logger.info(
                f"{label};"
                f"{agg.get('l1_early_c1', float('nan')):.4f};"
                f"{agg.get('l1_mid_c2c3', float('nan')):.4f};"
                f"{agg.get('l1_late_c4c6', float('nan')):.4f};"
                f"{agg.get('gatc_early_c1_median', float('nan')):.4f};"
                f"{agg.get('gatc_mid_c2c3_median', float('nan')):.4f};"
                f"{agg.get('gatc_late_c4c6_median', float('nan')):.4f};"
                f"{agg.get('tcd_early_c1_median', float('nan')):.2f};"
                f"{agg.get('tcd_mid_c2c3_median', float('nan')):.2f};"
                f"{agg.get('tcd_late_c4c6_median', float('nan')):.2f}"
            )

    logger.info("")
    logger.info("=" * 90)
    logger.info("  END OF REPORT")
    logger.info("=" * 90)


# ═══════════════════════════════════════════════════════════════════════════════
# JSON output
# ═══════════════════════════════════════════════════════════════════════════════
def save_results_json(
    all_checkpoint_results: List[Tuple[str, str, dict, List[dict]]],
    seeds: List[int],
    args: argparse.Namespace,
    save_path: str,
) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(save_path, f"quant_eval_results_{timestamp}.json")
    os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)

    def _sanitise(ep: dict) -> dict:
        """Strip large array fields for JSON serialisation."""
        out = {}
        for k, v in ep.items():
            if k.startswith("_"):
                continue
            if k == "frame_decay":
                out[k] = {
                    kk: vv for kk, vv in v.items()
                    if kk not in ("l1_per_frame", "ssim_per_frame")
                }
            else:
                out[k] = v
        return out

    payload = {
        "metadata": {
            "timestamp": timestamp,
            "seeds": seeds,
            "num_datasets": len(DATASET_CONFIGS),
            "num_episodes_per_dataset": args.num_episodes,
            "max_chunks": MAX_CHUNKS,
            "chunk_size": CHUNK_SIZE,
            "sam3_checkpoint": args.sam3_checkpoint,
            "datasets": [
                {"name": c["name"], "path": c["path"],
                 "episode_ids": c["episode_ids"][: args.num_episodes]}
                for c in DATASET_CONFIGS
            ],
        },
        "checkpoints": [],
    }

    for label, path, agg, eps in all_checkpoint_results:
        payload["checkpoints"].append({
            "label": label,
            "path": path,
            "aggregated": agg,
            "episodes": [_sanitise(e) for e in eps],
        })

    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    logger.info(f"Results saved to {json_path}")
    return json_path


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Cosmos surgical simulator — quantitative evaluation",
    )

    # Checkpoint(s)
    p.add_argument(
        "--ckpt_path", type=str, nargs="+", default=[],
        help="Path(s) to checkpoint .pt file(s).  Pass one or more.",
    )
    p.add_argument(
        "--ckpt_labels", type=str, nargs="+", default=[],
        help="Human-readable label(s) for each checkpoint (must match --ckpt_path length).",
    )
    p.add_argument(
        "--evaluate_all_checkpoints", action="store_true",
        help="Evaluate all checkpoints defined in CHECKPOINT_CONFIGS.",
    )

    # Model config
    p.add_argument(
        "--experiment", type=str,
        default="cosmos_predict2p5_2B_action_conditioned_cmr_13frame_4nodes_release_oss",
    )
    p.add_argument("--s3_cred", type=str, default="credentials/s3_checkpoint.secret")
    p.add_argument("--context_parallel_size", type=int, default=1)

    # Data
    p.add_argument(
        "--data_split", type=str, default="test", choices=["train", "test", "full"],
    )
    p.add_argument(
        "--num_episodes", type=int, default=5,
        help="Episodes per dataset (max 5). Total per seed = 4 × this.",
    )

    # Inference
    p.add_argument("--guidance", type=float, default=0)
    p.add_argument("--seed", type=int, default=0, help="Base seed")
    p.add_argument("--num_seeds", type=int, default=3, help="Number of seeds (derived from base)")

    # SAM3 (required for GATC + TCD)
    p.add_argument(
        "--sam3_checkpoint", type=str, required=True,
        help="Path to Medical-SAM3 checkpoint (required for GATC and TCD metrics).",
    )
    p.add_argument("--sam3_prompt", type=str, default="surgical tool")
    p.add_argument("--sam3_score_threshold", type=float, default=0.3)

    # GATC tuning
    p.add_argument(
        "--gatc_k", type=int, default=3,
        help="Translation search radius in px for GATC (default 3; k=5 is more thorough but slower).",
    )
    p.add_argument(
        "--gatc_use_grad", action="store_true",
        help="Use gradient magnitude for GATC ZNCC (default: grayscale, which is more sensitive).",
    )

    # Output
    p.add_argument("--save_path", type=str, default="output/quant_eval_cmr")
    p.add_argument("--save_fps", type=int, default=10)
    p.add_argument("--save_videos", action="store_true", help="Save generated / comparison videos")

    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    torch.set_grad_enabled(False)
    args = parse_args()

    seeds = [args.seed + i for i in range(args.num_seeds)]
    os.makedirs(args.save_path, exist_ok=True)

    # Resolve checkpoints
    checkpoint_list: List[Tuple[str, str, str]] = []  # (label, experiment, path)

    if args.evaluate_all_checkpoints:
        if not CHECKPOINT_CONFIGS:
            logger.error("CHECKPOINT_CONFIGS is empty. Add entries or use --ckpt_path.")
            return
        for exp_name, exp_cfg in CHECKPOINT_CONFIGS.items():
            for ckpt in exp_cfg["checkpoints"]:
                lbl = f"{exp_cfg['label']}-{ckpt['iter'] // 1000}k"
                checkpoint_list.append((lbl, exp_cfg["experiment"], ckpt["path"]))
    elif args.ckpt_path:
        labels = args.ckpt_labels if args.ckpt_labels else []
        for i, cp in enumerate(args.ckpt_path):
            lbl = labels[i] if i < len(labels) else os.path.basename(cp).replace(".pt", "")
            checkpoint_list.append((lbl, args.experiment, cp))
    else:
        logger.error("Provide --ckpt_path or --evaluate_all_checkpoints.")
        return

    # Validate SAM3 availability early — fail fast before any expensive work
    if not SAM3_AVAILABLE:
        raise RuntimeError(
            "sam3_inference could not be imported. "
            "Install Medical-SAM3 and ensure sam3_inference is on PYTHONPATH. "
            "GATC and TCD metrics require SAM3."
        )
    if not os.path.isfile(args.sam3_checkpoint):
        raise FileNotFoundError(
            f"SAM3 checkpoint not found: {args.sam3_checkpoint}"
        )

    logger.info("Validating SAM3 model can be loaded and has required API ...")
    _sam3_test = SAM3Model(checkpoint_path=args.sam3_checkpoint, confidence_threshold=0.1)
    _sam3_test.load_model()
    if _sam3_test.processor is None:
        raise RuntimeError(
            "SAM3 processor is None after load_model(). "
            "Check SAM3 checkpoint integrity."
        )
    if not hasattr(_sam3_test.processor, "set_text_prompt"):
        raise RuntimeError(
            "SAM3 processor lacks set_text_prompt(). "
            "Ensure you are using a compatible Medical-SAM3 version."
        )
    logger.info("SAM3 validation passed — model loads, processor.set_text_prompt available.")
    del _sam3_test
    torch.cuda.empty_cache()

    logger.info(f"Checkpoints to evaluate: {len(checkpoint_list)}")
    for lbl, exp, pth in checkpoint_list:
        logger.info(f"  {lbl}: {pth}")
    logger.info(f"Seeds: {seeds}")
    logger.info(f"SAM3 checkpoint: {args.sam3_checkpoint}")

    # Evaluate each checkpoint
    all_results: List[Tuple[str, str, dict, List[dict]]] = []
    total_start = time.time()

    for lbl, exp, pth in checkpoint_list:
        eval_args = deepcopy(args)
        eval_args.experiment = exp

        episodes = evaluate_checkpoint(eval_args, pth, lbl, seeds)
        if not episodes:
            logger.warning(f"No results for checkpoint {lbl}")
            continue

        agg = aggregate_checkpoint_results(episodes)
        all_results.append((lbl, pth, agg, episodes))

    total_dur = time.time() - total_start

    # Report
    if all_results:
        print_quantitative_report(all_results, seeds, args)
        json_path = save_results_json(all_results, seeds, args, args.save_path)
        logger.info(f"\nTotal evaluation time: {total_dur / 60:.1f} min")
        logger.info(f"JSON results: {json_path}")
    else:
        logger.warning("No results were generated.")


if __name__ == "__main__":
    main()
