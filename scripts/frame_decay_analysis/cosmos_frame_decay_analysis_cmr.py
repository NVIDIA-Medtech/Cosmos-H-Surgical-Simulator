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
Frame decay analysis script for CMR Versius surgical robot world model evaluation.

This script:
1. Loads episodes from 4 predefined CMR Versius datasets (prostatectomy, inguinal hernia,
   hysterectomy, cholecystectomy) with fixed episode IDs for each
2. Generates videos using Cosmos model with given checkpoint(s) - LIMITED TO 6 ACTION CHUNKS (72 frames)
3. Computes per-frame L1 and SSIM metrics to analyze quality decay
4. Saves results to JSON for later plotting
5. Supports multiple checkpoints and seeds for robust evaluation

DATASETS EVALUATED (fixed):
- /CMR_Versius/prostatectomy_480p: episodes 10155, 10156, 10157
- /CMR_Versius/inguinal_hernia_480p: episodes 7186, 7187, 7188
- /CMR_Versius/hysterectomy_480p: episodes 7369, 7370, 7371
- /CMR_Versius/cholecystectomy_480p: episodes 4646, 4647, 4648

METRICS COMPUTED:
- Per-frame L1 distance: Pixel-level absolute difference for each generated frame
  Computed on frames normalized to [-1, 1] range for consistent out-of-domain comparisons
- Per-frame SSIM: Structural similarity index for each generated frame
  Computed on frames normalized to [-1, 1] range with data_range=2.0

OUTPUT:
- JSON file containing per-frame L1 and SSIM metrics averaged across episodes and seeds

Example usage:

Single checkpoint evaluation (evaluates all 4 datasets, 3 episodes each by default):
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/frame_decay_analysis/cosmos_frame_decay_analysis_cmr.py \
  --experiment cosmos_predict2p5_2B_action_conditioned_cmr_13frame_4nodes_release_oss \
  --ckpt_path /path/to/checkpoint/model_ema_bf16.pt \
  --save_path output/frame_decay_cmr \
  --seed 0

Single checkpoint with fewer episodes per dataset:
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/frame_decay_analysis/cosmos_frame_decay_analysis_cmr.py \
  --experiment cosmos_predict2p5_2B_action_conditioned_cmr_13frame_4nodes_release_oss \
  --ckpt_path /path/to/checkpoint/model_ema_bf16.pt \
  --save_path output/frame_decay_cmr \
  --num_episodes 1 \
  --seed 0

Multiple checkpoints evaluation:
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/frame_decay_analysis/cosmos_frame_decay_analysis_cmr.py \
  --experiment cosmos_predict2p5_2B_action_conditioned_cmr_13frame_4nodes_release_oss \
  --save_path output/frame_decay_cmr \
  --seed 0 \
  --evaluate_all_checkpoints
"""

import argparse
import json
import os
from copy import deepcopy
from datetime import datetime

import mediapy
import numpy as np
import torch
from loguru import logger

# Set TOKENIZERS_PARALLELISM environment variable to avoid deadlocks with multiprocessing
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Import for SSIM metric
try:
    from skimage.metrics import structural_similarity as skimage_ssim

    def compute_ssim(im1, im2, data_range=1.0):
        """Compute SSIM using scikit-image"""
        # Ensure images are in HWC format for skimage
        if im1.shape[0] == 3:  # CHW format
            im1 = np.transpose(im1, (1, 2, 0))
            im2 = np.transpose(im2, (1, 2, 0))
        return skimage_ssim(im1, im2, data_range=data_range, channel_axis=2)

except ImportError:
    try:
        from pytorch_msssim import ssim as pt_ssim

        def compute_ssim(im1, im2, data_range=1.0):
            """Compute SSIM using pytorch"""
            # Convert numpy to torch if needed
            if isinstance(im1, np.ndarray):
                im1 = torch.from_numpy(im1).float()
                im2 = torch.from_numpy(im2).float()
            # Ensure CHW format and add batch dimension
            if im1.dim() == 3:
                if im1.shape[-1] == 3:  # HWC
                    im1 = im1.permute(2, 0, 1)
                    im2 = im2.permute(2, 0, 1)
                im1 = im1.unsqueeze(0)
                im2 = im2.unsqueeze(0)
            elif im1.dim() == 2:  # Grayscale
                im1 = im1.unsqueeze(0).unsqueeze(0)
                im2 = im2.unsqueeze(0).unsqueeze(0)
            return pt_ssim(im1, im2, data_range=data_range).item()

    except ImportError:
        logger.warning("Neither scikit-image nor pytorch_msssim installed for SSIM computation")
        compute_ssim = None

from cosmos_predict2._src.predict2.action.inference.inference_pipeline import (
    ActionVideo2WorldInference,
)
from cosmos_predict2._src.predict2.action.datasets.gr00t_dreams.data.dataset import (
    LeRobotDataset,
)

# Constants matching training config in groot_configs.py for cmr_versius
# CMR Versius: 60Hz data → 10fps (TIMESTEP_INTERVAL=6), 13 video frames, 12 action frames
NUM_FRAMES = 13  # Video frames (1 context + 12 prediction)
TIMESTEP_INTERVAL = 6  # Frame stride: 60Hz / 10fps = 6
CHUNK_SIZE = 12  # Action frames per window (NUM_FRAMES - 1)
MAX_CHUNKS = 6  # Limit to 6 chunks (72 frames) for frame decay analysis


# Fixed dataset configurations with episode IDs for evaluation
# Each dataset has 3 episodes that can be evaluated
DATASET_CONFIGS = [
    {
        "path": "/CMR_Versius/prostatectomy_480p",
        "name": "prostatectomy",
        "episode_ids": [10155, 10156, 10157],
    },
    {
        "path": "/CMR_Versius/inguinal_hernia_480p",
        "name": "inguinal_hernia",
        "episode_ids": [7186, 7187, 7188],
    },
    {
        "path": "/CMR_Versius/hysterectomy_480p",
        "name": "hysterectomy",
        "episode_ids": [7369, 7370, 7371],
    },
    {
        "path": "/CMR_Versius/cholecystectomy_480p",
        "name": "cholecystectomy",
        "episode_ids": [4646, 4647, 4648],
    },
]


# Define checkpoint configurations to evaluate
# Add your checkpoints here following this format
CHECKPOINT_CONFIGS = {
    # "cmr-exp-1": {
    #     "label": "cmr-exp-1",
    #     "experiment": "cosmos_predict2p5_2B_action_conditioned_cmr_13frame_4nodes_release_oss",
    #     "checkpoints": [
    #         {
    #             "iter": 10000,
    #             "path": "/path/to/checkpoint_10000.pt"
    #         },
    #         {
    #             "iter": 20000,
    #             "path": "/path/to/checkpoint_20000.pt"
    #         },
    #     ]
    # },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Frame decay analysis for CMR Versius world model")
    
    # Model arguments
    parser.add_argument(
        "--experiment",
        type=str,
        default="cosmos_predict2p5_2B_action_conditioned_cmr_13frame_4nodes_release_oss",
        help="Experiment config name",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="",
        help="Path to the checkpoint (.pt file) for single checkpoint evaluation",
    )
    parser.add_argument("--s3_cred", type=str, default="credentials/s3_checkpoint.secret")
    
    # Data arguments
    parser.add_argument(
        "--data_split",
        type=str,
        default="test",
        choices=["train", "test", "full"],
        help="Data split to use for evaluation",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=3,
        choices=[1, 2, 3],
        help="Number of episodes to evaluate per dataset (1-3, default: 3). "
             "Total episodes = num_episodes × 4 datasets.",
    )
    
    # Inference arguments
    parser.add_argument("--guidance", type=float, default=0, help="Classifier-free guidance scale")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument("--num_seeds", type=int, default=3, help="Number of seeds to derive for multi-seed evaluation")
    
    # Output arguments
    parser.add_argument(
        "--save_path",
        type=str,
        default="output/frame_decay_cmr",
        help="Output directory for results",
    )
    parser.add_argument("--save_fps", type=int, default=10, help="FPS for saved videos")
    parser.add_argument(
        "--save_videos",
        action="store_true",
        help="Save generated and comparison videos",
    )
    
    # Evaluation mode arguments
    parser.add_argument(
        "--evaluate_all_checkpoints",
        action="store_true",
        help="Evaluate all predefined checkpoints instead of a single one",
    )
    parser.add_argument(
        "--resume_path",
        type=str,
        default=None,
        help="Path to a previous run directory to resume from",
    )
    
    # Context parallel arguments
    parser.add_argument(
        "--context_parallel_size",
        type=int,
        default=1,
        help="Context parallel size (number of GPUs)",
    )
    
    return parser.parse_args()


def build_episode_index_map(dataset: LeRobotDataset) -> dict[int, list[int]]:
    """
    Build a mapping from episode_id to list of dataset indices for that episode.

    Args:
        dataset: The LeRobotDataset instance

    Returns:
        Dict mapping episode_id -> list of (dataset_idx, base_index) sorted by base_index
    """
    inner_dataset = dataset.lerobot_datasets[0]
    all_steps = inner_dataset._all_steps  # List of (episode_id, base_index) tuples

    episode_map: dict[int, list[tuple[int, int]]] = {}
    for dataset_idx, (episode_id, base_index) in enumerate(all_steps):
        if episode_id not in episode_map:
            episode_map[episode_id] = []
        episode_map[episode_id].append((dataset_idx, base_index))

    # Sort each episode's entries by base_index
    for episode_id in episode_map:
        episode_map[episode_id].sort(key=lambda x: x[1])

    return episode_map


def get_episode_ids_in_split(dataset: LeRobotDataset) -> list[int]:
    """Get the unique episode IDs present in the dataset (after split is applied)."""
    inner_dataset = dataset.lerobot_datasets[0]
    all_steps = inner_dataset._all_steps
    episode_ids = sorted(set(ep_id for ep_id, _ in all_steps))
    return episode_ids


def find_chunk_indices(
    episode_map: dict[int, list[tuple[int, int]]],
    episode_id: int,
    chunk_size: int = CHUNK_SIZE,
    timestep_interval: int = TIMESTEP_INTERVAL,
    max_chunks: int = MAX_CHUNKS,
) -> list[int] | None:
    """
    Find dataset indices for non-overlapping chunks of an episode.

    For autoregressive inference, we need windows starting at base_index 0, 72, 144, ...
    (increments of chunk_size * timestep_interval = 12 * 6 = 72 for CMR Versius)

    Args:
        episode_map: Mapping from episode_id to list of (dataset_idx, base_index)
        episode_id: The episode to get chunks for
        chunk_size: Number of actions per chunk (default 12)
        timestep_interval: Temporal downsampling factor (default 6 for CMR Versius)
        max_chunks: Maximum number of chunks to return (default 6 for 72 frames)

    Returns:
        List of dataset indices for non-overlapping chunks, or None if episode
        doesn't have base_index=0 (i.e., episode is only partially in the split)
    """
    if episode_id not in episode_map:
        logger.warning(f"Episode {episode_id} not found in episode_map")
        return None

    entries = episode_map[episode_id]
    base_index_to_dataset_idx = {base_idx: ds_idx for ds_idx, base_idx in entries}

    # Must have base_index=0 to start autoregressive inference from beginning
    if 0 not in base_index_to_dataset_idx:
        return None

    # We need base_indices: 0, 72, 144, 216, ... (chunk_size * timestep_interval increments)
    stride = chunk_size * timestep_interval
    chunk_indices = []

    base_index = 0
    while base_index in base_index_to_dataset_idx and len(chunk_indices) < max_chunks:
        chunk_indices.append(base_index_to_dataset_idx[base_index])
        base_index += stride

    return chunk_indices


class FrameMetricsComputer:
    """Class for computing per-frame L1 and SSIM metrics."""
    
    def __init__(self, device='cuda'):
        self.device = device
    
    def compute_per_frame_l1(self, pred: np.ndarray, target: np.ndarray) -> list[float]:
        """Compute L1 distance for each frame.
        
        Args:
            pred: (T, H, W, C) array of predicted video (uint8, 0-255)
            target: (T, H, W, C) array of ground truth video (uint8, 0-255)
            
        Returns:
            List of L1 distances for each frame (excluding the first conditioning frame)
        """
        T = pred.shape[0]
        l1_distances = []
        
        # Normalize to [-1, 1] range for consistent comparisons
        pred_norm = pred.astype(np.float32) / 127.5 - 1.0
        target_norm = target.astype(np.float32) / 127.5 - 1.0
        
        # Skip the first frame (t=0) as it's the original ground truth conditioning frame
        for t in range(1, T):
            l1_dist = np.mean(np.abs(pred_norm[t] - target_norm[t]))
            l1_distances.append(float(l1_dist))
        
        return l1_distances
    
    def compute_per_frame_ssim(self, pred: np.ndarray, target: np.ndarray) -> list[float] | None:
        """Compute SSIM for each frame.
        
        Args:
            pred: (T, H, W, C) array of predicted video (uint8, 0-255)
            target: (T, H, W, C) array of ground truth video (uint8, 0-255)
            
        Returns:
            List of SSIM scores for each frame (excluding the first conditioning frame)
        """
        if compute_ssim is None:
            return None
        
        T = pred.shape[0]
        ssim_scores = []
        
        # Normalize to [-1, 1] range for consistent comparisons
        pred_norm = pred.astype(np.float32) / 127.5 - 1.0
        target_norm = target.astype(np.float32) / 127.5 - 1.0
        
        # Skip the first frame (t=0) as it's the original ground truth conditioning frame
        for t in range(1, T):
            # SSIM computation with data_range=2.0 for [-1, 1] range
            score = compute_ssim(pred_norm[t], target_norm[t], data_range=2.0)
            ssim_scores.append(float(score))
        
        return ssim_scores
    
    def compute_frame_metrics(self, pred: np.ndarray, target: np.ndarray) -> dict:
        """Compute per-frame L1 and SSIM metrics.
        
        Args:
            pred: (T, H, W, C) array of predicted video
            target: (T, H, W, C) array of target video
        
        Returns:
            Dictionary with 'l1_per_frame' and 'ssim_per_frame' lists
            Note: The first frame (conditioning) is excluded from analysis,
            so returned lists start from the first generated frame (frame 2 of video).
        """
        l1_per_frame = self.compute_per_frame_l1(pred, target)
        ssim_per_frame = self.compute_per_frame_ssim(pred, target)
        
        return {
            'l1_per_frame': l1_per_frame,
            'ssim_per_frame': ssim_per_frame
        }


def load_dataset(dataset_path: str, data_split: str) -> LeRobotDataset:
    """Load the LeRobotDataset for CMR inference.
    
    Args:
        dataset_path: Path to the LeRobot-format dataset
        data_split: Data split to use ('train', 'test', or 'full')
    
    Returns:
        LeRobotDataset instance
    """
    logger.info(f"Loading LeRobotDataset from {dataset_path} with split '{data_split}'")
    
    dataset = LeRobotDataset(
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
    
    return dataset


def setup_inference_pipeline(args: argparse.Namespace, ckpt_path: str) -> ActionVideo2WorldInference:
    """Initialize the inference pipeline."""
    logger.info(f"Loading model from {ckpt_path}")
    
    video2world = ActionVideo2WorldInference(
        args.experiment,
        ckpt_path,
        args.s3_cred,
        context_parallel_size=args.context_parallel_size,
    )
    
    mem_bytes = torch.cuda.memory_allocated()
    logger.info(f"GPU memory after model load: {mem_bytes / (1024**3):.2f} GB")
    
    return video2world


def process_episode(
    video2world: ActionVideo2WorldInference,
    dataset: LeRobotDataset,
    episode_map: dict,
    episode_id: int,
    args: argparse.Namespace,
    metrics_computer: FrameMetricsComputer,
    output_dir: str,
    seed: int,
) -> dict | None:
    """
    Process a single episode and compute frame metrics.
    
    Returns:
        Dictionary with metrics, or None if episode couldn't be processed
    """
    # Find dataset indices for non-overlapping chunks of this episode
    chunk_indices = find_chunk_indices(episode_map, episode_id)
    
    if chunk_indices is None:
        logger.warning(f"Episode {episode_id} doesn't start at base_index=0 in this split, skipping")
        return None
    
    if len(chunk_indices) < MAX_CHUNKS:
        logger.warning(f"Episode {episode_id} has only {len(chunk_indices)} chunks (need {MAX_CHUNKS}), skipping")
        return None
    
    logger.info(f"Episode {episode_id}: processing {len(chunk_indices)} chunks")
    
    predicted_chunks = []
    gt_chunks = []
    current_frame = None
    
    for chunk_idx, dataset_idx in enumerate(chunk_indices):
        # Get data from dataset - actions are already transformed (relative + normalized)
        data = dataset[dataset_idx]
        
        # video shape: (C, T, H, W) -> need (T, H, W, C) for inference
        video = data["video"].permute(1, 2, 3, 0).numpy()  # (T, H, W, C)
        actions = data["action"].numpy()  # (chunk_size, action_dim) - already normalized
        
        if chunk_idx == 0:
            logger.info(f"  Data shapes - video: {data['video'].shape} (C,T,H,W), "
                       f"action: {actions.shape} (T,D)")
            # First chunk: use ground truth first frame as conditioning
            current_frame = video[0]  # (H, W, C)
        
        # Store ground truth for comparison
        gt_chunks.append(video)
        
        # Run inference
        # Use a deterministic seed for each chunk based on episode seed
        chunk_seed = seed + chunk_idx
        
        next_frame, video_chunk = video2world.step_inference(
            img_array=current_frame,
            action=actions.astype(np.float32),
            guidance=args.guidance,
            seed=chunk_seed,
            num_latent_conditional_frames=1,
        )
        
        predicted_chunks.append(video_chunk)
        current_frame = next_frame
        
        logger.info(f"  Chunk {chunk_idx + 1}/{len(chunk_indices)} complete")
    
    if not predicted_chunks:
        logger.warning(f"No chunks generated for episode {episode_id}")
        return None
    
    # Stitch chunks together
    # First chunk: all frames, subsequent chunks: skip first frame (it's the conditioning)
    stitched_predicted = [predicted_chunks[0]]
    for chunk in predicted_chunks[1:]:
        stitched_predicted.append(chunk[1:])
    predicted_video = np.concatenate(stitched_predicted, axis=0)
    
    # Stitch ground truth the same way
    stitched_gt = [gt_chunks[0]]
    for chunk in gt_chunks[1:]:
        stitched_gt.append(chunk[1:])
    gt_video = np.concatenate(stitched_gt, axis=0)
    
    # Trim to same length
    min_len = min(len(gt_video), len(predicted_video))
    gt_video = gt_video[:min_len]
    predicted_video = predicted_video[:min_len]
    
    logger.info(f"Episode {episode_id}: Generated video has {len(predicted_video)} frames")
    
    # Compute metrics
    metrics = metrics_computer.compute_frame_metrics(predicted_video, gt_video)
    
    # Log summary
    if metrics.get('l1_per_frame'):
        logger.info(f"Episode {episode_id} L1 - Mean: {np.mean(metrics['l1_per_frame']):.6f}, "
                   f"Std: {np.std(metrics['l1_per_frame']):.6f}")
    if metrics.get('ssim_per_frame'):
        logger.info(f"Episode {episode_id} SSIM - Mean: {np.mean(metrics['ssim_per_frame']):.6f}, "
                   f"Std: {np.std(metrics['ssim_per_frame']):.6f}")
    
    # Optionally save videos
    if args.save_videos:
        os.makedirs(os.path.join(output_dir, "predicted"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "comparison"), exist_ok=True)
        
        # Save predicted video
        pred_path = os.path.join(output_dir, "predicted", f"episode_{episode_id:04d}_seed_{seed}.mp4")
        mediapy.write_video(pred_path, predicted_video, fps=args.save_fps)
        logger.info(f"Saved predicted video to {pred_path}")
        
        # Save side-by-side comparison
        comparison = np.concatenate([gt_video, predicted_video], axis=2)
        comp_path = os.path.join(output_dir, "comparison", f"episode_{episode_id:04d}_seed_{seed}.mp4")
        mediapy.write_video(comp_path, comparison, fps=args.save_fps)
        logger.info(f"Saved comparison video to {comp_path}")
    
    return metrics


def evaluate_checkpoint(
    args: argparse.Namespace,
    ckpt_path: str,
    checkpoint_label: str,
    seed: int,
    output_dir: str,
) -> dict | None:
    """
    Evaluate a single checkpoint with a single seed across all predefined datasets.
    
    Iterates over all 4 CMR Versius datasets (prostatectomy, inguinal hernia,
    hysterectomy, cholecystectomy) and evaluates the specified number of episodes
    from each dataset.
    
    Returns:
        Dictionary with aggregated metrics, or None if evaluation failed
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Evaluating checkpoint: {checkpoint_label}")
    logger.info(f"Seed: {seed}")
    logger.info(f"Path: {ckpt_path}")
    logger.info(f"Datasets: {len(DATASET_CONFIGS)} (evaluating {args.num_episodes} episodes each)")
    logger.info(f"Total episodes: {len(DATASET_CONFIGS) * args.num_episodes}")
    logger.info(f"{'='*80}\n")
    
    # Set random seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Load inference pipeline (only once for all datasets)
    video2world = setup_inference_pipeline(args, ckpt_path)
    
    # Initialize metrics computer
    metrics_computer = FrameMetricsComputer(device='cuda')
    
    # Storage for all per-frame metrics across all datasets and episodes
    all_episode_metrics = []
    dataset_episode_counts = {}
    
    # Iterate over all predefined datasets
    for dataset_config in DATASET_CONFIGS:
        dataset_path = dataset_config["path"]
        dataset_name = dataset_config["name"]
        episode_ids = dataset_config["episode_ids"][:args.num_episodes]
        
        logger.info(f"\n{'-'*60}")
        logger.info(f"Processing dataset: {dataset_name}")
        logger.info(f"Path: {dataset_path}")
        logger.info(f"Episodes to evaluate: {episode_ids}")
        logger.info(f"{'-'*60}")
        
        try:
            # Load dataset
            dataset = load_dataset(dataset_path, args.data_split)
            episode_map = build_episode_index_map(dataset)
            logger.info(f"Built index map for {len(episode_map)} episodes in '{args.data_split}' split")
            
            episodes_processed = 0
            
            # Process episodes for this dataset
            for episode_idx, episode_id in enumerate(episode_ids):
                logger.info(f"Processing {dataset_name} episode {episode_id} ({episode_idx + 1}/{len(episode_ids)})")
                
                try:
                    # Create dataset-specific output directory
                    dataset_output_dir = os.path.join(output_dir, dataset_name)
                    
                    metrics = process_episode(
                        video2world=video2world,
                        dataset=dataset,
                        episode_map=episode_map,
                        episode_id=episode_id,
                        args=args,
                        metrics_computer=metrics_computer,
                        output_dir=dataset_output_dir,
                        seed=seed,
                    )
                    
                    if metrics:
                        # Add dataset info to metrics
                        metrics['dataset_name'] = dataset_name
                        metrics['dataset_path'] = dataset_path
                        all_episode_metrics.append(metrics)
                        episodes_processed += 1
                
                except Exception as e:
                    logger.error(f"Error processing {dataset_name} episode {episode_id}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            dataset_episode_counts[dataset_name] = episodes_processed
            logger.info(f"Completed {dataset_name}: {episodes_processed}/{len(episode_ids)} episodes processed")
        
        except Exception as e:
            logger.error(f"Error loading dataset {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            dataset_episode_counts[dataset_name] = 0
            continue
    
    # Cleanup
    video2world.cleanup()
    
    # Aggregate metrics across all episodes from all datasets
    if not all_episode_metrics:
        logger.warning(f"No metrics collected for checkpoint {checkpoint_label}")
        return None
    
    # Find the maximum number of frames
    max_frames = max(len(m['l1_per_frame']) for m in all_episode_metrics if m.get('l1_per_frame'))
    
    # Aggregate L1 and SSIM per frame across all episodes
    aggregated_l1 = []
    aggregated_ssim = []
    
    for frame_idx in range(max_frames):
        frame_l1_values = []
        frame_ssim_values = []
        
        for episode_metrics in all_episode_metrics:
            if episode_metrics.get('l1_per_frame') and frame_idx < len(episode_metrics['l1_per_frame']):
                frame_l1_values.append(episode_metrics['l1_per_frame'][frame_idx])
            if episode_metrics.get('ssim_per_frame') and frame_idx < len(episode_metrics['ssim_per_frame']):
                frame_ssim_values.append(episode_metrics['ssim_per_frame'][frame_idx])
        
        if frame_l1_values:
            aggregated_l1.append({
                'mean': float(np.mean(frame_l1_values)),
                'std': float(np.std(frame_l1_values)),
                'values': frame_l1_values
            })
        
        if frame_ssim_values:
            aggregated_ssim.append({
                'mean': float(np.mean(frame_ssim_values)),
                'std': float(np.std(frame_ssim_values)),
                'values': frame_ssim_values
            })
    
    # Log summary
    logger.info(f"\n{'='*60}")
    logger.info(f"Checkpoint {checkpoint_label} evaluation complete")
    logger.info(f"Total episodes processed: {len(all_episode_metrics)}")
    for dataset_name, count in dataset_episode_counts.items():
        logger.info(f"  - {dataset_name}: {count} episodes")
    logger.info(f"{'='*60}")
    
    return {
        'checkpoint_label': checkpoint_label,
        'checkpoint_path': ckpt_path,
        'seed': seed,
        'aggregated_l1_per_frame': aggregated_l1,
        'aggregated_ssim_per_frame': aggregated_ssim,
        'num_episodes': len(all_episode_metrics),
        'max_frames': max_frames,
        'dataset_episode_counts': dataset_episode_counts,
    }


def derive_eval_seeds(base_seed: int, num_seeds: int = 3) -> list[int]:
    """Derive deterministic evaluation seeds from a base seed."""
    return [base_seed + offset for offset in range(num_seeds)]


def aggregate_results_across_seeds(per_seed_results: dict, seeds: list[int]) -> list[dict]:
    """Aggregate per-frame metrics across seeds for each checkpoint."""
    aggregated_results = []
    
    if not per_seed_results:
        return aggregated_results
    
    # Get all checkpoint labels
    checkpoint_labels = set()
    for seed in seeds:
        for result in per_seed_results.get(seed, []) or []:
            checkpoint_labels.add(result['checkpoint_label'])
    
    for checkpoint_label in sorted(checkpoint_labels):
        checkpoint_path = None
        max_frames = 0
        num_episodes_per_seed = {}
        per_seed_l1_means = {}
        per_seed_ssim_means = {}
        seed_results_present = []
        
        for seed in seeds:
            seed_result_list = per_seed_results.get(seed, []) or []
            checkpoint_result = next(
                (res for res in seed_result_list if res['checkpoint_label'] == checkpoint_label),
                None
            )
            
            if checkpoint_result is None:
                num_episodes_per_seed[seed] = 0
                per_seed_l1_means[seed] = []
                per_seed_ssim_means[seed] = []
                continue
            
            seed_results_present.append((seed, checkpoint_result))
            checkpoint_path = checkpoint_result.get('checkpoint_path', checkpoint_path)
            max_frames = max(max_frames, checkpoint_result.get('max_frames', 0))
            num_episodes_per_seed[seed] = int(checkpoint_result.get('num_episodes', 0))
            
            per_seed_l1_means[seed] = [
                float(frame.get('mean', 0.0))
                for frame in checkpoint_result.get('aggregated_l1_per_frame', [])
            ]
            per_seed_ssim_means[seed] = [
                float(frame.get('mean', 0.0))
                for frame in checkpoint_result.get('aggregated_ssim_per_frame', [])
            ]
        
        if not seed_results_present:
            logger.warning(f"No seed produced results for checkpoint {checkpoint_label}; skipping aggregation")
            continue
        
        # Aggregate across seeds
        aggregated_l1 = []
        aggregated_ssim = []
        
        for frame_idx in range(max_frames):
            frame_l1_means = []
            frame_ssim_means = []
            
            for seed, seed_result in seed_results_present:
                l1_data = seed_result.get('aggregated_l1_per_frame', [])
                if frame_idx < len(l1_data):
                    frame_l1_means.append(float(l1_data[frame_idx]['mean']))
                
                ssim_data = seed_result.get('aggregated_ssim_per_frame', [])
                if frame_idx < len(ssim_data):
                    frame_ssim_means.append(float(ssim_data[frame_idx]['mean']))
            
            if frame_l1_means:
                aggregated_l1.append({
                    'mean': float(np.mean(frame_l1_means)),
                    'std': float(np.std(frame_l1_means)),
                    'values': frame_l1_means,
                })
            
            if frame_ssim_means:
                aggregated_ssim.append({
                    'mean': float(np.mean(frame_ssim_means)),
                    'std': float(np.std(frame_ssim_means)),
                    'values': frame_ssim_means,
                })
        
        aggregated_results.append({
            'checkpoint_label': checkpoint_label,
            'checkpoint_path': checkpoint_path,
            'aggregated_l1_per_frame': aggregated_l1,
            'aggregated_ssim_per_frame': aggregated_ssim,
            'num_episodes': int(sum(num_episodes_per_seed.values())),
            'num_episodes_per_seed': num_episodes_per_seed,
            'max_frames': max_frames,
            'per_seed_l1_means': per_seed_l1_means,
            'per_seed_ssim_means': per_seed_ssim_means,
        })
    
    return aggregated_results


def save_results_to_json(results: list[dict], seeds: list[int], save_path: str, num_episodes_per_dataset: int, filename_prefix: str = "frame_decay_results") -> str:
    """Save results to a JSON file for later plotting."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(save_path, f"{filename_prefix}_{timestamp}.json")
    
    os.makedirs(os.path.dirname(json_path) or '.', exist_ok=True)
    
    # Prepare serializable data
    serializable_data = {
        'metadata': {
            'timestamp': timestamp,
            'seeds': seeds,
            'max_chunks': MAX_CHUNKS,
            'num_frames_per_chunk': CHUNK_SIZE,
            'total_frames_analyzed': MAX_CHUNKS * CHUNK_SIZE,  # 72 frames
            'num_datasets': len(DATASET_CONFIGS),
            'num_episodes_per_dataset': num_episodes_per_dataset,
            'total_episodes_per_seed': len(DATASET_CONFIGS) * num_episodes_per_dataset,
            'datasets': [
                {
                    'name': cfg['name'],
                    'path': cfg['path'],
                    'episode_ids': cfg['episode_ids'][:num_episodes_per_dataset],
                }
                for cfg in DATASET_CONFIGS
            ],
        },
        'results': []
    }
    
    for result in results:
        entry = {
            'checkpoint_label': result['checkpoint_label'],
            'checkpoint_path': result.get('checkpoint_path'),
            'num_episodes': int(result.get('num_episodes', 0)),
            'max_frames': int(result.get('max_frames', 0)),
            'l1_per_frame': {
                'means': [float(d['mean']) for d in result.get('aggregated_l1_per_frame', [])],
                'stds': [float(d['std']) for d in result.get('aggregated_l1_per_frame', [])],
            },
            'ssim_per_frame': {
                'means': [float(d['mean']) for d in result.get('aggregated_ssim_per_frame', [])],
                'stds': [float(d['std']) for d in result.get('aggregated_ssim_per_frame', [])],
            },
        }
        
        # Add per-seed data if available
        if 'num_episodes_per_seed' in result:
            entry['num_episodes_per_seed'] = {
                str(seed): int(result['num_episodes_per_seed'].get(seed, 0))
                for seed in seeds
            }
        if 'per_seed_l1_means' in result:
            entry['per_seed_l1_means'] = {
                str(seed): [float(v) for v in result['per_seed_l1_means'].get(seed, [])]
                for seed in seeds
            }
        if 'per_seed_ssim_means' in result:
            entry['per_seed_ssim_means'] = {
                str(seed): [float(v) for v in result['per_seed_ssim_means'].get(seed, [])]
                for seed in seeds
            }
        
        serializable_data['results'].append(entry)
    
    with open(json_path, 'w') as f:
        json.dump(serializable_data, f, indent=2)
    
    logger.info(f"Saved frame decay results to: {json_path}")
    return json_path


def print_summary(results: list[dict], seeds: list[int]):
    """Print summary of frame decay analysis."""
    logger.info("\n" + "=" * 80)
    logger.info("FRAME DECAY ANALYSIS SUMMARY")
    logger.info("=" * 80)
    
    # Print dataset configuration
    logger.info(f"\nDatasets evaluated ({len(DATASET_CONFIGS)} total):")
    for cfg in DATASET_CONFIGS:
        logger.info(f"  - {cfg['name']}: {cfg['path']}")
    
    if not results:
        logger.warning("No results to summarize")
        return
    
    for result in results:
        checkpoint_label = result['checkpoint_label']
        num_episodes = result.get('num_episodes', 0)
        max_frames = result.get('max_frames', 0)
        
        logger.info(f"\nCheckpoint: {checkpoint_label}")
        logger.info("-" * 60)
        logger.info(f"  Total episodes evaluated: {num_episodes}")
        logger.info(f"  Seeds: {seeds}")
        logger.info(f"  Maximum frames: {max_frames}")
        
        # Summarize L1 decay
        l1_data = result.get('aggregated_l1_per_frame', [])
        if l1_data:
            l1_means = [d['mean'] for d in l1_data]
            if len(l1_means) > 1:
                frames = np.arange(len(l1_means))
                slope = np.polyfit(frames, l1_means, 1)[0]
                
                logger.info(f"  L1 Distance:")
                logger.info(f"    - First frame: {l1_means[0]:.6f}")
                logger.info(f"    - Last frame: {l1_means[-1]:.6f}")
                logger.info(f"    - Increase: {l1_means[-1] - l1_means[0]:.6f} ({(l1_means[-1]/l1_means[0] - 1)*100:.1f}%)")
                logger.info(f"    - Trend (slope): {slope:.6f}/frame")
        
        # Summarize SSIM decay
        ssim_data = result.get('aggregated_ssim_per_frame', [])
        if ssim_data:
            ssim_means = [d['mean'] for d in ssim_data]
            if len(ssim_means) > 1:
                frames = np.arange(len(ssim_means))
                slope = np.polyfit(frames, ssim_means, 1)[0]
                
                logger.info(f"  SSIM Score:")
                logger.info(f"    - First frame: {ssim_means[0]:.6f}")
                logger.info(f"    - Last frame: {ssim_means[-1]:.6f}")
                logger.info(f"    - Decrease: {ssim_means[0] - ssim_means[-1]:.6f} ({(1 - ssim_means[-1]/ssim_means[0])*100:.1f}%)")
                logger.info(f"    - Trend (slope): {slope:.6f}/frame")
    
    logger.info("\n" + "=" * 80)


def evaluate_all_checkpoints_multi_seed(args: argparse.Namespace):
    """Evaluate all predefined checkpoints with multiple seeds."""
    # Derive seeds
    seeds = derive_eval_seeds(args.seed, args.num_seeds)
    logger.info(f"Derived evaluation seeds: {seeds}")
    
    # Setup output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_save_path = os.path.join(args.save_path, f"run_{timestamp}")
    os.makedirs(base_save_path, exist_ok=True)
    
    # Save run configuration
    run_config = {
        'timestamp': timestamp,
        'arguments': vars(args),
        'seeds': seeds,
        'checkpoint_configs': {k: {'label': v['label'], 'experiment': v['experiment']} for k, v in CHECKPOINT_CONFIGS.items()},
        'datasets': [
            {'name': cfg['name'], 'path': cfg['path'], 'episode_ids': cfg['episode_ids'][:args.num_episodes]}
            for cfg in DATASET_CONFIGS
        ],
        'num_episodes_per_dataset': args.num_episodes,
        'total_episodes_per_seed': len(DATASET_CONFIGS) * args.num_episodes,
    }
    config_path = os.path.join(base_save_path, 'run_config.json')
    with open(config_path, 'w') as f:
        json.dump(run_config, f, indent=2)
    logger.info(f"Saved run configuration to: {config_path}")
    
    # Collect results per seed
    per_seed_results = {}
    
    for seed in seeds:
        logger.info(f"\n{'#'*80}")
        logger.info(f"STARTING EVALUATION WITH SEED {seed}")
        logger.info(f"{'#'*80}\n")
        
        seed_results = []
        seed_save_path = os.path.join(base_save_path, f"seed_{seed}")
        os.makedirs(seed_save_path, exist_ok=True)
        
        for exp_name, exp_config in CHECKPOINT_CONFIGS.items():
            for checkpoint in exp_config['checkpoints']:
                checkpoint_label = f"{exp_config['label']}-{checkpoint['iter']//1000}k"
                checkpoint_save_path = os.path.join(seed_save_path, checkpoint_label)
                os.makedirs(checkpoint_save_path, exist_ok=True)
                
                # Update args with checkpoint-specific settings
                eval_args = deepcopy(args)
                eval_args.experiment = exp_config['experiment']
                
                try:
                    result = evaluate_checkpoint(
                        args=eval_args,
                        ckpt_path=checkpoint['path'],
                        checkpoint_label=checkpoint_label,
                        seed=seed,
                        output_dir=checkpoint_save_path,
                    )
                    
                    if result:
                        seed_results.append(result)
                        
                        # Save intermediate results
                        intermediate_path = os.path.join(
                            seed_save_path,
                            f"intermediate_results_{checkpoint_label}.json"
                        )
                        with open(intermediate_path, 'w') as f:
                            json.dump({
                                'checkpoint_label': result['checkpoint_label'],
                                'checkpoint_path': result['checkpoint_path'],
                                'seed': result['seed'],
                                'num_episodes': result['num_episodes'],
                                'max_frames': result['max_frames'],
                                'aggregated_l1_per_frame': result['aggregated_l1_per_frame'],
                                'aggregated_ssim_per_frame': result['aggregated_ssim_per_frame'],
                            }, f, indent=2)
                        
                except Exception as e:
                    logger.error(f"Failed to evaluate checkpoint {checkpoint_label}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
        
        per_seed_results[seed] = seed_results
    
    # Aggregate results across seeds
    aggregated_results = aggregate_results_across_seeds(per_seed_results, seeds)
    
    # Save final results
    if aggregated_results:
        json_path = save_results_to_json(aggregated_results, seeds, base_save_path, args.num_episodes)
        print_summary(aggregated_results, seeds)
        logger.info(f"\nFinal results saved to: {json_path}")
    else:
        logger.warning("No results were generated across seeds")
    
    return aggregated_results


def evaluate_single_checkpoint_multi_seed(args: argparse.Namespace):
    """Evaluate a single checkpoint with multiple seeds."""
    if not args.ckpt_path:
        raise ValueError("Please provide --ckpt_path for single checkpoint evaluation")
    
    # Derive seeds
    seeds = derive_eval_seeds(args.seed, args.num_seeds)
    logger.info(f"Derived evaluation seeds: {seeds}")
    
    # Setup output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_save_path = os.path.join(args.save_path, f"run_{timestamp}")
    os.makedirs(base_save_path, exist_ok=True)
    
    # Extract checkpoint label from path
    checkpoint_label = os.path.basename(args.ckpt_path).replace('.pt', '')
    
    # Save run configuration
    run_config = {
        'timestamp': timestamp,
        'arguments': vars(args),
        'seeds': seeds,
        'checkpoint_path': args.ckpt_path,
        'checkpoint_label': checkpoint_label,
        'datasets': [
            {'name': cfg['name'], 'path': cfg['path'], 'episode_ids': cfg['episode_ids'][:args.num_episodes]}
            for cfg in DATASET_CONFIGS
        ],
        'num_episodes_per_dataset': args.num_episodes,
        'total_episodes_per_seed': len(DATASET_CONFIGS) * args.num_episodes,
    }
    config_path = os.path.join(base_save_path, 'run_config.json')
    with open(config_path, 'w') as f:
        json.dump(run_config, f, indent=2)
    
    # Collect results per seed
    per_seed_results = {}
    
    for seed in seeds:
        logger.info(f"\n{'#'*80}")
        logger.info(f"STARTING EVALUATION WITH SEED {seed}")
        logger.info(f"{'#'*80}\n")
        
        seed_save_path = os.path.join(base_save_path, f"seed_{seed}")
        os.makedirs(seed_save_path, exist_ok=True)
        
        try:
            result = evaluate_checkpoint(
                args=args,
                ckpt_path=args.ckpt_path,
                checkpoint_label=checkpoint_label,
                seed=seed,
                output_dir=seed_save_path,
            )
            
            if result:
                per_seed_results[seed] = [result]
                
                # Save intermediate results
                intermediate_path = os.path.join(
                    seed_save_path,
                    f"intermediate_results_{checkpoint_label}.json"
                )
                with open(intermediate_path, 'w') as f:
                    json.dump({
                        'checkpoint_label': result['checkpoint_label'],
                        'checkpoint_path': result['checkpoint_path'],
                        'seed': result['seed'],
                        'num_episodes': result['num_episodes'],
                        'max_frames': result['max_frames'],
                        'aggregated_l1_per_frame': result['aggregated_l1_per_frame'],
                        'aggregated_ssim_per_frame': result['aggregated_ssim_per_frame'],
                    }, f, indent=2)
            else:
                per_seed_results[seed] = []
                
        except Exception as e:
            logger.error(f"Failed to evaluate with seed {seed}: {e}")
            import traceback
            traceback.print_exc()
            per_seed_results[seed] = []
            continue
    
    # Aggregate results across seeds
    aggregated_results = aggregate_results_across_seeds(per_seed_results, seeds)
    
    # Save final results
    if aggregated_results:
        json_path = save_results_to_json(aggregated_results, seeds, base_save_path, args.num_episodes)
        print_summary(aggregated_results, seeds)
        logger.info(f"\nFinal results saved to: {json_path}")
    else:
        logger.warning("No results were generated across seeds")
    
    return aggregated_results


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    args = parse_args()
    
    if args.evaluate_all_checkpoints:
        if not CHECKPOINT_CONFIGS:
            logger.error("No checkpoints defined in CHECKPOINT_CONFIGS. Please add checkpoint configurations.")
            logger.error("Example configuration:")
            logger.error("""
CHECKPOINT_CONFIGS = {
    "cmr-exp-1": {
        "label": "cmr-exp-1",
        "experiment": "cosmos_predict2p5_2B_action_conditioned_cmr_13frame_4nodes_release_oss",
        "checkpoints": [
            {"iter": 10000, "path": "/path/to/checkpoint_10000.pt"},
        ]
    },
}
""")
            exit(1)
        evaluate_all_checkpoints_multi_seed(args)
    else:
        evaluate_single_checkpoint_multi_seed(args)
    
    logger.info("Frame decay analysis complete!")
