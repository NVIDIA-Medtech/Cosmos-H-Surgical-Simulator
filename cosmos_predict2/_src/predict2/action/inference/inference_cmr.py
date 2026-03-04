#!/usr/bin/env python
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
Intended Script Location: cosmos_predict2/_src/predict2/action/inference/inference_cmr.py

Action conditioned inference script for CMR Versius surgical robot post-training.

This script runs autoregressive video generation on episodes from LeRobot datasets.
For each episode, it:
1. Uses the first frame as conditioning
2. Generates 12 frames using the model with ground-truth actions
3. Uses the last predicted frame as conditioning for the next chunk
4. Stitches all chunks into a full episode video

Uses LeRobotDataset directly to ensure actions are transformed identically to training
(relative action computation + normalization via the transform pipeline).

Usage:
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python cosmos_predict2/_src/predict2/action/inference/inference_cmr.py \
        --experiment=cosmos_predict2p5_2B_action_conditioned_cmr_13frame_4nodes_release_oss \
        --ckpt_path /path/to/checkpoint/model_ema_bf16.pt \
        --dataset_path /CMR_Versius/cholecystectomy_480p \
        --save_root results/cmr_eval/cholecystectomy_480p \
        --data_split test \
        --episode_ids 0,1,2
"""

import argparse
import os
import time

import mediapy
import numpy as np
import torch
from loguru import logger

from cosmos_predict2._src.predict2.action.datasets.gr00t_dreams.data.dataset import (
    LeRobotDataset,
)
from cosmos_predict2._src.predict2.action.inference.inference_pipeline import (
    ActionVideo2WorldInference,
)

# Constants matching training config in groot_configs.py for cmr_versius
# CMR Versius: 60Hz data → 10fps (TIMESTEP_INTERVAL=6), 13 video frames, 12 action frames
NUM_FRAMES = 13  # Video frames (1 context + 12 prediction)
TIMESTEP_INTERVAL = 6  # Frame stride: 60Hz / 10fps = 6
CHUNK_SIZE = 12  # Action frames per window (NUM_FRAMES - 1)


def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments for the cmr inference script."""
    parser = argparse.ArgumentParser(description="Action conditioned Cosmos-Predict 2.5 inference script")

    # Model arguments
    parser.add_argument("--experiment", type=str, required=True, help="Experiment config name")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="Path to the checkpoint (.pt file)",
    )
    parser.add_argument("--s3_cred", type=str, default="credentials/s3_checkpoint.secret")

    # Data arguments
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the LeRobot-format dataset",
    )
    parser.add_argument(
        "--data_split",
        type=str,
        default="test",
        choices=["train", "test", "full"],
        help="Data split to use for evaluation",
    )
    parser.add_argument(
        "--episode_ids",
        type=str,
        required=True,
        help="Comma-separated list of episode IDs to evaluate (e.g., '0,1,2'). If not specified, evaluates all episodes in the split.",
    )

    # Inference arguments
    parser.add_argument("--guidance", type=float, default=0, help="Classifier-free guidance scale")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Output arguments
    parser.add_argument("--save_root", type=str, default="results/cmr_eval", help="Output directory")
    parser.add_argument("--save_fps", type=int, default=10, help="FPS for saved videos")
    parser.add_argument(
        "--save_comparison",
        action="store_true",
        help="Save side-by-side comparison with ground truth",
    )
    parser.add_argument(
        "--max_chunks",
        type=int,
        default=20,
        help="Maximum number of chunks to process per episode (default: 20)",
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
    # Access the underlying WrappedLeRobotSingleDataset
    # LeRobotDataset wraps one or more WrappedLeRobotSingleDataset
    inner_dataset = dataset.lerobot_datasets[0]
    all_steps = inner_dataset._all_steps  # List of (episode_id, base_index) tuples

    # Build mapping: episode_id -> [(dataset_idx, base_index), ...]
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
    """
    Get the unique episode IDs present in the dataset (after split is applied).

    Args:
        dataset: The LeRobotDataset instance (already has split applied)

    Returns:
        Sorted list of episode IDs in the dataset
    """
    inner_dataset = dataset.lerobot_datasets[0]
    all_steps = inner_dataset._all_steps
    episode_ids = sorted(set(ep_id for ep_id, _ in all_steps))
    return episode_ids


def find_chunk_indices(
    episode_map: dict[int, list[tuple[int, int]]],
    episode_id: int,
    chunk_size: int = CHUNK_SIZE,
    timestep_interval: int = TIMESTEP_INTERVAL,
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

    Returns:
        List of dataset indices for non-overlapping chunks, or None if episode
        doesn't have base_index=0 (i.e., episode is only partially in the split)
    """
    if episode_id not in episode_map:
        logger.warning(f"Episode {episode_id} not found in episode_map")
        return None

    entries = episode_map[episode_id]  # [(dataset_idx, base_index), ...] sorted by base_index
    base_index_to_dataset_idx = {base_idx: ds_idx for ds_idx, base_idx in entries}

    # Must have base_index=0 to start autoregressive inference from beginning
    if 0 not in base_index_to_dataset_idx:
        return None

    # We need base_indices: 0, 36, 72, 108, ... (chunk_size * timestep_interval increments)
    stride = chunk_size * timestep_interval
    chunk_indices = []

    base_index = 0
    while base_index in base_index_to_dataset_idx:
        chunk_indices.append(base_index_to_dataset_idx[base_index])
        base_index += stride

    return chunk_indices


def main():
    torch.set_grad_enabled(False)
    args = parse_arguments()

    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Load dataset with transforms - ensures actions are processed identically to training
    logger.info(f"Loading LeRobotDataset from {args.dataset_path} with split '{args.data_split}'")

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
        dataset_path=args.dataset_path,
        data_split=args.data_split,
        embodiment="cmr_versius",
        downscaled_res=False,
    )

    # Build mapping from episode_id to dataset indices
    episode_map = build_episode_index_map(dataset)
    logger.info(f"Built index map for {len(episode_map)} episodes in '{args.data_split}' split")
    logger.info(f"Episode IDs in map: {sorted(episode_map.keys())}")

    # Determine which episodes to evaluate
    if args.episode_ids:
        episode_ids = [int(x) for x in args.episode_ids.split(",")]
    else:
        # Use all episodes in the split that have base_index=0 (complete episodes)
        episode_ids = get_episode_ids_in_split(dataset)

    logger.info(f"Requested episodes: {episode_ids}")

    # Initialize inference pipeline
    logger.info(f"Loading model from {args.ckpt_path}")
    video2world = ActionVideo2WorldInference(
        args.experiment,
        args.ckpt_path,
        args.s3_cred,
        context_parallel_size=args.context_parallel_size,
    )

    mem_bytes = torch.cuda.memory_allocated()
    logger.info(f"GPU memory after model load: {mem_bytes / (1024**3):.2f} GB")

    # Create output directories
    os.makedirs(args.save_root, exist_ok=True)
    os.makedirs(os.path.join(args.save_root, "predicted"), exist_ok=True)
    if args.save_comparison:
        os.makedirs(os.path.join(args.save_root, "comparison"), exist_ok=True)

    # Performance tracking
    perf_stats = {
        "total_episodes": 0,
        "total_chunks": 0,
        "total_frames_generated": 0,
        "total_inference_time": 0.0,
        "chunk_times": [],
        "episode_times": [],
        "episode_fps": [],
    }

    # Process each episode
    for episode_id in episode_ids:
        logger.info(f"Processing episode {episode_id}")

        try:
            # Find dataset indices for non-overlapping chunks of this episode
            # Returns None if episode doesn't have base_index=0 (partial episode in split)
            chunk_indices = find_chunk_indices(episode_map, episode_id)

            if chunk_indices is None:
                logger.warning(f"Episode {episode_id} doesn't start at base_index=0 in this split, skipping")
                continue

            if len(chunk_indices) == 0:
                logger.warning(f"No chunks found for episode {episode_id}, skipping")
                continue

            # Limit number of chunks if specified
            total_chunks = len(chunk_indices)
            if args.max_chunks > 0 and total_chunks > args.max_chunks:
                chunk_indices = chunk_indices[: args.max_chunks]
                logger.info(
                    f"Episode {episode_id}: processing {len(chunk_indices)}/{total_chunks} chunks (limited by --max_chunks)"
                )
            else:
                logger.info(f"Episode {episode_id}: {len(chunk_indices)} chunks")

            predicted_chunks = []
            gt_chunks = []  # For comparison
            current_frame = None
            episode_start_time = time.perf_counter()
            episode_chunk_times = []

            for chunk_idx, dataset_idx in enumerate(chunk_indices):
                # Get data from dataset - actions are already transformed (relative + normalized)
                data = dataset[dataset_idx]

                # video shape: (C, T, H, W) -> need (T, H, W, C) for inference
                video = data["video"].permute(1, 2, 3, 0).numpy()  # (T, H, W, C)
                actions = data["action"].numpy()  # (chunk_size, action_dim) - already normalized

                if chunk_idx == 0:
                    # Log shapes on first chunk for verification
                    logger.info(
                        f"  Data shapes - video: {data['video'].shape} (C,T,H,W), "
                        f"action: {actions.shape} (T,D), "
                        f"expected: (3,{NUM_FRAMES},H,W), ({CHUNK_SIZE},44)"
                    )

                    # First chunk: use ground truth first frame as conditioning
                    current_frame = video[0]  # (H, W, C)

                # Store ground truth for comparison
                gt_chunks.append(video)

                # Run inference with timing
                torch.cuda.synchronize()  # Ensure GPU is ready
                chunk_start_time = time.perf_counter()

                next_frame, video_chunk = video2world.step_inference(
                    img_array=current_frame,
                    action=actions.astype(np.float32),
                    guidance=args.guidance,
                    seed=args.seed + chunk_idx,
                    num_latent_conditional_frames=1,
                )

                torch.cuda.synchronize()  # Wait for GPU to finish
                chunk_end_time = time.perf_counter()
                chunk_inference_time = chunk_end_time - chunk_start_time
                episode_chunk_times.append(chunk_inference_time)

                # Calculate chunk metrics (each chunk generates NUM_FRAMES-1=12 new frames)
                frames_in_chunk = NUM_FRAMES - 1  # 12 frames generated per chunk
                chunk_fps = frames_in_chunk / chunk_inference_time

                predicted_chunks.append(video_chunk)

                # Use last predicted frame as next conditioning
                current_frame = next_frame

                logger.info(
                    f"  Chunk {chunk_idx + 1}/{len(chunk_indices)} complete | "
                    f"Time: {chunk_inference_time:.2f}s | "
                    f"FPS: {chunk_fps:.2f} | "
                    f"Frames: {frames_in_chunk}"
                )

            if not predicted_chunks:
                logger.warning(f"No chunks generated for episode {episode_id}")
                continue

            # Calculate episode performance metrics
            episode_end_time = time.perf_counter()
            episode_total_time = episode_end_time - episode_start_time
            episode_inference_time = sum(episode_chunk_times)
            episode_frames = len(chunk_indices) * (NUM_FRAMES - 1)  # Total frames generated
            episode_fps = episode_frames / episode_inference_time if episode_inference_time > 0 else 0

            # Update global stats
            perf_stats["total_episodes"] += 1
            perf_stats["total_chunks"] += len(chunk_indices)
            perf_stats["total_frames_generated"] += episode_frames
            perf_stats["total_inference_time"] += episode_inference_time
            perf_stats["chunk_times"].extend(episode_chunk_times)
            perf_stats["episode_times"].append(episode_inference_time)
            perf_stats["episode_fps"].append(episode_fps)

            logger.info(f"Episode {episode_id} performance summary:")
            logger.info(f"  Total inference time: {episode_inference_time:.2f}s")
            logger.info(f"  Total wall time (incl. data loading): {episode_total_time:.2f}s")
            logger.info(f"  Chunks processed: {len(chunk_indices)}")
            logger.info(f"  Frames generated: {episode_frames}")
            logger.info(f"  Average FPS: {episode_fps:.2f}")
            logger.info(f"  Average time per chunk: {episode_inference_time / len(chunk_indices):.2f}s")
            logger.info(f"  Time per generated video (inference): {episode_inference_time:.2f}s")

            # Stitch chunks together
            # First chunk: all frames, subsequent chunks: skip first frame (it's the conditioning)
            stitched_predicted = [predicted_chunks[0]]
            for chunk in predicted_chunks[1:]:
                stitched_predicted.append(chunk[1:])
            predicted_video = np.concatenate(stitched_predicted, axis=0)

            # Save predicted video
            pred_path = os.path.join(args.save_root, "predicted", f"episode_{episode_id:04d}.mp4")
            mediapy.write_video(pred_path, predicted_video, fps=args.save_fps)
            logger.info(f"Saved predicted video to {pred_path}")

            # Save side-by-side comparison if requested
            if args.save_comparison:
                # Stitch ground truth the same way
                stitched_gt = [gt_chunks[0]]
                for chunk in gt_chunks[1:]:
                    stitched_gt.append(chunk[1:])
                gt_video = np.concatenate(stitched_gt, axis=0)

                # Trim to same length
                min_len = min(len(gt_video), len(predicted_video))
                gt_video = gt_video[:min_len]
                predicted_video_trimmed = predicted_video[:min_len]

                # Concatenate side by side (GT on left, predicted on right)
                comparison = np.concatenate([gt_video, predicted_video_trimmed], axis=2)

                comp_path = os.path.join(args.save_root, "comparison", f"episode_{episode_id:04d}.mp4")
                mediapy.write_video(comp_path, comparison, fps=args.save_fps)
                logger.info(f"Saved comparison video to {comp_path}")

        except Exception as e:
            logger.error(f"Error processing episode {episode_id}: {e}")
            import traceback

            traceback.print_exc()
            continue

    # Print final performance summary
    logger.info("=" * 60)
    logger.info("PERFORMANCE SUMMARY")
    logger.info("=" * 60)

    if perf_stats["total_episodes"] > 0:
        avg_fps = (
            perf_stats["total_frames_generated"] / perf_stats["total_inference_time"]
            if perf_stats["total_inference_time"] > 0
            else 0
        )
        avg_chunk_time = np.mean(perf_stats["chunk_times"]) if perf_stats["chunk_times"] else 0
        std_chunk_time = np.std(perf_stats["chunk_times"]) if perf_stats["chunk_times"] else 0
        avg_episode_time = np.mean(perf_stats["episode_times"]) if perf_stats["episode_times"] else 0
        std_episode_time = np.std(perf_stats["episode_times"]) if perf_stats["episode_times"] else 0

        logger.info(f"Total episodes processed: {perf_stats['total_episodes']}")
        logger.info(f"Total chunks processed: {perf_stats['total_chunks']}")
        logger.info(f"Total frames generated: {perf_stats['total_frames_generated']}")
        logger.info(f"Total inference time: {perf_stats['total_inference_time']:.2f}s")
        logger.info("-" * 60)
        logger.info(f"Overall average FPS: {avg_fps:.2f}")
        logger.info(f"Average time per chunk: {avg_chunk_time:.2f}s (+/- {std_chunk_time:.2f}s)")
        logger.info(f"Average inference time per video: {avg_episode_time:.2f}s (+/- {std_episode_time:.2f}s)")
        logger.info(f"Frames per chunk: {NUM_FRAMES - 1}")
        logger.info("-" * 60)

        # Per-episode breakdown
        if len(perf_stats["episode_fps"]) > 1:
            logger.info(
                f"FPS range across episodes: {min(perf_stats['episode_fps']):.2f} - {max(perf_stats['episode_fps']):.2f}"
            )
            logger.info(
                f"Episode time range: {min(perf_stats['episode_times']):.2f}s - {max(perf_stats['episode_times']):.2f}s"
            )

        # Throughput metrics
        if perf_stats["total_inference_time"] > 0:
            videos_per_minute = 60.0 / avg_episode_time if avg_episode_time > 0 else 0
            chunks_per_second = perf_stats["total_chunks"] / perf_stats["total_inference_time"]
            logger.info(f"Throughput: {videos_per_minute:.2f} videos/min | {chunks_per_second:.2f} chunks/sec")
    else:
        logger.warning("No episodes were successfully processed.")

    logger.info("=" * 60)

    # Cleanup
    video2world.cleanup()
    logger.info("Done!")


if __name__ == "__main__":
    main()
