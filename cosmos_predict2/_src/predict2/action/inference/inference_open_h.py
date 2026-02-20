#!/usr/bin/env python
"""
Action conditioned inference script for Open-H multi-embodiment post-training.

Supports any embodiment registered in EMBODIMENT_REGISTRY (dVRK, Hamlyn, Stanford,
USTC, Moon, etc.) as well as CMR Versius.  For each episode, it:
1. Uses the first frame as conditioning
2. Generates 12 frames using the model with ground-truth actions
3. Uses the last predicted frame as conditioning for the next chunk
4. Stitches all chunks into a full episode video

Uses LeRobotDataset / WrappedLeRobotSingleDataset directly to ensure actions are
transformed identically to training (relative action computation + normalization
via the transform pipeline).

Action vectors are zero-padded to MAX_ACTION_DIM (44) to match the multi-embodiment
training setup.

Usage:
    # CMR Versius (same as before, just with --embodiment flag):
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python cosmos_predict2/_src/predict2/action/inference/inference_open_h.py \\
        --experiment cosmos_predict2p5_2B_action_conditioned_open_h_13frame_8nodes_release_oss \\
        --ckpt_path /path/to/checkpoint/model_ema_bf16.pt \\
        --dataset_path /CMR_Versius/cholecystectomy_480p \\
        --embodiment cmr_versius \\
        --episode_ids 0,1,2

    # dVRK JHU (monocular):
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python cosmos_predict2/_src/predict2/action/inference/inference_open_h.py \\
        --experiment cosmos_predict2p5_2B_action_conditioned_open_h_13frame_8nodes_release_oss \\
        --ckpt_path /path/to/checkpoint/model_ema_bf16.pt \\
        --dataset_path /path/to/jhu/suturebot_2 \\
        --embodiment jhu_dvrk_mono \\
        --episode_ids 0,1,2

    # Stanford Real (with exclude_splits):
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python cosmos_predict2/_src/predict2/action/inference/inference_open_h.py \\
        --experiment cosmos_predict2p5_2B_action_conditioned_open_h_13frame_8nodes_release_oss \\
        --ckpt_path /path/to/checkpoint/model_ema_bf16.pt \\
        --dataset_path /path/to/stanford/Needle_Transfer \\
        --embodiment dvrk_stanford_real \\
        --exclude_splits fail bad_frames \\
        --episode_ids 0,1,2
"""

import argparse
import os
import time

import mediapy
import numpy as np
import torch
from loguru import logger

from cosmos_predict2._src.predict2.action.inference.inference_pipeline import (
    ActionVideo2WorldInference,
)
from cosmos_predict2._src.predict2.action.datasets.gr00t_dreams.data.dataset import (
    LeRobotDataset,
    WrappedLeRobotSingleDataset,
)
from cosmos_predict2._src.predict2.action.datasets.gr00t_dreams.data.embodiment_tags import EmbodimentTag
from cosmos_predict2._src.predict2.action.datasets.gr00t_dreams.groot_configs import (
    EMBODIMENT_REGISTRY,
    MAX_ACTION_DIM,
    construct_modality_config_and_transforms,
)


# Video frames per inference window (1 context + 12 prediction)
NUM_FRAMES = 13
CHUNK_SIZE = NUM_FRAMES - 1  # 12 action frames per window


def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments for the Open-H inference script."""
    parser = argparse.ArgumentParser(
        description="Action conditioned Cosmos-Predict 2.5 inference for Open-H embodiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- Model arguments ---
    parser.add_argument("--experiment", type=str, required=True, help="Experiment config name")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to the checkpoint (.pt file)")
    parser.add_argument("--s3_cred", type=str, default="credentials/s3_checkpoint.secret")

    # --- Data arguments ---
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the LeRobot-format dataset")
    parser.add_argument(
        "--embodiment", type=str, required=True,
        choices=list(EMBODIMENT_REGISTRY.keys()) + [EmbodimentTag.CMR_VERSIUS.value],
        help="Embodiment tag (e.g., jhu_dvrk_mono, dvrk_stanford_real, cmr_versius)",
    )
    parser.add_argument(
        "--data_split", type=str, default="test",
        choices=["train", "test", "full"],
        help="Data split to use for evaluation",
    )
    parser.add_argument(
        "--episode_ids", type=str, required=True,
        help="Comma-separated list of episode IDs to evaluate (e.g., '0,1,2')",
    )
    parser.add_argument(
        "--exclude_splits", type=str, nargs="+", default=None,
        help="Split names from info.json to exclude (e.g., --exclude_splits fail bad_frames)",
    )

    # --- Inference arguments ---
    parser.add_argument("--guidance", type=float, default=0, help="Classifier-free guidance scale")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # --- Output arguments ---
    parser.add_argument("--save_root", type=str, default="results/open_h_eval", help="Output directory")
    parser.add_argument("--save_fps", type=int, default=10, help="FPS for saved videos")
    parser.add_argument("--save_comparison", action="store_true", help="Save side-by-side GT vs predicted")
    parser.add_argument("--max_chunks", type=int, default=20, help="Maximum chunks per episode (default: 20)")

    # --- Context parallel arguments ---
    parser.add_argument("--context_parallel_size", type=int, default=1, help="Context parallel size (GPUs)")

    return parser.parse_args()


def build_episode_index_map(dataset: WrappedLeRobotSingleDataset) -> dict[int, list[tuple[int, int]]]:
    """Build mapping from episode_id to list of (dataset_idx, base_index), sorted by base_index."""
    all_steps = dataset._all_steps
    episode_map: dict[int, list[tuple[int, int]]] = {}
    for dataset_idx, (episode_id, base_index) in enumerate(all_steps):
        if episode_id not in episode_map:
            episode_map[episode_id] = []
        episode_map[episode_id].append((dataset_idx, base_index))
    for episode_id in episode_map:
        episode_map[episode_id].sort(key=lambda x: x[1])
    return episode_map


def find_chunk_indices(
    episode_map: dict[int, list[tuple[int, int]]],
    episode_id: int,
    timestep_interval: int,
    chunk_size: int = CHUNK_SIZE,
) -> list[int] | None:
    """Find dataset indices for non-overlapping chunks of an episode.

    Chunks start at base_index 0, stride, 2*stride, ...
    where stride = chunk_size * timestep_interval.

    Returns None if the episode doesn't have base_index=0.
    """
    if episode_id not in episode_map:
        logger.warning(f"Episode {episode_id} not found in episode_map")
        return None

    entries = episode_map[episode_id]
    base_index_to_dataset_idx = {base_idx: ds_idx for ds_idx, base_idx in entries}

    if 0 not in base_index_to_dataset_idx:
        return None

    stride = chunk_size * timestep_interval
    chunk_indices = []
    base_index = 0
    while base_index in base_index_to_dataset_idx:
        chunk_indices.append(base_index_to_dataset_idx[base_index])
        base_index += stride

    return chunk_indices


def pad_action(action: np.ndarray, max_dim: int) -> np.ndarray:
    """Zero-pad action to max_dim along the last axis (matching training pipeline)."""
    if action.shape[-1] >= max_dim:
        return action
    pad_width = [(0, 0)] * (action.ndim - 1) + [(0, max_dim - action.shape[-1])]
    return np.pad(action, pad_width, mode="constant", constant_values=0.0)


def main():
    torch.set_grad_enabled(False)
    args = parse_arguments()

    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    embodiment = args.embodiment

    # Resolve timestep_interval from the registry (or CMR default)
    if embodiment == EmbodimentTag.CMR_VERSIUS.value:
        timestep_interval = 6  # CMR: 60Hz → 10fps
    else:
        reg = EMBODIMENT_REGISTRY.get(embodiment)
        if reg is None:
            raise ValueError(f"Unknown embodiment '{embodiment}'. Available: {list(EMBODIMENT_REGISTRY.keys())}")
        timestep_interval = reg["timestep_interval"]

    logger.info(f"Embodiment: {embodiment}")
    logger.info(f"Timestep interval: {timestep_interval}")
    logger.info(f"Chunk size: {CHUNK_SIZE} action frames")
    logger.info(f"Max action dim (padded): {MAX_ACTION_DIM}")

    # Build modality config + transforms using the REAL pipeline
    config, train_transform, test_transform = construct_modality_config_and_transforms(
        num_frames=NUM_FRAMES, embodiment=embodiment, downscaled_res=False,
    )

    # Extract modality_filename from config if present
    modality_filename = None
    if isinstance(config, dict) and "modality_filename" in config:
        modality_filename = config.pop("modality_filename")

    # Create dataset with the same transform pipeline as training
    logger.info(f"Loading dataset from {args.dataset_path} with split '{args.data_split}'")
    dataset = WrappedLeRobotSingleDataset(
        dataset_path=args.dataset_path,
        modality_configs=config,
        transforms=test_transform,
        embodiment_tag=embodiment,
        data_split=args.data_split,
        modality_filename=modality_filename,
        exclude_splits=args.exclude_splits,
    )

    # Build mapping from episode_id to dataset indices
    episode_map = build_episode_index_map(dataset)
    logger.info(f"Built index map for {len(episode_map)} episodes in '{args.data_split}' split")
    logger.info(f"Episode IDs in map: {sorted(episode_map.keys())[:20]}{'...' if len(episode_map) > 20 else ''}")

    # Determine which episodes to evaluate
    episode_ids = [int(x) for x in args.episode_ids.split(",")]
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
    save_root = os.path.join(args.save_root, embodiment)
    os.makedirs(save_root, exist_ok=True)
    os.makedirs(os.path.join(save_root, "predicted"), exist_ok=True)
    if args.save_comparison:
        os.makedirs(os.path.join(save_root, "comparison"), exist_ok=True)

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
            chunk_indices = find_chunk_indices(episode_map, episode_id, timestep_interval)

            if chunk_indices is None:
                logger.warning(f"Episode {episode_id} doesn't start at base_index=0 in this split, skipping")
                continue

            if len(chunk_indices) == 0:
                logger.warning(f"No chunks found for episode {episode_id}, skipping")
                continue

            total_chunks = len(chunk_indices)
            if args.max_chunks > 0 and total_chunks > args.max_chunks:
                chunk_indices = chunk_indices[:args.max_chunks]
                logger.info(f"Episode {episode_id}: processing {len(chunk_indices)}/{total_chunks} chunks (limited by --max_chunks)")
            else:
                logger.info(f"Episode {episode_id}: {len(chunk_indices)} chunks")

            predicted_chunks = []
            gt_chunks = []
            current_frame = None
            episode_start_time = time.perf_counter()
            episode_chunk_times = []

            for chunk_idx, dataset_idx in enumerate(chunk_indices):
                data = dataset[dataset_idx]

                # video shape from WrappedLeRobotSingleDataset: (C, T, H, W)
                video = data["video"]
                if isinstance(video, torch.Tensor):
                    video = video.permute(1, 2, 3, 0).numpy()  # (T, H, W, C)
                elif video.ndim == 4 and video.shape[0] == 3:
                    video = np.transpose(video, (1, 2, 3, 0))

                actions = data["action"]
                if isinstance(actions, torch.Tensor):
                    actions = actions.numpy()

                # Pad actions to unified MAX_ACTION_DIM (matching training)
                actions = pad_action(actions, MAX_ACTION_DIM)

                if chunk_idx == 0:
                    action_dim_raw = data["action"].shape[-1] if isinstance(data["action"], torch.Tensor) else data["action"].shape[-1]
                    logger.info(f"  Data shapes — video: {data['video'].shape}, "
                                f"action: {action_dim_raw}D (raw) → {actions.shape[-1]}D (padded)")
                    current_frame = video[0]  # (H, W, C)

                gt_chunks.append(video)

                # Run inference
                torch.cuda.synchronize()
                chunk_start_time = time.perf_counter()

                next_frame, video_chunk = video2world.step_inference(
                    img_array=current_frame,
                    action=actions.astype(np.float32),
                    guidance=args.guidance,
                    seed=args.seed + chunk_idx,
                    num_latent_conditional_frames=1,
                )

                torch.cuda.synchronize()
                chunk_end_time = time.perf_counter()
                chunk_inference_time = chunk_end_time - chunk_start_time
                episode_chunk_times.append(chunk_inference_time)

                frames_in_chunk = NUM_FRAMES - 1
                chunk_fps = frames_in_chunk / chunk_inference_time

                predicted_chunks.append(video_chunk)
                current_frame = next_frame

                logger.info(f"  Chunk {chunk_idx + 1}/{len(chunk_indices)} | "
                           f"Time: {chunk_inference_time:.2f}s | FPS: {chunk_fps:.2f}")

            if not predicted_chunks:
                logger.warning(f"No chunks generated for episode {episode_id}")
                continue

            # Episode performance
            episode_inference_time = sum(episode_chunk_times)
            episode_frames = len(chunk_indices) * (NUM_FRAMES - 1)
            episode_fps = episode_frames / episode_inference_time if episode_inference_time > 0 else 0

            perf_stats["total_episodes"] += 1
            perf_stats["total_chunks"] += len(chunk_indices)
            perf_stats["total_frames_generated"] += episode_frames
            perf_stats["total_inference_time"] += episode_inference_time
            perf_stats["chunk_times"].extend(episode_chunk_times)
            perf_stats["episode_times"].append(episode_inference_time)
            perf_stats["episode_fps"].append(episode_fps)

            logger.info(f"Episode {episode_id}: {len(chunk_indices)} chunks, "
                       f"{episode_frames} frames, {episode_inference_time:.2f}s, "
                       f"{episode_fps:.2f} FPS")

            # Stitch chunks
            stitched_predicted = [predicted_chunks[0]]
            for chunk in predicted_chunks[1:]:
                stitched_predicted.append(chunk[1:])
            predicted_video = np.concatenate(stitched_predicted, axis=0)

            # Save predicted video
            pred_path = os.path.join(save_root, "predicted", f"episode_{episode_id:04d}.mp4")
            mediapy.write_video(pred_path, predicted_video, fps=args.save_fps)
            logger.info(f"Saved predicted video to {pred_path}")

            # Save comparison if requested
            if args.save_comparison:
                stitched_gt = [gt_chunks[0]]
                for chunk in gt_chunks[1:]:
                    stitched_gt.append(chunk[1:])
                gt_video = np.concatenate(stitched_gt, axis=0)

                min_len = min(len(gt_video), len(predicted_video))
                comparison = np.concatenate([gt_video[:min_len], predicted_video[:min_len]], axis=2)

                comp_path = os.path.join(save_root, "comparison", f"episode_{episode_id:04d}.mp4")
                mediapy.write_video(comp_path, comparison, fps=args.save_fps)
                logger.info(f"Saved comparison to {comp_path}")

        except Exception as e:
            logger.error(f"Error processing episode {episode_id}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Performance summary
    logger.info("=" * 60)
    logger.info(f"PERFORMANCE SUMMARY — {embodiment}")
    logger.info("=" * 60)

    if perf_stats["total_episodes"] > 0:
        avg_fps = perf_stats["total_frames_generated"] / perf_stats["total_inference_time"] if perf_stats["total_inference_time"] > 0 else 0
        avg_chunk_time = np.mean(perf_stats["chunk_times"])
        avg_episode_time = np.mean(perf_stats["episode_times"])

        logger.info(f"Episodes: {perf_stats['total_episodes']} | "
                    f"Chunks: {perf_stats['total_chunks']} | "
                    f"Frames: {perf_stats['total_frames_generated']}")
        logger.info(f"Total inference: {perf_stats['total_inference_time']:.2f}s | "
                    f"Avg FPS: {avg_fps:.2f}")
        logger.info(f"Avg chunk time: {avg_chunk_time:.2f}s | "
                    f"Avg episode time: {avg_episode_time:.2f}s")
    else:
        logger.warning("No episodes were successfully processed.")

    logger.info("=" * 60)

    video2world.cleanup()
    logger.info("Done!")


if __name__ == "__main__":
    main()
