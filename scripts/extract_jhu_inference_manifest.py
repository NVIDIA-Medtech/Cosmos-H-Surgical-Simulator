#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Create streaming-inference manifests from JHU dVRK tabletop episodes.

The script writes one ground-truth MP4 and one normalized, 44D-padded action
array per selected episode, plus the JSON manifest consumed by
``action_video2world_streaming``. It uses the same ``MixedLeRobotDataset``
transforms and ``meta/stats_cosmos.json`` files as the reference training
recipe, so actions must not be normalized again downstream.

Example:

.. code-block:: bash

    python scripts/extract_jhu_inference_manifest.py \
        --subset hf_suturebot \
        --episode-ids 1440,1441,1442 \
        --num-frames 73 \
        --output-dir sf_inference_data/jhu_tabletop_test_h73

For multiple subsets, pass ``--episodes-json`` with a list of objects:

.. code-block:: json

    [
      {
        "subset": "hf_suturebot",
        "path": "/datasets/jhu_tabletop/hf_suturebot",
        "ids": [1440, 1441, 1442]
      }
    ]

The ``path`` field is optional when the subset basename is present in
``JHU_DVRK_MONO_FINETUNE_TRAIN_DATASET_SPECS``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

DEFAULT_TIMESTEP_INTERVAL = 3


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract JHU dVRK episode MP4/action pairs and an input manifest "
            "for reference-tabletop streaming inference."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--subset", help="Dataset basename, for example hf_suturebot.")
    parser.add_argument("--episode-ids", help="Comma-separated LeRobot episode IDs.")
    parser.add_argument("--dataset-path", help="Override the path resolved for --subset.")
    parser.add_argument(
        "--episodes-json",
        help="JSON list of {subset, path?, ids} objects; overrides single-subset arguments.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=73,
        help="Pixel-frame horizon per window; (num_frames-1) must be divisible by four.",
    )
    parser.add_argument(
        "--num-chunks",
        type=int,
        default=1,
        help="Maximum consecutive non-overlapping windows per episode.",
    )
    parser.add_argument(
        "--start-margin",
        type=int,
        default=0,
        help="Raw source-frame offset for the first window.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "test", "full"),
        default="test",
        help="Select the augmented or deterministic modality transform.",
    )
    parser.add_argument(
        "--timestep-interval",
        type=int,
        default=DEFAULT_TIMESTEP_INTERVAL,
        help="Raw-frame stride used by the JHU embodiment.",
    )
    parser.add_argument(
        "--test-split-ratio",
        type=float,
        default=0.02,
        help="Held-out ratio used while constructing the dataset.",
    )
    parser.add_argument("--output-dir", required=True, help="Manifest and episode output directory.")
    parser.add_argument(
        "--tag",
        default="jhu_tabletop",
        help="Filename, manifest, and prediction-directory prefix.",
    )
    parser.add_argument(
        "--predicted-output-root",
        default="interactive-output",
        help="Manifest output_video directory prefix.",
    )
    parser.add_argument("--fps", type=float, default=10.0, help="Output MP4 frame rate.")
    return parser.parse_args()


def _load_episode_specs(args: argparse.Namespace) -> list[dict]:
    if args.episodes_json:
        with open(args.episodes_json) as file:
            raw = json.load(file)
        if not isinstance(raw, list):
            raise ValueError("--episodes-json must contain a JSON list")
        specs = []
        for index, entry in enumerate(raw):
            if "subset" not in entry or "ids" not in entry:
                raise ValueError(f"--episodes-json entry {index} requires 'subset' and 'ids'")
            specs.append(
                {
                    "subset": entry["subset"],
                    "path": entry.get("path"),
                    "ids": [int(value) for value in entry["ids"]],
                }
            )
        return specs

    if not args.subset or not args.episode_ids:
        raise ValueError("provide --episodes-json or both --subset and --episode-ids")
    ids = [int(value) for value in args.episode_ids.split(",") if value.strip()]
    return [{"subset": args.subset, "path": args.dataset_path, "ids": ids}]


def _video_to_numpy(sample_video) -> np.ndarray:
    """Convert a ``(C,T,H,W)`` sample to ``(T,H,W,C)`` uint8."""
    array = sample_video.permute(1, 2, 3, 0).cpu().numpy()
    if array.dtype == np.uint8:
        return array
    return (np.clip(array, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def main() -> int:
    args = parse_arguments()
    if args.num_frames < 2 or (args.num_frames - 1) % 4 != 0:
        print(
            f"ERROR: --num-frames must be >= 2 and satisfy "
            f"(num_frames-1) % 4 == 0; got {args.num_frames}",
            file=sys.stderr,
        )
        return 2
    if args.num_chunks < 1:
        print("ERROR: --num-chunks must be >= 1", file=sys.stderr)
        return 2

    try:
        episode_specs = _load_episode_specs(args)
    except ValueError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    try:
        import mediapy

        from cosmos_predict2._src.predict2.action.datasets.gr00t_dreams.data.dataset import (
            MixedLeRobotDataset,
        )
        from cosmos_predict2._src.predict2.action.datasets.gr00t_dreams.data.embodiment_tags import (
            EmbodimentTag,
        )
        from cosmos_predict2._src.predict2.action.datasets.gr00t_dreams.groot_configs import (
            JHU_DVRK_MONO_FINETUNE_TRAIN_DATASET_SPECS,
            MAX_ACTION_DIM,
        )
    except ImportError as error:
        print(
            f"ERROR: failed to import runtime dependencies: {error}\n"
            "Run inside the Cosmos-Predict2.5 environment.",
            file=sys.stderr,
        )
        return 2

    subset_to_path = {
        Path(spec["path"]).name: spec["path"]
        for spec in JHU_DVRK_MONO_FINETUNE_TRAIN_DATASET_SPECS
    }
    num_actions = args.num_frames - 1
    raw_chunk_stride = num_actions * args.timestep_interval
    output_dir = Path(args.output_dir).expanduser().resolve()
    episodes_dir = output_dir / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    predicted_dir = Path(args.predicted_output_root) / args.tag
    manifest_entries: list[dict] = []

    for spec in episode_specs:
        subset = spec["subset"]
        dataset_path = spec["path"] or subset_to_path.get(subset)
        if dataset_path is None:
            print(f"ERROR: no path found for subset '{subset}'", file=sys.stderr)
            return 2
        dataset_path = os.path.abspath(os.path.expanduser(dataset_path))
        if not os.path.isdir(dataset_path):
            print(f"ERROR: dataset path is missing: {dataset_path}", file=sys.stderr)
            return 2

        dataset = MixedLeRobotDataset(
            dataset_specs=[
                {
                    "path": dataset_path,
                    "embodiment": EmbodimentTag.JHU_DVRK_MONO,
                    "mix_ratio": 1.0,
                }
            ],
            num_frames=args.num_frames,
            data_split=args.split,
            max_action_dim=MAX_ACTION_DIM,
            downscaled_res=False,
            test_split_ratio=args.test_split_ratio,
        )
        sub_dataset = dataset.sub_datasets[0]
        trajectory_ids = np.asarray(sub_dataset.trajectory_ids).astype(int)
        trajectory_lengths = np.asarray(sub_dataset.trajectory_lengths).astype(int)
        length_by_id = dict(zip(trajectory_ids.tolist(), trajectory_lengths.tolist()))

        def fetch_window(episode_id: int, base_index: int):
            saved_steps = sub_dataset._all_steps
            try:
                sub_dataset._all_steps = [(episode_id, base_index)]
                return dataset[0]
            finally:
                sub_dataset._all_steps = saved_steps

        for episode_id in spec["ids"]:
            if episode_id not in length_by_id:
                print(f"WARNING: episode {episode_id} is absent from '{subset}'; skipping")
                continue
            usable_frames = length_by_id[episode_id] - 1 - args.start_margin
            chunks_that_fit = max(0, usable_frames // raw_chunk_stride)
            chunks_to_use = min(args.num_chunks, chunks_that_fit)
            if chunks_to_use < 1:
                print(f"WARNING: episode {episode_id} is shorter than one full window; skipping")
                continue

            action_chunks = []
            video_chunks = []
            for chunk_index in range(chunks_to_use):
                base_index = args.start_margin + chunk_index * raw_chunk_stride
                sample = fetch_window(episode_id, base_index)
                actions = sample["action"].cpu().numpy().astype(np.float32)
                if actions.shape != (num_actions, MAX_ACTION_DIM):
                    raise ValueError(
                        f"episode {episode_id} produced action shape {actions.shape}; "
                        f"expected {(num_actions, MAX_ACTION_DIM)}"
                    )
                action_chunks.append(actions)
                video_chunks.append(_video_to_numpy(sample["video"]))

            actions = np.concatenate(action_chunks, axis=0)
            video = np.concatenate(
                [video_chunks[0], *(chunk[1:] for chunk in video_chunks[1:])],
                axis=0,
            )
            stem = f"{args.tag}_{subset}_ep{episode_id:06d}"
            video_path = episodes_dir / f"{stem}.mp4"
            actions_path = episodes_dir / f"{stem}_actions.npy"
            mediapy.write_video(str(video_path), video, fps=args.fps)
            np.save(actions_path, actions)

            manifest_entries.append(
                {
                    "input_video": os.path.relpath(video_path, start=os.getcwd()),
                    "input_action": os.path.relpath(actions_path, start=os.getcwd()),
                    "output_video": str(predicted_dir / f"{stem}.mp4"),
                    "_subset": subset,
                    "_episode_id": episode_id,
                    "_num_chunks": chunks_to_use,
                    "_action_shape": list(actions.shape),
                    "_video_shape_thwc": list(video.shape),
                }
            )

    if not manifest_entries:
        print("ERROR: no episodes were extracted", file=sys.stderr)
        return 1

    manifest_path = output_dir / f"{args.tag}_inference_manifest.json"
    manifest_path.write_text(json.dumps(manifest_entries, indent=2))
    provenance = {
        "script": os.path.relpath(__file__, start=os.getcwd()),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "embodiment": "jhu_dvrk_mono",
        "num_frames": args.num_frames,
        "state_t": 1 + num_actions // 4,
        "num_action_per_chunk": num_actions,
        "timestep_interval": args.timestep_interval,
        "episode_specs": episode_specs,
        "manifest_file": manifest_path.name,
    }
    (output_dir / f"{args.tag}_inference_manifest_provenance.json").write_text(
        json.dumps(provenance, indent=2)
    )
    print(f"Wrote {len(manifest_entries)} entries to {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
