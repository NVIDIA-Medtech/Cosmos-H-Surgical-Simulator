#!/usr/bin/env python3
"""Print per-dataset episode lists for the 5% test split of the Open-H mixture.

For each sub-dataset in OPEN_H_DATASET_SPECS, this script constructs the test
split (last 5% of step indices) and extracts the unique episode IDs that fall
within that split. The output is both a human-readable table and a JSON file
that downstream inference scripts can consume to select specific episodes.

Usage (inside the container or via cpu_batch_template):
    python scripts/print_test_datasets_and_episodes.py [--output test_episodes.json]

The JSON output has the structure:
    {
        "<dataset_name>": {
            "embodiment": "<tag>",
            "path": "<dataset_path>",
            "num_test_steps": <int>,
            "num_test_episodes": <int>,
            "episode_ids": [<int>, ...]
        },
        ...
    }
"""

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path

from cosmos_predict2._src.predict2.action.datasets.gr00t_dreams.groot_configs import (
    OPEN_H_DATASET_SPECS,
    construct_modality_config_and_transforms,
)
from cosmos_predict2._src.predict2.action.datasets.gr00t_dreams.data.dataset import (
    WrappedLeRobotSingleDataset,
)
from cosmos_predict2._src.predict2.action.datasets.gr00t_dreams.data.embodiment_tags import (
    EmbodimentTag,
)


NUM_FRAMES = 13  # 1 context + 12 prediction (must match training config)


def get_test_episodes(spec: dict) -> dict:
    """Load a single dataset in test mode and return its episode metadata.

    Args:
        spec: A single entry from OPEN_H_DATASET_SPECS with keys
              ``path``, ``embodiment``, ``mix_ratio``, and optionally
              ``exclude_splits`` and ``modality_filename``.

    Returns:
        Dict with dataset name, embodiment, path, test step/episode counts,
        and a sorted list of unique episode IDs in the test split.
    """
    path = spec["path"]
    raw_embodiment = spec["embodiment"]
    embodiment = raw_embodiment.value if isinstance(raw_embodiment, EmbodimentTag) else raw_embodiment
    exclude_splits = spec.get("exclude_splits", None)

    config, _, test_transform = construct_modality_config_and_transforms(
        num_frames=NUM_FRAMES, embodiment=embodiment, downscaled_res=False,
    )

    modality_filename = None
    if isinstance(config, dict) and "modality_filename" in config:
        modality_filename = config.pop("modality_filename")

    dataset = WrappedLeRobotSingleDataset(
        dataset_path=path,
        modality_configs=config,
        transforms=test_transform,
        embodiment_tag=embodiment,
        data_split="test",
        modality_filename=modality_filename,
        exclude_splits=exclude_splits,
    )

    # _all_steps is a list of (episode_id, step_index) tuples after the 5% test slice
    test_steps = dataset.all_steps
    unique_episodes = sorted(set(int(ep_id) for ep_id, _ in test_steps))
    dataset_name = Path(path).name

    return {
        "dataset_name": dataset_name,
        "embodiment": embodiment,
        "path": path,
        "num_test_steps": len(test_steps),
        "num_test_episodes": len(unique_episodes),
        "episode_ids": unique_episodes,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Print test-split episode lists for all Open-H datasets."
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Path to write JSON output. If omitted, prints to stdout only.",
    )
    args = parser.parse_args()

    results: OrderedDict[str, dict] = OrderedDict()
    total_test_steps = 0
    total_test_episodes = 0

    print("=" * 90)
    print("OPEN-H TEST SPLIT — EPISODE LISTING")
    print("=" * 90)
    print(f"{'Dataset':<40} {'Embodiment':<22} {'Test Steps':>12} {'Test Episodes':>14}")
    print("-" * 90)

    for i, spec in enumerate(OPEN_H_DATASET_SPECS):
        dataset_name = Path(spec["path"]).name
        try:
            info = get_test_episodes(spec)
        except Exception as e:
            print(f"[{i}] ERROR loading {dataset_name}: {e}", file=sys.stderr)
            continue

        results[info["dataset_name"]] = {
            "embodiment": info["embodiment"],
            "path": info["path"],
            "num_test_steps": info["num_test_steps"],
            "num_test_episodes": info["num_test_episodes"],
            "episode_ids": info["episode_ids"],
        }

        total_test_steps += info["num_test_steps"]
        total_test_episodes += info["num_test_episodes"]

        print(
            f"{info['dataset_name']:<40} {info['embodiment']:<22} "
            f"{info['num_test_steps']:>12,} {info['num_test_episodes']:>14,}"
        )
        # Compact episode list (truncated for readability)
        ep_list = info["episode_ids"]
        if len(ep_list) <= 10:
            ep_str = str(ep_list)
        else:
            ep_str = f"[{ep_list[0]}, {ep_list[1]}, ..., {ep_list[-2]}, {ep_list[-1]}]  ({len(ep_list)} total)"
        print(f"  episodes: {ep_str}")

    print("-" * 90)
    print(
        f"{'TOTAL':<40} {'':<22} {total_test_steps:>12,} {total_test_episodes:>14,}"
    )
    print("=" * 90)

    # Write JSON output
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nJSON written to: {output_path}")
    else:
        print("\nJSON output (use --output <path> to save to file):")
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
