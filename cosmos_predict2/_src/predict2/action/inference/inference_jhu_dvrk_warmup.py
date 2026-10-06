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
JHU dVRK Mono variant of the GR00T warmup inference script (Phase 0 of the
self-forcing pipeline). Generates teacher trajectory caches over the 9-dataset
JHU dVRK mono training mixture (frame-proportional weighted via
``JHU_DVRK_MONO_FINETUNE_TRAIN_DATASET_SPECS``) instead of a single dataset.

Output layout matches the upstream warmup cache convention so it is consumed
unchanged by ``ActionDatasetSFWarmup`` in Phase 1:

    <save_root>/
        latents/<vidx>.pt    # teacher denoising-trajectory latents at query_steps
        images/<vidx>.png    # first conditioning frame
        actions/<vidx>.json  # the (num_frames-1)-step, zero-padded 44D action chunk
        videos/<vidx>.mp4    # the num_frames-frame ground-truth video clip
        indices.json         # the global ordered index list used by this run
                             # (seed + strategy + len + first/last few indices)

Where ``<vidx>`` is the *virtual* MixedLeRobotDataset index (NOT the slot index
in this rank's shard). This way file names are still globally unique, the
identity check ``cache[vidx] == dataset[vidx]`` continues to work, and the
``ActionDatasetSFWarmup`` glob loader doesn't care that virtual indices are
sparse / non-contiguous after random sampling.

Sampling strategies (--sample_strategy):

  - ``random`` (default, recommended): draw ``total_samples`` virtual indices
    via ``np.random.permutation`` over the FULL virtual range. Mirrors what
    the warmup trainer's ``DistributedSampler(shuffle=True)`` would draw at
    training time, so the cache is frame-proportional across the mixture.
  - ``uniform``: ``np.linspace(0, len(dataset) - 1, total_samples)``. Evenly
    spaced; deterministic; gives every subset its frame-proportional share
    within +/-1 sample.
  - ``sequential``: legacy upstream behaviour. Walks indices [0, total_samples)
    contiguously. With MixedLeRobotDataset this samples ONLY the first subset
    in the specs list and leaves the other 8 subsets untouched. Kept for
    backward-compat / debugging; do NOT use this for production caches on
    multi-subset mixtures.

The global index list is built ONCE (seeded by --indices_seed for ``random``)
and then sharded across SLURM ranks via --start/--end, where these now slice
into the index LIST not the dataset directly. So rank N processes the indices
``all_indices[start:end]`` regardless of where those land in the virtual range.

Already-existing samples in ``<save_root>`` are skipped, so the script can be
re-run / requeued safely (each rank only generates the ones it owns and that
are not yet on disk).

Example:

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python \\
        cosmos_predict2/_src/predict2/action/inference/inference_jhu_dvrk_warmup.py \\
        --experiment cosmos_predict2p5_2B_action_conditioned_jhu_dvrk_mono_finetune_13frame_8nodes_release_oss \\
        --ckpt_path /path/to/iter_000005000/model_ema_bf16.safetensors \\
        --save_root datasets/jhu_dvrk_mono_warmup_4step_h73_tabletop \\
        --resolution 288,512 --guidance 0 --num_frames 73 --chunk_size 72 \\
        --sample_strategy random --total_samples 10000 --indices_seed 0 \\
        --start 0 --end 10000 \\
        --query_steps 0,9,18,27,34
"""

import argparse
import json
import os

import mediapy
import numpy as np
import torch
import tqdm
from loguru import logger

from cosmos_predict2._src.predict2.action.datasets.gr00t_dreams.data.dataset import MixedLeRobotDataset
from cosmos_predict2._src.predict2.action.datasets.gr00t_dreams.groot_configs import (
    JHU_DVRK_MONO_FINETUNE_TRAIN_DATASET_SPECS,
    MAX_ACTION_DIM,
)
from cosmos_predict2._src.predict2.action.inference.inference_pipeline import (
    ActionVideo2WorldInference,
)


def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments for the JHU dVRK warmup cache generator."""
    parser = argparse.ArgumentParser(description="JHU dVRK Mono Phase 0 warmup-cache generator")
    parser.add_argument("--experiment", type=str, required=True, help="Experiment config name")
    parser.add_argument("--chunk_size", type=int, default=12, help="Action chunk size (must match teacher training)")
    parser.add_argument(
        "--num_frames",
        type=int,
        default=13,
        help="Clip horizon to cache (1 context + num_frames-1 prediction frames). "
        "Defaults to 13 (state_t=4, the baseline SF cascade). Set to 73 for the "
        "reference tabletop h73 student (state_t=19). Constraint: (num_frames-1) must be "
        "divisible by 4 (valid horizons 13, 25, 49, 73, ...). Keep --chunk_size = "
        "num_frames-1 and select the matching long-horizon teacher checkpoint/experiment.",
    )
    parser.add_argument("--guidance", type=float, default=0.0, help="Classifier-free guidance scale (0 = no CFG)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="",
        help="Path to the teacher checkpoint (.safetensors, legacy .pt, or DCP dir). If empty, falls back to the experiment's load_path.",
    )
    parser.add_argument("--s3_cred", type=str, default="credentials/s3_checkpoint.secret")
    parser.add_argument(
        "--resolution",
        type=str,
        default="288,512",
        help="Resolution of the rendered video, format H,W. Default 288,512 matches the JHU dVRK teacher training.",
    )
    parser.add_argument(
        "--save_root",
        type=str,
        default="datasets/jhu_dvrk_mono_warmup_4step",
        help="Output directory (relative to repo root). Sub-dirs latents/ images/ actions/ videos/ are created.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help=(
            "Start position (inclusive) for this rank's shard. Slices into the "
            "GLOBAL ORDERED INDEX LIST built per --sample_strategy, NOT into the "
            "dataset directly. So rank N processes all_indices[start:end]."
        ),
    )
    parser.add_argument(
        "--end",
        type=int,
        default=10000,
        help="End position (exclusive) for this rank's shard. See --start.",
    )
    parser.add_argument(
        "--total_samples",
        type=int,
        default=10000,
        help=(
            "Size of the GLOBAL index list (the cache target size across all ranks). "
            "Each rank slices [start:end] of this list. Default 10000."
        ),
    )
    parser.add_argument(
        "--sample_strategy",
        type=str,
        default="random",
        choices=["sequential", "uniform", "random"],
        help=(
            "How to draw the global index list. 'random' (default): permutation over "
            "the full virtual range, seeded by --indices_seed. Mirrors the warmup "
            "trainer's DistributedSampler(shuffle=True). 'uniform': np.linspace. "
            "'sequential': legacy upstream pattern; on MixedLeRobotDataset this "
            "draws ONLY the first subset and is generally not what you want."
        ),
    )
    parser.add_argument(
        "--indices_seed",
        type=int,
        default=0,
        help=(
            "Seed for the --sample_strategy=random permutation. All ranks must use "
            "the same seed so they slice the same global ordered list. Default 0."
        ),
    )
    parser.add_argument(
        "--num_latent_conditional_frames",
        type=int,
        default=1,
        help="Number of latent conditional frames (warmup uses 1 = single image conditioning).",
    )
    parser.add_argument(
        "--query_steps",
        type=lambda x: [int(i) for i in x.split(",")],
        default=[0, 9, 18, 27, 34],
        help="Denoising-step indices at which to snapshot the teacher's latent trajectory.",
    )
    parser.add_argument(
        "--context_parallel_size",
        type=int,
        default=1,
        help="Context parallel size (default 1 = no CP). Set to 8 if launching on 8 GPUs jointly.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        default=True,
        help="Skip indices whose cache files already exist on disk. Default True (safe re-run / requeue).",
    )
    parser.add_argument(
        "--no_skip_existing",
        action="store_false",
        dest="skip_existing",
        help="Force regenerate even if cache files already exist.",
    )
    return parser.parse_args()


def _is_already_cached(save_root: str, idx: int) -> bool:
    """Return True iff all 4 cache artefacts for ``idx`` already exist on disk."""
    return (
        os.path.exists(os.path.join(save_root, "latents", f"{idx}.pt"))
        and os.path.exists(os.path.join(save_root, "images", f"{idx}.png"))
        and os.path.exists(os.path.join(save_root, "actions", f"{idx}.json"))
        and os.path.exists(os.path.join(save_root, "videos", f"{idx}.mp4"))
    )


def build_global_index_list(
    dataset_len: int,
    total_samples: int,
    strategy: str,
    seed: int,
) -> np.ndarray:
    """Construct the global ordered list of virtual indices to cache.

    All ranks call this with identical arguments and slice [start:end] of the
    returned array. So the deterministic-by-seed property of ``random`` (and
    the inherent determinism of ``uniform`` / ``sequential``) gives every
    rank the same global view, just a different shard of it.
    """
    n = min(total_samples, dataset_len)
    if strategy == "sequential":
        return np.arange(n, dtype=np.int64)
    if strategy == "uniform":
        # endpoint=True: linspace covers [0, dataset_len - 1] inclusive.
        return np.linspace(0, dataset_len - 1, n, dtype=np.int64)
    if strategy == "random":
        rng = np.random.default_rng(seed)
        # Note: np.random.default_rng(seed) is a fresh stream per call but
        # deterministic by seed. So all ranks get the same permutation.
        perm = rng.permutation(dataset_len)
        return perm[:n].astype(np.int64)
    raise ValueError(f"Unknown sample_strategy={strategy!r}")


def _save_indices_manifest(
    save_root: str,
    all_indices: np.ndarray,
    args: argparse.Namespace,
    dataset_len: int,
) -> None:
    """Persist the global ordered index list (and how it was built) to disk.

    Writes ``<save_root>/indices.json``. The full list is written each rank-0
    write but with a small head/tail preview for human inspection of the
    first few writers; rank > 0 will overwrite with identical content (since
    the list is deterministic-by-seed)."""
    manifest = {
        "sample_strategy": args.sample_strategy,
        "indices_seed": args.indices_seed,
        "total_samples": int(args.total_samples),
        "dataset_len": int(dataset_len),
        "n_indices": int(len(all_indices)),
        "first_10": [int(x) for x in all_indices[:10]],
        "last_10": [int(x) for x in all_indices[-10:]],
        "min": int(all_indices.min()) if len(all_indices) else None,
        "max": int(all_indices.max()) if len(all_indices) else None,
        "all_indices": [int(x) for x in all_indices],
    }
    out_path = os.path.join(save_root, "indices.json")
    try:
        with open(out_path, "w") as f:
            json.dump(manifest, f, indent=2)
    except Exception as exc:
        # Non-fatal: caching itself is what matters; log and continue.
        logger.warning(f"Failed to write {out_path}: {exc}")


def main():
    torch.enable_grad(False)
    args = parse_arguments()

    # Reproducibility for the dataset's internal sample shuffling.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Build the JHU dVRK Mono training mixture (9 datasets, frame-proportional
    # weighting via mix_ratio = total_frames per subset). Using data_split="train"
    # ensures we cache from the same partition the teacher saw during fine-tuning,
    # honoring per-spec test_split_ratio_override (0%-2%) and the ood spec's
    # data_split_override="full".
    # The VAE temporal_compression_ratio=4 collapses 4 pixel frames into 1
    # latent frame, so (num_frames-1) must be divisible by 4. The teacher's
    # state_t MUST match this clip horizon (state_t = 1 + (num_frames-1)//4),
    # e.g. the reference tabletop h73 teacher uses state_t=19.
    if (args.num_frames - 1) % 4 != 0:
        raise ValueError(
            f"--num_frames must satisfy (num_frames-1) % 4 == 0 "
            f"(got {args.num_frames}); valid horizons are 13, 25, 49, 73, ..."
        )
    dataset = MixedLeRobotDataset(
        dataset_specs=JHU_DVRK_MONO_FINETUNE_TRAIN_DATASET_SPECS,
        num_frames=args.num_frames,
        data_split="train",
        max_action_dim=MAX_ACTION_DIM,
        downscaled_res=False,
        test_split_ratio=0.02,
    )
    logger.info(
        f"JHU dVRK Mono cache horizon: num_frames={args.num_frames} "
        f"(state_t={1 + (args.num_frames - 1) // 4}), chunk_size={args.chunk_size}"
    )
    logger.info(
        f"MixedLeRobotDataset built: virtual size = {len(dataset)}, "
        f"max_action_dim = {MAX_ACTION_DIM}, "
        f"requesting indices [{args.start}, {args.end})"
    )

    # Initialize the inference handler with context parallel support
    video2world_cli = ActionVideo2WorldInference(
        args.experiment,
        args.ckpt_path,
        args.s3_cred,
        context_parallel_size=args.context_parallel_size,
    )

    mem_bytes = torch.cuda.memory_allocated(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info(f"GPU memory after model dcp.load: {mem_bytes / (1024**3):.2f} GB")

    save_root = args.save_root
    os.makedirs(os.path.join(save_root, "latents"), exist_ok=True)
    os.makedirs(os.path.join(save_root, "images"), exist_ok=True)
    os.makedirs(os.path.join(save_root, "actions"), exist_ok=True)
    os.makedirs(os.path.join(save_root, "videos"), exist_ok=True)

    # Build the GLOBAL ordered index list once per rank (all ranks build the
    # same list because all inputs are deterministic / seed-controlled). Then
    # this rank slices [start:end] of that list. Cache files are named by the
    # actual VIRTUAL index (not the slot in the shard) so file names remain
    # globally unique across ranks and the identity check still works.
    all_indices = build_global_index_list(
        dataset_len=len(dataset),
        total_samples=args.total_samples,
        strategy=args.sample_strategy,
        seed=args.indices_seed,
    )
    logger.info(
        f"Global index list built: strategy={args.sample_strategy}, "
        f"seed={args.indices_seed}, n={len(all_indices)}, "
        f"min={int(all_indices.min())}, max={int(all_indices.max())}, "
        f"head={all_indices[:5].tolist()}, tail={all_indices[-5:].tolist()}"
    )
    _save_indices_manifest(save_root, all_indices, args, len(dataset))

    end = min(args.end, len(all_indices))
    start = max(0, min(args.start, end))
    rank_indices = all_indices[start:end].tolist()
    logger.info(
        f"This rank: processing slot range [{start}, {end}) "
        f"({len(rank_indices)} virtual indices, "
        f"first={rank_indices[0] if rank_indices else 'n/a'}, "
        f"last={rank_indices[-1] if rank_indices else 'n/a'})"
    )

    n_skipped = 0
    n_generated = 0
    for vidx in tqdm.tqdm(rank_indices, desc=f"Phase 0 cache slots [{start},{end})"):
        if args.skip_existing and _is_already_cached(save_root, vidx):
            n_skipped += 1
            continue

        data = dataset[vidx]
        # data["video"] is shape (C, T, H, W) in uint8. Match the upstream warmup
        # convention: feed the FIRST frame as the conditioning image; the action
        # chunk and the num_frames-frame ground-truth video are saved alongside.
        img_np_array = data["video"][:, 0, :, :].permute(1, 2, 0).cpu().numpy()
        video_np_array = data["video"].permute(1, 2, 3, 0).cpu().numpy()
        action = data["action"].cpu().numpy()

        next_img_array, video_clamped, latents_to_save = video2world_cli.step_inference_with_latents(
            img_array=img_np_array,
            action=action,
            guidance=args.guidance,
            seed=args.seed,
            num_latent_conditional_frames=args.num_latent_conditional_frames,
            query_steps=args.query_steps,
        )

        for k in latents_to_save:
            latents_to_save[k] = latents_to_save[k].squeeze(0).cpu()

        torch.save(latents_to_save, os.path.join(save_root, "latents", f"{vidx}.pt"))
        mediapy.write_image(os.path.join(save_root, "images", f"{vidx}.png"), img_np_array)
        mediapy.write_video(os.path.join(save_root, "videos", f"{vidx}.mp4"), video_np_array)
        with open(os.path.join(save_root, "actions", f"{vidx}.json"), "w") as f:
            json.dump(action.tolist(), f, indent=4)
        n_generated += 1

    logger.info(
        f"Phase 0 cache shard slots [{start},{end}) complete: "
        f"generated {n_generated}, skipped (already cached) {n_skipped}"
    )


if __name__ == "__main__":
    main()
