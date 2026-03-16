# Surgical Inference Guide

This guide covers action-conditioned surgical video generation using the Open-H multi-embodiment pipeline and CMR-only mode.

For base model inference (Text2World, Image2World, Video2World), see the [Inference Guide](inference.md).
For the generic action-conditioned pipeline (Bridge dataset), see the [Action-Conditioned Inference Guide](inference_robot_action_cond.md).

## Prerequisites

1. [Setup Guide](setup.md) — environment and dependencies
2. [HuggingFace authentication](setup.md#downloading-checkpoints) — required for checkpoint downloads
3. A trained checkpoint (`model_ema_bf16.pt`) — see [Surgical Post-Training](post-training_surgical.md) or download from [HuggingFace](https://huggingface.co/nvidia/Cosmos-H-Surgical-Simulator)

## Multi-Embodiment Inference (Open-H)

Run action-conditioned video generation on any supported embodiment using `inference_open_h.py`. Each dataset is loaded with its embodiment-specific transforms via `WrappedLeRobotSingleDataset`, supporting heterogeneous action spaces, timestep intervals, and normalization. Action vectors are automatically zero-padded to 44D.

### CMR Versius

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python cosmos_predict2/_src/predict2/action/inference/inference_open_h.py \
  --experiment cosmos_predict2p5_2B_action_conditioned_open_h-fixed_13frame_8nodes_release_oss \
  --ckpt_path /path/to/model_ema_bf16.pt \
  --dataset_path /CMR_Versius/cholecystectomy_480p \
  --embodiment cmr_versius \
  --episode_ids 0,1,2
```

### dVRK JHU (Monocular)

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python cosmos_predict2/_src/predict2/action/inference/inference_open_h.py \
  --experiment cosmos_predict2p5_2B_action_conditioned_open_h-fixed_13frame_8nodes_release_oss \
  --ckpt_path /path/to/model_ema_bf16.pt \
  --dataset_path /path/to/suturebot_2 \
  --embodiment jhu_dvrk_mono \
  --episode_ids 0,1,2
```

### Stanford Real (with Split Exclusions)

Some datasets contain splits that should be excluded during evaluation (e.g., failed trials, corrupted frames):

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python cosmos_predict2/_src/predict2/action/inference/inference_open_h.py \
  --experiment cosmos_predict2p5_2B_action_conditioned_open_h-fixed_13frame_8nodes_release_oss \
  --ckpt_path /path/to/model_ema_bf16.pt \
  --dataset_path /path/to/Needle_Transfer \
  --embodiment dvrk_stanford_real \
  --exclude_splits fail bad_frames \
  --episode_ids 0,1,2
```

### Key Arguments

| Argument | Description |
|---|---|
| `--experiment` | Experiment config name (use the Open-H config shown above) |
| `--ckpt_path` | Path to converted checkpoint (`model_ema_bf16.pt`) |
| `--dataset_path` | Path to the LeRobot dataset directory |
| `--embodiment` | Embodiment tag (see [Supported Embodiments](../README.md#supported-embodiments)) |
| `--exclude_splits` | Dataset splits to exclude (e.g., `fail`, `bad_frames`) |
| `--episode_ids` | Comma-separated episode indices to generate |

### Per-Embodiment Handling

The inference pipeline automatically adapts to each embodiment:

| Aspect | How it adapts |
|---|---|
| Timestep interval | Resolved from `EMBODIMENT_REGISTRY` (e.g., 6 for CMR, 5 for dVRK JHU, 3 for Stanford) |
| Action transform | `GenericRelativeActionTransform` or `CMRVersiusRelativeActionTransform` |
| Action padding | Zero-padded to 44D to match model input dimension |
| Split filtering | `exclude_splits` looked up from `OPEN_H_DATASET_SPECS` |

## CMR-Only Inference

For models trained exclusively on CMR Surgical Versius data, use `inference_cmr.py`:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python cosmos_predict2/_src/predict2/action/inference/inference_cmr.py \
  --experiment cosmos_predict2p5_2B_action_conditioned_cmr_13frame_44D_8nodes_release_oss \
  --ckpt_path /path/to/model_ema_bf16.pt \
  --dataset_path /CMR_Versius/cholecystectomy_480p \
  --save_root results/cmr_eval/cholecystectomy \
  --data_split test \
  --episode_ids 0,1,2
```

This mode evaluates the 4 hardcoded CMR Versius procedures (prostatectomy, inguinal hernia, hysterectomy, cholecystectomy) using `LeRobotDataset` with up to 5 episodes per dataset and 6 autoregressive chunks per episode.

## See Also

- [Action Space Design](../scripts/README_ACTION_SPACE.md) — unified 44D action vector and per-embodiment mapping
- [Evaluation Pipeline](../scripts/README_EVALUATION.md) — quantitative evaluation with FDS, GATC, TCD metrics
- [Codebase Guide](codebase-guide.md) — key source files and architecture overview
