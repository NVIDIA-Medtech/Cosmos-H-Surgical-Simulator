# Codebase Guide

This guide provides an overview of the key source files in Cosmos-H-Surgical-Simulator, focused on the surgical-specific components that extend the upstream [Cosmos Predict2.5](https://github.com/nvidia-cosmos/cosmos-predict2.5) pipeline.

## Upstream Base

This repository is a domain-specific fork of [Cosmos-Predict2.5](https://github.com/nvidia-cosmos/cosmos-predict2.5). The base model documentation (Text2World, Image2World, Video2World, distillation, multiview) remains available in [docs/](./).

## Key Source Files

| Component | File | Description |
|---|---|---|
| Embodiment registry | [`groot_configs.py`] | Central configuration for all surgical embodiments. Defines `EMBODIMENT_REGISTRY` (per-embodiment action keys, video keys, transforms, normalization), `OPEN_H_DATASET_SPECS` (training mixture with weights and split exclusions), and `MAX_ACTION_DIM = 44`. Entry point for adding new embodiments or datasets. |
| Embodiment tags | [`embodiment_tags.py`] | `EmbodimentTag` enum defining all supported surgical robot embodiments with docstrings describing raw action dimensions, output dimensions, and key channels. |
| Dataset & padding | [`dataset.py`] | `MixedLeRobotDataset` loads heterogeneous surgical datasets in [LeRobot](https://github.com/huggingface/lerobot) format, applies per-embodiment transforms, and zero-pads all action tensors to 44D for batched training. Implements weighted sampling via repeat factors so smaller datasets are upsampled to match their target mix ratio. |
| Action transforms | [`state_action.py`] | Contains both transform classes. `CMRVersiusRelativeActionTransform` handles clutch-aware motion (engagement tracking, motion scaling, gripper sample-and-hold, energy zeroing). `GenericRelativeActionTransform` handles all other embodiments with configurable rotation formats (quaternion xyzw/wxyz, Euler RPY, rot6d) via `ActionKeyConfig`. Also contains normalization and rotation utilities. |
| Multi-embodiment inference | [`inference_open_h.py`] | Runs action-conditioned video generation across any Open-H embodiment. Loads dataset-specific transforms and normalization from `EMBODIMENT_REGISTRY` automatically. Supports `--embodiment`, `--exclude_splits`, and multi-episode generation. |
| CMR-only inference | [`inference_cmr.py`] | Specialized inference entry point for CMR Surgical Versius. Evaluates the 4 hardcoded CMR procedures with configurable episode count and autoregressive chunk depth. |
| Experiment configs | [`exp_...gr00t.py`] | Hydra experiment definitions for three training modes: Open-H (all 9 embodiments), CMR-only, and SutureBot downstream fine-tuning. Sets `action_dim=44`, GPU/node counts, and data config overrides. |
| Data config | [`data.py`] | Hydra data registration connecting `OPEN_H_DATASET_SPECS` to training/validation dataloaders. Defines `open_h_multi_train_dataset` and `open_h_multi_val_dataset`. |
| Evaluation script | [`quant_eval.py`] | Two-phase quantitative evaluation: (1) load Cosmos checkpoint, generate videos, compute FDS (L1 + SSIM); (2) unload Cosmos, load Medical-SAM3, compute GATC and TCD on stored video pairs. Supports multi-dataset and legacy CMR-only modes. See [README_EVALUATION.md](../scripts/README_EVALUATION.md). |

[`groot_configs.py`]: ../cosmos_predict2/_src/predict2/action/datasets/gr00t_dreams/groot_configs.py
[`embodiment_tags.py`]: ../cosmos_predict2/_src/predict2/action/datasets/gr00t_dreams/data/embodiment_tags.py
[`dataset.py`]: ../cosmos_predict2/_src/predict2/action/datasets/gr00t_dreams/data/dataset.py
[`state_action.py`]: ../cosmos_predict2/_src/predict2/action/datasets/gr00t_dreams/data/transform/state_action.py
[`inference_open_h.py`]: ../cosmos_predict2/_src/predict2/action/inference/inference_open_h.py
[`inference_cmr.py`]: ../cosmos_predict2/_src/predict2/action/inference/inference_cmr.py
[`exp_...gr00t.py`]: ../cosmos_predict2/_src/predict2/action/configs/action_conditioned/experiment/exp_2B_action_conditioned_rectify_flow_gr00t.py
[`data.py`]: ../cosmos_predict2/_src/predict2/action/configs/action_conditioned/data.py
[`quant_eval.py`]: ../scripts/cosmos_h_surgical_simulator_quant_eval.py

## Data Flow

### Training

```
OPEN_H_DATASET_SPECS (groot_configs.py)
    | dataset names, mix ratios, split exclusions
    v
MixedLeRobotDataset (dataset.py)
    | per-embodiment: load LeRobot dataset
    | per-embodiment: apply transforms (state_action.py)
    | zero-pad actions to 44D
    | weighted sampling via repeat factors
    v
DataLoader --> Cosmos Diffusion Transformer
```

### Inference

```
inference_open_h.py / inference_cmr.py
    | load embodiment config from EMBODIMENT_REGISTRY
    | construct transforms via groot_configs.py
    v
WrappedLeRobotSingleDataset (dataset.py)
    | load dataset with embodiment-specific normalization
    | extract context frame + action sequence
    | zero-pad actions to 44D
    v
Cosmos checkpoint --> generate video frames
```

### Evaluation

```
Phase 1: Cosmos checkpoint loaded
    | generate videos for each episode
    | compute FDS (per-frame L1 + SSIM)
    | store video arrays in CPU memory
    | unload Cosmos checkpoint
    v
Phase 2: Medical-SAM3 loaded
    | segment surgical tools in GT and generated frames
    | compute GATC (tool visual consistency)
    | compute TCD (tool centroid displacement)
    v
Aggregated results (JSON + log report)
```

## Adding a New Embodiment

1. Add an `EmbodimentTag` entry in [`embodiment_tags.py`] with a docstring describing the action space
2. Add the embodiment to `EMBODIMENT_REGISTRY` in [`groot_configs.py`] with action keys, video keys, state keys, and normalization config
3. Add dataset entries to `OPEN_H_DATASET_SPECS` with mix ratios and any split exclusions
4. Pre-compute normalization stats and save as `meta/stats_cosmos.json` in the LeRobot dataset directory
5. Test with `inference_open_h.py --embodiment your_new_tag`

See [README_ACTION_SPACE.md](../scripts/README_ACTION_SPACE.md) for the full action vector layout and per-embodiment mapping details.
