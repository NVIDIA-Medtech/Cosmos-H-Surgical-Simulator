# Surgical Post-Training Guide

This guide covers post-training Cosmos-H-Surgical-Simulator on surgical data from the Open-H benchmark.

For general post-training concepts (checkpointing, W&B logging, HuggingFace setup), see the [Post-Training Guide](post-training.md).
For the generic action-conditioned pipeline (Bridge dataset), see the [Action-Conditioned Post-Training Guide](post-training_video2world_action.md).

## Prerequisites

1. [Setup Guide](setup.md) — environment and dependencies
2. [HuggingFace authentication](setup.md#downloading-checkpoints) — required for checkpoint downloads
3. Configure training output directory:

```bash
export IMAGINAIRE_OUTPUT_ROOT=/path/to/output
```

## Open-H Multi-Embodiment (Recommended)

Train across all 9 embodiments with the unified 44D action space. The data mixture allocates 50% of training compute to CMR Versius (4 surgical procedures) and 50% to all other embodiments, step-weighted by frame count.

```bash
torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train \
  --config=cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py \
  -- experiment=cosmos_predict2p5_2B_action_conditioned_open_h-fixed_13frame_8nodes_release_oss \
  ~dataloader_train.dataloaders
```

Training configuration: 8 nodes / 64 GPUs, batch size 16 per GPU (global 1024), 512x288 resolution, 13 video frames (1 context + 12 prediction), 12 action timesteps per sample.

## CMR Versius Only

Train on CMR Surgical Versius data exclusively (44D action space):

```bash
torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train \
  --config=cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py \
  -- experiment=cosmos_predict2p5_2B_action_conditioned_cmr_13frame_44D_8nodes_release_oss \
  ~dataloader_train.dataloaders
```

## Downstream Fine-Tuning

The multi-embodiment Open-H model serves as a foundation for further specialization on individual datasets. For example, fine-tuning on SutureBot (dVRK JHU). The 20D SutureBot actions are zero-padded to 44D automatically:

```bash
torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train \
  --config=cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py \
  -- experiment=cosmos_predict2p5_2B_action_conditioned_suturebot_13frame_4nodes_release_oss \
  ~dataloader_train.dataloaders
```

## Checkpoint Conversion

Training saves checkpoints in DCP (Distributed Checkpoint) format. Convert to consolidated PyTorch `.pt` format for inference:

```bash
CHECKPOINTS_DIR=${IMAGINAIRE_OUTPUT_ROOT}/cosmos_predict2_action_conditioned/.../checkpoints
CHECKPOINT_ITER=$(cat $CHECKPOINTS_DIR/latest_checkpoint.txt)

python ./scripts/convert_distcp_to_pt.py $CHECKPOINTS_DIR/$CHECKPOINT_ITER/model $CHECKPOINTS_DIR/$CHECKPOINT_ITER
```

This produces three files:

- `model.pt` — full checkpoint with regular and EMA weights
- `model_ema_fp32.pt` — EMA weights in float32
- `model_ema_bf16.pt` — EMA weights in bfloat16 (recommended for inference)

## See Also

- [Surgical Inference Guide](inference_surgical.md) — running inference with trained checkpoints
- [Action Space Design](../scripts/README_ACTION_SPACE.md) — understanding the 44D action vector
- [Codebase Guide](codebase-guide.md) — key source files and data flow
