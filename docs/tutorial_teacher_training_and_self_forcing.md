# From a Cosmos-H-Surgical-Simulator Teacher to a Real-Time Causal Student

This recipe describes how to adapt the bidirectional Cosmos-H-Surgical-Simulator (C-H-S-S) model to a new paired kinematics-video dataset and then distill it into a causal, streaming student with [Self Forcing](https://arxiv.org/abs/2506.08009). The resulting student can be deployed with [Cosmos-H-Dreams](https://github.com/isaac-for-healthcare/Cosmos-H-Dreams) for low-latency, real-time closed-loop generation.

The reference run used the SutureBot tabletop dataset, which is part of Open-H-Embodiment, to train a 73-frame causal student that ran in real time with the Cosmos-H-Dreams library. The main steps to follow are:

1. Fine-tune a short-horizon, bidirectional teacher with 13-frame samples from C-H-S-S.
2. Fine-tune a long-horizon, bidirectional teacher with 73-frame samples from the short-horizon checkpoint.
3. Generate a cache of teacher denoising trajectories (Phase 0).
4. Warm up a causal student against the cached trajectories.
5. Run Self Forcing distillation.
6. Convert and deploy the causal student with Cosmos-H-Dreams.

This tutorial first explains what you will train, then provides a reference recipe you can adapt to your dataset.

## Table of contents

- [1. What is being trained?](#1-what-is-being-trained)
- [2. Reference recipe](#2-reference-recipe)
- [3. Prerequisites](#3-prerequisites)
- [4. Prepare a custom action-conditioned dataset](#4-prepare-a-custom-action-conditioned-dataset)
- [5. Stage 1: short-horizon bidirectional teacher](#5-stage-1-short-horizon-bidirectional-teacher)
- [6. Stage 2: long-horizon bidirectional teacher](#6-stage-2-long-horizon-bidirectional-teacher)
- [7. Stage 3: Phase 0 teacher-trajectory cache](#7-stage-3-phase-0-teacher-trajectory-cache)
- [8. Stage 4: causal-student warmup](#8-stage-4-causal-student-warmup)
- [9. Stage 5: Self Forcing distillation](#9-stage-5-self-forcing-distillation)
- [10. Convert and validate the distilled student](#10-convert-and-validate-the-distilled-student)
- [11. Deploy with Cosmos-H-Dreams](#11-deploy-with-cosmos-h-dreams)
- [12. Common failure modes](#12-common-failure-modes)
- [13. Further reading and resources](#13-further-reading-and-resources)

## 1. What is being trained?

Each stage serves a different purpose:


| Stage         | Network                                              | Attention                    | Training signal                                | Output                             | Initialization / checkpoint source                   |
| ------------- | ---------------------------------------------------- | ---------------------------- | ---------------------------------------------- | ---------------------------------- | ---------------------------------------------------- |
| Short teacher | `cosmos_v1_2B_action_chunk_conditioned`              | Bidirectional                | Rectified-flow teacher objective on real clips | Domain-adapted 13-frame teacher    | C-H-S-S                                              |
| Long teacher  | Same teacher network                                 | Bidirectional                | Same objective on longer clips                 | 73-frame teacher                   | Short teacher                                        |
| Phase 0       | Long teacher, frozen                                 | Bidirectional                | Inference only                                 | Cached teacher latent trajectories | Long teacher                                         |
| Warmup        | `action_causal_cosmos_v1_2B`                         | Causal                       | Regression to cached teacher trajectories      | Initialized causal student         | Short-teacher; Phase 0 supplies long teacher targets |
| Self Forcing  | Causal student + frozen teacher + fake-score network | Causal student rollout       | DMD/fake-score/adversarial objectives          | Streaming causal student           | Warmup student plus frozen long teacher              |
| Deployment    | Causal student, frozen                               | Causal with rolling KV cache | Inference only                                 | Real-time generated video          | Self Forcing student                                 |


The key design decision is to learn the target domain first with a bidirectional model. Causal distillation is then performed only after the teacher produces useful long-horizon rollouts.

## 2. Reference recipe

The reference tabletop run used the following values. Treat them as a validated starting point, not universal constants.


| Quantity                          | Short teacher                          | Long teacher                  | Phase 0                  | Warmup                        | Self Forcing                          |
| --------------------------------- | -------------------------------------- | ----------------------------- | ------------------------ | ----------------------------- | ------------------------------------- |
| Pixel frames                      | 13                                     | 73                            | 73                       | 73                            | 73                                    |
| Predicted/action frames           | 12                                     | 72                            | 72                       | 72                            | 72                                    |
| Latent temporal length, `state_t` | 4                                      | 19                            | 19                       | 19                            | 19                                    |
| Per-GPU batch, 8 nodes × 8 GPUs   | 16                                     | 4                             | N/A                      | 2                             | 1                                     |
| Global batch                      | 1024                                   | 256                           | N/A                      | 128                           | 64                                    |
| Main learning rate                | `1.6e-4`                               | `4e-5`                        | N/A                      | `3e-5`                        | `5e-8`                                |
| Iteration budget                  | selected at 16k, then 4k cosine annealing | 5k                            | 10k cached samples       | 20k ceiling; 18k selected     | 3k                                    |
| LR schedule                       | base training, then cosine annealing    | 1k warmup + cosine            | N/A                      | Constant                      | Constant                              |
| Resolution                        | 288 × 512                              | 288 × 512                     | 288 × 512                | 288 × 512                     | 288 × 512                             |
| Model-facing action width         | 44                                     | 44                            | 44                       | 44                            | 44                                    |
| Stage warm-start/input            | C-H-S-S DCP                            | annealed h13 teacher, iter 4k | h73 teacher EMA, iter 5k | annealed h13 teacher, iter 4k | warmup iter 18k + h73 teacher iter 5k |




### 2.1 Reference implementation

This repository contains the complete reference tabletop implementation used to obtain the released [checkpoint on Hugging Face](https://huggingface.co/nvidia/Cosmos-H-Dreams). Use these files as the concrete reference while adapting the `my_robot_*` examples below:

- dataset mixture: [`JHU_DVRK_MONO_FINETUNE_*_DATASET_SPECS`](../cosmos_predict2/_src/predict2/action/datasets/gr00t_dreams/groot_configs.py);
- 13- and 73-frame loaders: [`action_conditioned/data.py`](../cosmos_predict2/_src/predict2/action/configs/action_conditioned/data.py);
- short teacher, cosine-annealing stage, and long teacher: [`exp_2B_action_conditioned_rectify_flow_gr00t.py`](../cosmos_predict2/_src/predict2/action/configs/action_conditioned/experiment/exp_2B_action_conditioned_rectify_flow_gr00t.py);
- Phase 0 cache generator: [`inference_jhu_dvrk_warmup.py`](../cosmos_predict2/_src/predict2/action/inference/inference_jhu_dvrk_warmup.py);
- cache loader: [`interactive/configs/data.py`](../cosmos_predict2/_src/predict2/interactive/configs/data.py);
- causal warmup: [`exp_action_warmup.py`](../cosmos_predict2/_src/predict2/interactive/configs/experiment/exp_action_warmup.py);
- Self Forcing: [`exp_action_self_forcing.py`](../cosmos_predict2/_src/predict2/interactive/configs/experiment/exp_action_self_forcing.py);
- launch templates and environment contract: [`train_scripts/tabletop/`](../train_scripts/tabletop/README.md).

The configs read the environment variables `JHU_TABLETOP_DATA_ROOT`, `CHSS_CHECKPOINT_DIR`, and `IMAGINAIRE_OUTPUT_ROOT`.

Cosmos-Predict2.5's video tokenizer has a temporal compression ratio of four. The pixel-frame and latent-frame relationship used throughout this recipe is:

```text
num_frames = 1 + num_actions
state_t    = 1 + num_actions // 4
```

Therefore:

```text
13 frames = 1 context + 12 predicted frames -> state_t = 4
73 frames = 1 context + 72 predicted frames -> state_t = 19
```

Valid horizons satisfy:

```python
assert (num_frames - 1) % 4 == 0
```

Examples are 13, 25, 49, and 73 frames.

## 3. Prerequisites

Follow [setup.md](setup.md), authenticate with Hugging Face, and download the C-H-S-S checkpoint.

```bash
git clone https://github.com/NVIDIA-Medtech/Cosmos-H-Surgical-Simulator.git
cd Cosmos-H-Surgical-Simulator
git lfs pull

curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
uv sync --extra=cu128
source .venv/bin/activate

uv tool install -U "huggingface_hub[cli]"
hf auth login

export IMAGINAIRE_OUTPUT_ROOT=/persistent/path/imaginaire/output
export IMAGINAIRE_CACHE_DIR=/persistent/path/imaginaire/cache
mkdir -p "$IMAGINAIRE_OUTPUT_ROOT" "$IMAGINAIRE_CACHE_DIR"
```

The complete workflow is GPU-heavy. The reference run used 8 nodes, each with eight 80 GB GPUs. CPU-only validation is still useful for syntax, config registration, dataset metadata, and cache-integrity checks, but it does not validate model memory or throughput.

### 3.1 Always use persistent output storage

Set `IMAGINAIRE_OUTPUT_ROOT` *inside the training container or process*, not just in the login shell. Otherwise, the experiment directory may be created in the container's overlay storage. A checkpoint save may appear to succeed but disappear when the container exits.

For an Enroot/Pyxis-style launch:

```bash
OUTPUT_DIR=/persistent/path/imaginaire/output
mkdir -p "$OUTPUT_DIR"
export OUTPUT_DIR

srun \
  --container-mounts="${OUTPUT_DIR}:${OUTPUT_DIR},${PWD}:/workspace" \
  --container-workdir=/workspace \
  bash -c '
    export IMAGINAIRE_OUTPUT_ROOT="$OUTPUT_DIR"
    python -m scripts.train ...
  '
```

Verify persistence after the first save:

```bash
test -f \
  "$IMAGINAIRE_OUTPUT_ROOT/<project>/<group>/<run>/checkpoints/latest_checkpoint.txt"
```



## 4. Prepare a custom action-conditioned dataset

The existing pipeline expects LeRobot-format datasets. Before training, each dataset needs:

- video referenced by `meta/modality.json`;
- robot state at the context/reference timestep;
- per-timestep robot actions;
- `meta/info.json`, episode metadata, and parquet data in the normal LeRobot layout;
- post-transform normalization statistics;
- an embodiment entry describing action keys, state keys, transforms, frame stride, and target video resolution.

Read [README_ACTION_SPACE.md](../scripts/README_ACTION_SPACE.md) before defining a new action space.

### 4.1 Define the model-facing action representation

C-H-S-S uses a unified 44D action input. For non-CMR embodiments, actions are transformed into their native model-facing representation and then zero-padded to 44D. For a dual-arm dVRK:

```text
PSM1 relative xyz + rot6d  9D
PSM1 absolute gripper      1D
PSM2 relative xyz + rot6d  9D
PSM2 absolute gripper      1D
                              = 20D native
20D native + 24 trailing zeros = 44D model input
```

Do not pad raw parquet columns by hand if you use `MixedLeRobotDataset`; it pads the transformed action tensor. Make sure deployment uses the same transform, normalization, concatenation order, and padding convention as training.

For an existing embodiment, reuse its registry entry. For a new dual-arm embodiment, add a registry entry in `cosmos_predict2/_src/predict2/action/datasets/gr00t_dreams/groot_configs.py`:

```python
EMBODIMENT_REGISTRY["my_dual_arm_robot"] = {
    "timestep_interval": 3,  # 30 Hz raw -> 10 Hz model rate
    "video_keys": ["video.endoscope_left"],
    "state_keys": [
        "state.psm1_pose",
        "state.psm1_gripper",
        "state.psm2_pose",
        "state.psm2_gripper",
    ],
    "action_keys": [
        "action.psm1_pose",
        "action.psm1_gripper",
        "action.psm2_pose",
        "action.psm2_gripper",
    ],
    "action_key_configs": _dual_arm_eef_configs(
        "action.psm1_pose",
        "action.psm1_gripper",
        "action.psm2_pose",
        "action.psm2_gripper",
        "state.psm1_pose",
        "state.psm2_pose",
        input_rot="quat",
        ref_rot="quat",
        input_quat="xyzw",
        ref_quat="xyzw",
    ),
    "video_width": 512,
    "video_height": 288,
    "modality_filename": "meta/modality.json",
    "normalization_mode": "mean_std",
}
```

Also add an `EmbodimentTag` value in
`cosmos_predict2/_src/predict2/action/datasets/gr00t_dreams/data/embodiment_tags.py`.

### 4.2 Create the dataset mixture

Define one spec per dataset. Setting `mix_ratio` in proportion to frame count makes sampling roughly frame-proportional:

The reference nine-subset configuration is defined in
[`groot_configs.py`](../cosmos_predict2/_src/predict2/action/datasets/gr00t_dreams/groot_configs.py),
with all paths resolved relative to `JHU_TABLETOP_DATA_ROOT`.

```python
MY_TRAIN_SPECS = [
    {
        "path": "/datasets/my_robot/success",
        "embodiment": EmbodimentTag.MY_DUAL_ARM_ROBOT,
        "mix_ratio": 500_000.0,
        "test_split_ratio_override": 0.01,
    },
    {
        "path": "/datasets/my_robot/failures",
        "embodiment": EmbodimentTag.MY_DUAL_ARM_ROBOT,
        "mix_ratio": 100_000.0,
        "test_split_ratio_override": 0.02,
    },
    {
        "path": "/datasets/my_robot/ood",
        "embodiment": EmbodimentTag.MY_DUAL_ARM_ROBOT,
        "mix_ratio": 50_000.0,
        "data_split_override": "full",
    },
]

MY_VAL_SPECS = MY_TRAIN_SPECS[:2]
```

Equal `mix_ratio=1.0` values weight subsets equally, not individual samples. This can oversample a small failure set hundreds of times, so use equal ratios only when that is your intent.

As noted in [Cosmos-Surg-dVRK](https://arxiv.org/abs/2510.16240), failure episodes are essential when training a simulator. A simulator trained only on successful episodes will be biased toward successful outcomes and may not handle failures well. The reference tabletop run included both failure episodes and out-of-distribution (OOD) data consisting of random trajectories.

### 4.3 Generate post-transform statistics

Raw `meta/stats.json` is insufficient when transforms change dimensionality, for example, when converting a 7D quaternion representation to a 9D xyz+rot6d representation. Generate statistics for the exact post-transform representation using [`scripts/compute_openh_action_stats.py`](../scripts/compute_openh_action_stats.py):

```bash
python scripts/compute_openh_action_stats.py \
  --dataset-path /datasets/my_robot/success \
  --embodiment my_dual_arm_robot

python scripts/compute_openh_action_stats.py \
  --dataset-path /datasets/my_robot/failures \
  --embodiment my_dual_arm_robot
```

Each dataset should then contain:

```text
meta/stats_cosmos.json
```

CMR Versius uses its specialized `meta/stats_cosmos-44D.json` path instead. If you change `timestep_interval`, rotation convention, key order, or action representation, regenerate statistics.

### 4.4 Register 13-frame and 73-frame dataloaders

In `cosmos_predict2/_src/predict2/action/configs/action_conditioned/data.py`:

The reference `jhu_dvrk_mono_finetune_{train,val}` and
`jhu_dvrk_mono_finetune_h73_{train,val}` loaders are already registered in
[`action_conditioned/data.py`](../cosmos_predict2/_src/predict2/action/configs/action_conditioned/data.py).
The following generic form shows what to change for another robot:

```python
my_robot_h13_train_dataset = L(MixedLeRobotDataset)(
    dataset_specs=MY_TRAIN_SPECS,
    num_frames=13,
    data_split="train",
    max_action_dim=MAX_ACTION_DIM,
    downscaled_res=False,
)
my_robot_h13_val_dataset = L(MixedLeRobotDataset)(
    dataset_specs=MY_VAL_SPECS,
    num_frames=13,
    data_split="test",
    max_action_dim=MAX_ACTION_DIM,
    downscaled_res=False,
)

my_robot_h73_train_dataset = L(MixedLeRobotDataset)(
    dataset_specs=MY_TRAIN_SPECS,
    num_frames=73,
    data_split="train",
    max_action_dim=MAX_ACTION_DIM,
    downscaled_res=False,
)
my_robot_h73_val_dataset = L(MixedLeRobotDataset)(
    dataset_specs=MY_VAL_SPECS,
    num_frames=73,
    data_split="test",
    max_action_dim=MAX_ACTION_DIM,
    downscaled_res=False,
)
```

Register each with a `DataLoader` and Hydra `ConfigStore`, following the existing `suturebot_train` and `suturebot_val` registrations:

```python
cs.store(
    group="data_train",
    package="dataloader_train",
    name="my_robot_h13_train",
    node=L(DataLoader)(
        dataset=my_robot_h13_train_dataset,
        sampler=L(get_sampler)(dataset=my_robot_h13_train_dataset),
        batch_size=1,
        drop_last=True,
    ),
)
```

Repeat for `my_robot_h13_val`, `my_robot_h73_train`, and `my_robot_h73_val`.

### 4.5 Dataset preflight

Before allocating many GPUs, instantiate the 13-frame and 73-frame datasets in a single process and check:

```python
sample = dataset[0]
assert sample["video"].shape[1] == NUM_FRAMES
assert sample["action"].shape == (NUM_FRAMES - 1, 44)
assert sample["video"].dtype == torch.uint8
assert torch.isfinite(sample["action"]).all()
```

Confirm that:

- the context and actions come from the same episode;
- the 73-frame window does not cross episode boundaries;
- video and action timestamps use the same stride;
- transformed actions have plausible means and scales;
- train and validation episode splits do not overlap.

Unlike the CMR procedure-filtering path, a standard `MixedLeRobotDataset` does not require a separate filter manifest for each horizon. It does require a `stats_cosmos.json` file for every dataset.

## 5. Stage 1: short-horizon bidirectional teacher

Warm-start the fine-tuning from the [C-H-S-S checkpoint](https://huggingface.co/nvidia/Cosmos-H-Surgical-Simulator) rather than from the generic Cosmos-Predict2.5 checkpoint to benefit from the surgical visual prior and the trained 44D action embedder.

Add an experiment to
`cosmos_predict2/_src/predict2/action/configs/action_conditioned/experiment/exp_2B_action_conditioned_rectify_flow_gr00t.py`:

The committed reference symbol is
`AC_CHUNK_SINGLE_VIEW_2B_JHU_DVRK_MONO_FINETUNE_13FRAME_8NODES_OSS` in
[`exp_2B_action_conditioned_rectify_flow_gr00t.py`](../cosmos_predict2/_src/predict2/action/configs/action_conditioned/experiment/exp_2B_action_conditioned_rectify_flow_gr00t.py).

```python
MY_ROBOT_H13_TEACHER = LazyDict(
    dict(
        defaults=[
            "/experiment/2b_bridge_action_conditioned_oss",
            {"override /net": "cosmos_v1_2B_action_chunk_conditioned"},
            {"override /data_train": "my_robot_h13_train"},
            {"override /data_val": "my_robot_h13_val"},
            "_self_",
        ],
        job=dict(
            project="cosmos_predict2_action_conditioned",
            group="official_runs_vid2vid",
            name="my_robot_h13_teacher",
        ),
        checkpoint=dict(
            # DCP directory, not model_ema_bf16.pt.
            load_path="/checkpoints/cosmos_h_surgical_simulator/iter_NNNNN",
            load_training_state=False,
            strict_resume=False,
        ),
        model=dict(
            config=dict(
                state_t=1 + 12 // 4,
                net=dict(action_dim=44),
            ),
        ),
        dataloader_train=dict(batch_size=16),
        optimizer=dict(lr=1.6e-4, weight_decay=0.1),
    ),
    flags={"allow_objects": True},
)
```

Register it:

```python
cs.store(
    group="experiment",
    package="_global_",
    name=MY_ROBOT_H13_TEACHER["job"]["name"],
    node=MY_ROBOT_H13_TEACHER,
)
```

Important checkpoint rules:

- Training initialization expects the DCP iteration directory.
- `load_training_state=False` resets the source optimizer, scheduler, and iteration count.
- `strict_resume=False` allows nonessential source-run state to differ.
- The model-facing action width must remain 44 if the C-H-S-S action embedder is to load without shape changes.
- A consolidated `model_ema_bf16.pt` is primarily for inference, not this DCP warm start.

Launch the committed reference config directly, or use the sanitized
[`01_train_short_teacher_h13.sh`](../train_scripts/tabletop/01_train_short_teacher_h13.sh)
template:

```bash
torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train \
  --config=cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py \
  -- \
  experiment=cosmos_predict2p5_2B_action_conditioned_jhu_dvrk_mono_finetune_13frame_8nodes_release_oss \
  checkpoint.save_iter=200 \
  ~dataloader_train.dataloaders
```

For multiple nodes, add the usual `torchrun --nnodes`, `--node_rank`, and `--master_addr` arguments under your scheduler.

### 5.1 Fine-tune the short teacher with cosine annealing

In the reference tabletop run, iteration 16,000 was selected as the base checkpoint and refined in a separate 4,000-iteration cosine-annealing run. This made the phase boundary explicit and avoided carrying over the optimizer and scheduler state:

The exact reference config is
`AC_CHUNK_SINGLE_VIEW_2B_JHU_DVRK_MONO_FINETUNE_13FRAME_8NODES_OSS_FINE_ANNEAL_4K`;
the portable launcher is
[`02_fine_anneal_short_teacher_h13.sh`](../train_scripts/tabletop/02_fine_anneal_short_teacher_h13.sh).

```python
MY_ROBOT_H13_FINE_ANNEAL = LazyDict(
    dict(
        defaults=["/experiment/my_robot_h13_teacher", "_self_"],
        job=dict(
            project="cosmos_predict2_action_conditioned",
            group="official_runs_vid2vid",
            name="my_robot_h13_teacher_fine_anneal_4k",
        ),
        checkpoint=dict(
            load_path="/output/.../my_robot_h13_teacher/checkpoints/iter_000016000",
            load_training_state=False,
            strict_resume=False,
        ),
        scheduler=L(LambdaWarmUpCosineScheduler)(
            warm_up_steps=[100],
            f_start=[0.10],
            f_max=[1.00],
            f_min=[0.05],
            cycle_lengths=[4000],
        ),
        trainer=dict(max_iter=4000),
    ),
    flags={"allow_objects": True},
)
```

Do not automatically select iteration 16,000 for a different dataset. Use validation rollouts and smoothed loss to select the short-teacher checkpoint, then run the cosine-annealing stage from that checkpoint.

## 6. Stage 2: long-horizon bidirectional teacher

This recipe applies the progressive temporal post-training approach from [OmniDreams](https://arxiv.org/abs/2606.03159): first learn the target domain over a manageable short horizon, then extend the same teacher to a longer horizon. The network weights are compatible across horizons; the longer run changes the training clip length and latent temporal length.

For the 73-frame reference tabletop run, the committed symbol is
`AC_CHUNK_SINGLE_VIEW_2B_JHU_DVRK_MONO_TABLETOP_H73_8NODES_OSS`; see
[`exp_2B_action_conditioned_rectify_flow_gr00t.py`](../cosmos_predict2/_src/predict2/action/configs/action_conditioned/experiment/exp_2B_action_conditioned_rectify_flow_gr00t.py)
and [`03_train_long_teacher_h73.sh`](../train_scripts/tabletop/03_train_long_teacher_h73.sh).

```python
MY_ROBOT_H73_TEACHER = LazyDict(
    dict(
        defaults=[
            "/experiment/my_robot_h13_teacher",
            {"override /data_train": "my_robot_h73_train"},
            {"override /data_val": "my_robot_h73_val"},
            "_self_",
        ],
        job=dict(
            project="cosmos_predict2_action_conditioned",
            group="official_runs_vid2vid",
            name="my_robot_h73_teacher",
        ),
        checkpoint=dict(
            load_path=(
                "/output/.../my_robot_h13_teacher_fine_anneal_4k/"
                "checkpoints/iter_000004000"
            ),
            load_training_state=False,
            strict_resume=False,
        ),
        model=dict(
            config=dict(
                state_t=1 + 72 // 4,  # 19
                net=dict(action_dim=44),
            ),
        ),
        dataloader_train=dict(batch_size=4),
        optimizer=dict(lr=4e-5, weight_decay=0.1),
        scheduler=L(LambdaWarmUpCosineScheduler)(
            warm_up_steps=[1000],
            f_start=[0.10],
            f_max=[1.00],
            f_min=[0.05],
            cycle_lengths=[5000],
        ),
        trainer=dict(max_iter=5000),
    ),
    flags={"allow_objects": True},
)
```

The reference schedule was:

```text
iterations 0-1000:    linear warmup, 0.1x -> 1.0x base LR
iterations 1000-5000: cosine decay, 1.0x -> 0.05x base LR
```

With a base LR of `4e-5`, the peak is `4e-5` and the final LR is `2e-6`.
Keep `cycle_lengths[0] == trainer.max_iter` so the cosine reaches its endpoint at the checkpoint consumed by Phase 0.

Launch it with the same training entry point:

```bash
torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train \
  --config=cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py \
  -- \
  experiment=cosmos_predict2p5_2B_action_conditioned_jhu_dvrk_mono_finetune_13frame_8nodes_release_oss_h73_tabletop \
  checkpoint.save_iter=200 \
  ~dataloader_train.dataloaders
```



### 6.1 Batch and LR scaling

The reference 8-node run used:

```text
8 nodes x 8 GPUs x batch 4 = global batch 256, LR 4e-5
```

If 73 frames at batch 4 OOMs on 80-GB GPUs, use batch 2 and, if the node count is unchanged, reduce the LR to `2e-5`. For other cluster sizes, preserve the global batch and learning rate unless you intentionally want a different optimization run.

### 6.2 Decide whether the long teacher is ready

Raw rectified-flow loss depends strongly on sampled diffusion time and can look like a sawtooth. Prefer:

- a moving mean over at least 100 iterations;
- like-for-like loss buckets at the same logging offset;
- held-out 73-frame rollouts;
- FDS (frame decay score) or task-specific metrics;
- visual action responsiveness and temporal consistency.

In the reference tabletop run, the 100-step moving mean was flat for roughly the final 1,100 iterations, and the cosine had reached its floor at iteration 5,000. That was stronger evidence of convergence than the jagged raw loss.

Convert the selected teacher checkpoint for cache-generation inference:

```bash
TEACHER_DCP=/output/.../my_robot_h73_teacher/checkpoints/iter_000005000
python scripts/convert_distcp_to_pt.py \
  "$TEACHER_DCP/model" \
  "$TEACHER_DCP"

test -f "$TEACHER_DCP/model_ema_bf16.pt"
```



## 7. Stage 3: Phase 0 teacher-trajectory cache

Phase 0 runs the frozen bidirectional teacher and stores its intermediate denoising latents at selected query steps. The causal warmup does not train directly on source videos; it trains on these teacher targets.

The reference cache used:

```text
cache size:       10,000 examples
num_frames:       73
actions/example:  72
query steps:      0, 9, 18, 27, 34
sampling:         random over the full training mixture
seed:             0
guidance:         0
```

Expected layout:

```text
datasets/my_robot_warmup_4step_h73/
├── actions/
│   └── <dataset-index>.json
├── images/
│   └── <dataset-index>.png
├── latents/
│   └── <dataset-index>.pt
├── videos/
│   └── <dataset-index>.mp4
└── indices.json
```

Each latent file is a dictionary keyed by the queried denoising-step indices. Each action JSON file must contain an array with shape `(72, 44)`.

### 7.1 Use the mixed-dataset cache generator

The script `cosmos_predict2/_src/predict2/action/inference/inference_gr00t_warmup.py` shows the core call:

```python
_, _, latents_to_save = video2world_cli.step_inference_with_latents(
    img_array=first_frame,
    action=action,
    guidance=0,
    seed=0,
    num_latent_conditional_frames=1,
    query_steps=[0, 9, 18, 27, 34],
)

latents_to_save = {
    step: latent.squeeze(0).cpu()
    for step, latent in latents_to_save.items()
}
```

The script [`inference_jhu_dvrk_warmup.py`](../cosmos_predict2/_src/predict2/action/inference/inference_jhu_dvrk_warmup.py) implements the mixed-dataset cache generation used by the tabletop recipe. It does the following:

1. constructs the same `MixedLeRobotDataset` and dataset specs used by the teacher;
2. accepts `--num_frames` rather than hardcoding 13;
3. asserts `(num_frames - 1) % 4 == 0`;
4. samples across the full virtual mixture rather than taking the first 10,000 sequential indices;
5. writes globally unique filenames based on the selected dataset index;
6. skips only examples for which all required artifacts already exist.

Its seeded `build_global_index_list()` logic is equivalent to:

```python
def build_global_index_list(dataset_len: int, total: int, seed: int) -> np.ndarray:
    n = min(dataset_len, total)
    return np.random.default_rng(seed).permutation(dataset_len)[:n]
```

Every rank must build the same seeded list, then consume a disjoint slice.

### 7.2 Single-GPU command

For the reference tabletop mixture, run the committed cache generator directly:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python \
  cosmos_predict2/_src/predict2/action/inference/inference_jhu_dvrk_warmup.py \
  --experiment cosmos_predict2p5_2B_action_conditioned_jhu_dvrk_mono_finetune_13frame_8nodes_release_oss_h73_tabletop \
  --ckpt_path /output/.../iter_000005000/model_ema_bf16.pt \
  --save_root datasets/jhu_dvrk_mono_warmup_4step_h73_tabletop \
  --resolution 288,512 \
  --guidance 0 \
  --num_frames 73 \
  --chunk_size 72 \
  --sample_strategy random \
  --total_samples 10000 \
  --indices_seed 0 \
  --start 0 \
  --end 10000 \
  --query_steps 0,9,18,27,34
```

The Phase 0 `.pt` input is the consolidated long-horizon teacher EMA checkpoint. This is different from the DCP directory used to initialize training.

For SLURM, use the sanitized
[`04_phase0_teacher_cache_h73.sh`](../train_scripts/tabletop/04_phase0_teacher_cache_h73.sh)
template.

### 7.3 Multi-GPU sharding

Phase 0 performs inference only, and ranks can process their shards independently. With `N` total ranks:

```bash
SAMPLES_PER_RANK=$(( (TOTAL_SAMPLES + N_RANKS - 1) / N_RANKS ))
START=$(( GLOBAL_RANK * SAMPLES_PER_RANK ))
END=$(( (GLOBAL_RANK + 1) * SAMPLES_PER_RANK ))
(( END > TOTAL_SAMPLES )) && END=$TOTAL_SAMPLES
```

On SLURM with eight tasks per node:

```bash
GLOBAL_RANK=$(( SLURM_NODEID * 8 + SLURM_LOCALID ))
N_RANKS=$(( SLURM_NNODES * 8 ))
```

Do not shard only by `SLURM_LOCALID`; it repeats ranks 0-7 on every node and causes duplicate cache generation.

### 7.4 Cache integrity gate

Do not start warmup merely because the cache job exited. Verify complete artifact quartets:

```bash
CACHE=datasets/jhu_dvrk_mono_warmup_4step_h73_tabletop

python - "$CACHE" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
latent_ids = {p.stem for p in (root / "latents").glob("*.pt")}
action_ids = {p.stem for p in (root / "actions").glob("*.json")}
image_ids = {p.stem for p in (root / "images").glob("*.png")}
video_ids = {p.stem for p in (root / "videos").glob("*.mp4")}
complete = latent_ids & action_ids & image_ids & video_ids

print({
    "latents": len(latent_ids),
    "actions": len(action_ids),
    "images": len(image_ids),
    "videos": len(video_ids),
    "complete": len(complete),
})
assert len(complete) == 10_000

example = next(iter(complete))
actions = json.loads((root / "actions" / f"{example}.json").read_text())
assert len(actions) == 72
assert all(len(row) == 44 for row in actions)
PY
```

Use the same teacher checkpoint for Phase 0 and Self Forcing.

## 8. Stage 4: causal-student warmup



### 8.1 Register the cache

In `cosmos_predict2/_src/predict2/interactive/configs/data.py`:

The reference cache is already registered as `jhu_dvrk_mono_warmup_h73` in
[`interactive/configs/data.py`](../cosmos_predict2/_src/predict2/interactive/configs/data.py).
Use the following pattern for a differently named cache:

```python
dataset_my_robot_warmup_h73 = L(ActionDatasetSFWarmup)(
    data_path="datasets/my_robot_warmup_4step_h73",
    cr1_embeddings_path="cr1_empty_string_text_embeddings.pt",
)

cs.store(
    group="data_train",
    package="dataloader_train",
    name="my_robot_warmup_h73",
    node=make_dataloader(dataset_my_robot_warmup_h73),
)
cs.store(
    group="data_val",
    package="dataloader_val",
    name="my_robot_warmup_h73",
    node=make_dataloader(dataset_my_robot_warmup_h73),
)
```

`ActionDatasetSFWarmup` reads the five latent targets `[0, 9, 18, 27, 34]`, the context image, and the action sequence.

### 8.2 Register the warmup experiment

In `cosmos_predict2/_src/predict2/interactive/configs/experiment/exp_action_warmup.py`:

The committed reference symbol is `ACTION_JHU_DVRK_MONO_TABLETOP_H73_WARMUP` in
[`exp_action_warmup.py`](../cosmos_predict2/_src/predict2/interactive/configs/experiment/exp_action_warmup.py).

```python
MY_ROBOT_H73_WARMUP = make_experiment(
    name="my_robot_h73_warmup",
    data="my_robot_warmup_h73",
    overrides=dict(
        checkpoint=dict(
            # DCP directory. This initializes the causal network weights.
            load_path=(
                "/output/.../my_robot_h13_teacher_fine_anneal_4k/"
                "checkpoints/iter_000004000"
            ),
        ),
        model=dict(
            config=dict(
                state_t=1 + 72 // 4,
                net=dict(action_dim=44),
                resolution=288,
            ),
        ),
        dataloader_train=dict(batch_size=2),
        optimizer=dict(lr=3e-5),
        trainer=dict(max_iter=20000),
    ),
)
```

The validated 73-frame experiment deliberately kept the causal network's inherited `num_action_per_chunk=12`. This field controls the local causal generation chunk, not the full 72-action training horizon. The full horizon is defined by `state_t=19` and the cache tensor shape. Do not change the local chunk to 72 unless the network and runtime are explicitly designed and validated for that attention and chunking behavior.

Register a local/no-object-store variant with the override-preserving, resumable helper:

```python
cs.store(
    group="experiment",
    package="_global_",
    name="my_robot_h73_warmup_no_s3_resumable",
    node=build_no_s3_run_v2(
        MY_ROBOT_H73_WARMUP,
        local_path=True,
        resumable=True,
        load_training_state=False,
        wandb_mode="offline",
    ),
)
```

The tabletop experiment used a no-S3 resumable helper that preserved the custom overrides and loaded the EMA teacher weights into the fresh student's regular network.

### 8.3 Launch warmup

The portable reference launcher is
[`05_warmup_student_h73.sh`](../train_scripts/tabletop/05_warmup_student_h73.sh).

```bash
torchrun --nproc_per_node=8 --master_port=12342 -m scripts.train \
  --config=cosmos_predict2/_src/predict2/interactive/configs/config_warmup.py \
  -- \
  experiment=cosmos_predict2p5_2B_action_jhu_dvrk_mono_tabletop_h73_warmup_no_s3_resumable \
  checkpoint.save_iter=200
```

The reference scaling was:

```text
8 nodes x 8 GPUs x batch 2 = global batch 128, LR 3e-5
```



### 8.4 Select the warmup checkpoint

The reference warmup used 20,000 iterations as a ceiling, not a requirement. In the tabletop run:

- loss fell steeply through the first several thousand iterations;
- continued improving slowly after 10,000;
- became effectively flat around 16,000-19,000;
- iteration 18,000 was selected because it was a complete checkpoint inside the plateau.

Use a moving mean rather than a single batch loss. Once the moving mean has no trend for several thousand iterations, a checkpoint in that plateau is a reasonable Self Forcing initialization.

The causal warmup and Self Forcing networks require the NATTEN multidimensional attention backend used by `action_causal_cosmos_v1_2B`. Verify `import natten` and the configured attention backend before allocating a multi-node job.

## 9. Stage 5: Self Forcing distillation

Self Forcing closes the train-test gap by rolling out the causal student during training, conditioning future predictions on its own generated history, and matching the resulting distribution to the teacher's distribution. The training model holds:

- `net`: causal student;
- `net_teacher`: frozen bidirectional teacher;
- `net_fake_score`: auxiliary score/critic network;
- optional discriminator components used by the configured DMD/GAN losses.

The loss is adversarial and is not expected to decrease monotonically. Judge health by losses that remain finite and bounded, successful checkpointing, and rollout quality.

### 9.1 Register the Self Forcing experiment

In `cosmos_predict2/_src/predict2/interactive/configs/experiment/exp_action_self_forcing.py`:

The committed reference symbol is
`ACTION_JHU_DVRK_MONO_TABLETOP_H73_SELF_FORCING` in
[`exp_action_self_forcing.py`](../cosmos_predict2/_src/predict2/interactive/configs/experiment/exp_action_self_forcing.py).

```python
MY_ROBOT_H73_SELF_FORCING = make_experiment(
    name="my_robot_h73_self_forcing",
    data="my_robot_warmup_h73",
    overrides=dict(
        job=dict(
            project="cosmos_predict2_action_conditioned",
            group="interactive_self_forcing",
        ),
        checkpoint=dict(
            # Selected causal warmup DCP.
            load_path=(
                "/output/.../interactive_warmup/my_robot_h73_warmup/"
                "checkpoints/iter_000018000"
            ),
        ),
        trainer=dict(max_iter=3000),
        optimizer=dict(lr=5e-8),
        model=dict(
            config=dict(
                state_t=1 + 72 // 4,
                net=dict(action_dim=44),
                net_fake_score=dict(action_dim=44),
                net_teacher=dict(action_dim=44),
                optimizer_discriminator_config=dict(lr=5e-6),
                optimizer_fake_score_config=dict(lr=5e-6),
                resolution="288",
                teacher_load_from=dict(
                    # Must match the teacher used to create Phase 0.
                    load_path=(
                        "/output/.../my_robot_h73_teacher/"
                        "checkpoints/iter_000005000/model"
                    ),
                    credentials="",
                ),
            ),
        ),
    ),
)
```

The three path types are deliberately different:

```text
warmup student initialization:  .../iter_000018000        (DCP iteration dir)
frozen teacher initialization:  .../iter_000005000/model  (DCP model shard dir)
eventual deployment checkpoint: .../model_ema_bf16.pt     (consolidated file)
```

Use `config_distill.py`, which selects the distillation-aware checkpointer:

```python
cs.store(
    group="experiment",
    package="_global_",
    name="my_robot_h73_self_forcing_no_s3_resumable",
    node=build_no_s3_run_v2(
        MY_ROBOT_H73_SELF_FORCING,
        local_path=True,
        resumable=True,
        load_training_state=False,
        wandb_mode="offline",
    ),
)
```



### 9.2 Launch Self Forcing

Use [`06_self_forcing_h73.sh`](../train_scripts/tabletop/06_self_forcing_h73.sh)
for the validated 8-node reference.

```bash
torchrun --nproc_per_node=8 --master_port=12343 -m scripts.train \
  --config=cosmos_predict2/_src/predict2/interactive/configs/config_distill.py \
  -- \
  experiment=cosmos_predict2p5_2B_action_jhu_dvrk_mono_tabletop_h73_self_forcing_no_s3_resumable \
  checkpoint.save_iter=200
```

The reference 8-node run used:

```text
global batch:                    8 x 8 x 1 = 64
student/DMD LR:                  5e-8
discriminator LR:                5e-6
fake-score LR:                   5e-6
iterations:                      3000
```



### 9.3 Self Forcing health checks

Stop and investigate if you see:

- NaN or infinite losses;
- rapidly exploding loss magnitudes;
- repeated CUDA OOM before the first optimizer step;
- missing student, fake-score, optimizer, or trainer shards;
- a cache loaded with substantially fewer examples than expected;
- a teacher path different from the Phase 0 teacher;
- repeated restarts from iteration zero.

Do not stop merely because the composite training loss rises. Evaluate checkpoints with open-loop action sequences and, ideally, closed-loop policy rollouts.

## 10. Convert and validate the distilled student

The Self Forcing checkpointer stores multiple networks and training state in DCP format. Convert the selected iteration to `.pt` format:

```bash
SF_DCP=/output/.../interactive_self_forcing/my_robot_h73_self_forcing/checkpoints/iter_000003000

python scripts/convert_distcp_to_pt.py \
  "$SF_DCP/model" \
  "$SF_DCP"

ls -lh \
  "$SF_DCP/model.pt" \
  "$SF_DCP/model_ema_fp32.pt" \
  "$SF_DCP/model_ema_bf16.pt"
```

Use `model_ema_bf16.pt` for deployment unless the runtime's model card says otherwise.

Before moving to an optimized runtime, smoke-test the DCP checkpoint as follows:

1. Create the streaming manifest and its corresponding ground-truth MP4 files and normalized action files using the committed [`extract_jhu_inference_manifest.py`](../scripts/extract_jhu_inference_manifest.py):

```bash
python scripts/extract_jhu_inference_manifest.py \
  --subset hf_suturebot \
  --episode-ids 1440,1441,1442 \
  --num-frames 73 \
  --output-dir sf_inference_data/jhu_tabletop_test_h73
```

Choose test episode IDs that exist in your dataset. The extractor uses the same JHU action transform, 44D padding, and `stats_cosmos.json` normalization as training.

2. Generate rollouts for inspection using the generated streaming manifest:

```bash
python -m cosmos_predict2._src.predict2.interactive.inference.action_video2world_streaming \
  --config cosmos_predict2/_src/predict2/interactive/configs/config_distill.py \
  --experiment cosmos_predict2p5_2B_action_jhu_dvrk_mono_tabletop_h73_self_forcing_no_s3_resumable \
  --ckpt_path "$SF_DCP" \
  --input_json sf_inference_data/jhu_tabletop_test_h73/jhu_tabletop_inference_manifest.json \
  --resolution 288,512 \
  --num_steps 4 \
  --max_frames 73
```

Note: Parts of the public streaming script's rollout loop assume 12-action chunks. Audit and parameterize the affected fields before using it as the definitive 73-frame evaluation path.

## 11. Deploy with Cosmos-H-Dreams

[Cosmos-H-Dreams](https://github.com/isaac-for-healthcare/Cosmos-H-Dreams) runs the distilled surgical causal student in real time. Use its C-H-S-S/Cosmos-H model integration and add a checkpoint preset for the converted student. See the [documentation](https://github.com/isaac-for-healthcare/Cosmos-H-Dreams/blob/main/README.md) for instructions on running the student with your chosen controller.

Keep the following training and distillation requirements in mind during deployment.

### 11.1 Action preprocessing at runtime

The model does not accept an arbitrary vector of 44 raw robot values. Deployment must reproduce the training pipeline:

1. sample actions at the training rate;
2. use the same context/reference robot state;
3. convert translations and rotations to the configured relative representation;
4. normalize with the same `stats_cosmos.json`;
5. concatenate keys in the same order;
6. pad the transformed native action from 20D to 44D;
7. provide one 44D vector per generated pixel-frame transition.

For a dVRK-like 20D transformed vector:

```python
def pad_action_to_44(action_20d: np.ndarray) -> np.ndarray:
    assert action_20d.shape[-1] == 20
    out = np.zeros((*action_20d.shape[:-1], 44), dtype=np.float32)
    out[..., :20] = action_20d.astype(np.float32)
    return out
```

This helper is only the final padding operation. It does not perform pose conversion or normalization.

### 11.2 Real-time validation sequence

Validate in this order:

1. Load the checkpoint and verify that there are no missing or unexpected model keys.
2. Generate a rollout from a recorded first frame and normalized action trace.
3. Compare the first short rollout against the training-repository streaming implementation.
4. Verify action direction with simple isolated motions.
5. Measure first-chunk latency separately from steady-state latency.
6. Discard compilation/autotuning warmup chunks before reporting throughput.
7. Run a long open-loop trace and inspect rolling-cache degradation.
8. Connect a live controller or policy only after confirming parity with recorded traces.

Measure both generation throughput and end-to-end control-loop latency. A model can generate faster than real time while still having an unacceptable first-chunk or action-ingestion delay.

## 12. Common failure modes

The following failure modes can occur during distillation.

### Checkpoints “saved” but are absent

`IMAGINAIRE_OUTPUT_ROOT` points to container-local storage. Bind persistent storage and export the variable inside the container.

### The long teacher OOMs

Reduce the per-GPU batch size. If the number of GPUs is unchanged, scale the LR with the global batch. Context parallelism or activation checkpointing are the next options when a batch size of 1 still fails.

### A long-horizon run silently trains on 13 frames

Check all three values together:

```text
dataset num_frames = 73
model state_t       = 19
action sequence     = 72
```

Also inspect cache-generation and streaming scripts for hardcoded 12-action loops.

Run a separate horizon-aware evaluation job for 49- or 73-frame teachers.

### Warmup starts on a partial cache

Verify that every sample has all required artifacts; do not rely only on job completion or the number of latent files.

### Multi-node Phase 0 creates duplicates

Use global rank, not local rank, for shard boundaries.

### The student checkpoint fails to load in the runtime

Check `action_dim`, `num_action_per_latent_frame`, and `hidden_dim_in_action_embedder` first. In the reference tabletop run, `action_dim` was 44 and `num_action_per_latent_frame` was 4; derive the hidden width from the converted checkpoint. The current default is 8192.

### Self Forcing loss rises

That alone is not a divergence signal. Check that losses remain finite and bounded, then evaluate rollouts. The objective alternates student and critic/fake-score updates.

### Hugging Face returns HTTP 429 at distributed startup

Pre-download shared text embeddings and checkpoint assets to a mounted cache. Avoid having every rank independently fetch the same files.

## 13. Further reading and resources

- Cosmos-H-Surgical-Simulator: [Hugging Face model](https://huggingface.co/nvidia/Cosmos-H-Surgical-Simulator)
- Open-H-Embodiment: [Hugging Face dataset](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Open-H-Embodiment)
- Cosmos-H-Dreams code and examples: [GitHub repository](https://github.com/isaac-for-healthcare/Cosmos-H-Dreams)
- Cosmos-H-Dreams model: [Hugging Face checkpoint](https://huggingface.co/nvidia/Cosmos-H-Dreams)
- NVIDIA Cosmos-Predict2.5: [GitHub repository](https://github.com/nvidia-cosmos/cosmos-predict2.5)
- Cosmos-Surg-dVRK: *World Foundation Model-based Automated Online Evaluation of Surgical Robot Policy Learning*: [https://arxiv.org/abs/2510.16240](https://arxiv.org/abs/2510.16240)
- Self Forcing: *Bridging the Train-Test Gap in Autoregressive Video Diffusion*: [https://arxiv.org/abs/2506.08009](https://arxiv.org/abs/2506.08009)
- NVIDIA OmniDreams: *Real-Time Generative World Model for Closed-Loop Autonomous Vehicle Simulation*: [https://arxiv.org/abs/2606.03159](https://arxiv.org/abs/2606.03159)
