<p align="center">
    <img src="https://github.com/user-attachments/assets/28f2d612-bbd6-44a3-8795-833d05e9f05f" width="274" alt="NVIDIA Cosmos"/>
</p>

<p align="center">
  <a href="https://www.nvidia.com/en-us/ai/cosmos">Product Website</a>&nbsp | 🤗 <a href="https://huggingface.co/collections/nvidia/cosmos-predict25-68bb63255f2fc206c5e5b346">Hugging Face</a>&nbsp | <a href="https://arxiv.org/abs/2511.00062">Paper</a>&nbsp | <a href="https://research.nvidia.com/labs/dir/cosmos-predict2.5">Paper Website</a> | <a href="https://github.com/nvidia-cosmos/cosmos-cookbook">Cosmos Cookbook</a>
</p>

NVIDIA Cosmos™ is a platform purpose-built for physical AI, featuring state-of-the-art generative world foundation models (WFMs), robust guardrails, and an accelerated data processing and curation pipeline. Designed specifically for real-world systems, Cosmos enables developers to rapidly advance physical AI applications such as robotic surgery, autonomous vehicles, robots, and video analytics AI agents.

## News!
* [March 2026] Released Cosmos-H-Surgical-Simulator: action-conditioned video generation for the [Open-H](https://huggingface.co/datasets/nvidia/Open-H) multi-embodiment surgical robotics benchmark, supporting 9 robot embodiments across 32 datasets.

## Cosmos-H-Surgical-Simulator

We introduce Cosmos-H-Surgical-Simulator, specialized for simulating and predicting the future state of the surgical world in the form of action-conditioned video. Cosmos-H-Surgical-Simulator is a flow-based model that utilizes Cosmos-Reason1, a Physical AI reasoning vision language model (VLM), as the text encoder. It is built upon the [Cosmos-Predict2.5-2B](https://github.com/nvidia-cosmos/cosmos-predict2.5) model and adapted specifically for multi-embodiment surgical robotics data from the [Open-H](https://huggingface.co/datasets/nvidia/Open-H) benchmark.

Given a context frame and a sequence of robot actions, the model generates a video predicting the future visual state of the surgical scene. It supports **9 distinct robot embodiments** across **10+ institutions**, as well as single-embodiment training on CMR Surgical Versius data and downstream fine-tuning on individual datasets (e.g., SutureBot).

## Cosmos-H-Surgical-Simulator Model Family

| Model Name | Capability | Input |
| --- | --- | --- |
| Cosmos-H-Surgical-Simulator / Open-H | Multi-embodiment action-conditioned (44D) | action + image |

## Overview

Cosmos-H-Surgical-Simulator extends the upstream [Cosmos-Predict2.5](https://github.com/nvidia-cosmos/cosmos-predict2.5) action-conditioned video generation pipeline with:

- **Multi-embodiment surgical data loading** via `MixedLeRobotDataset`, supporting heterogeneous action spaces, timestep intervals, and per-embodiment normalization across the full Open-H benchmark.
- **Unified 44D action conditioning** — all embodiments project into a shared 44-dimensional action vector (defined by CMR Versius, the most complex embodiment), with smaller action spaces zero-padded to match.
- **Per-embodiment transform pipelines** that handle clutch-aware motion for CMR Versius, relative EEF pose computation with mixed rotation formats (quaternion, Euler, rot6d), and joint-space actions for USTC Torin.
- **Weighted data mixing** following the gr00t-H training recipe: 50% CMR Versius, 50% distributed proportionally across remaining embodiments by frame count.
- **Quantitative evaluation** with three complementary metrics (FDS, GATC, TCD) for measuring pixel fidelity, tool consistency, and spatial tool accuracy against ground truth.
- **Downstream fine-tuning** support — the multi-embodiment Open-H model serves as a foundation for further specialization on individual datasets (e.g., SutureBot).

## Supported Embodiments

| Embodiment | Robot | Action Dim | Institutions |
|---|---|---|---|
| `cmr_versius` | CMR Surgical Versius | **44D** (30D actions + 14D state conditioning) | CMR Surgical |
| `rob_surgical` | Rob Surgical bitrack | **36D** (4-arm EEF) | Rob Surgical |
| `dvrk` / `jhu_dvrk_mono` | da Vinci Research Kit | **20D** (dual-arm EEF + gripper) | JHU, JHU LSCR ARCADE |
| `dvrk_ucb` | dVRK UC Berkeley | **20D** | UC Berkeley |
| `dvrk_stanford_real` | dVRK Stanford | **20D** (Euler RPY input) | Stanford |
| `dvrk_ucsd` | dVRK UCSD | **20D** | UCSD |
| `hamlyn_30hz` | Hamlyn Centre dVRK | **20D** | Hamlyn Centre |
| `turin_mitic_ex_vivo` | Turin MITIC | **18D** (no grippers) | University of Turin |
| `ustc_torin` | USTC Torin | **variable** (joint-space) | USTC |
| `moon` | Moon Surgical Maestro | **6D** (delta XYZ only) | Moon Surgical |

All embodiments are zero-padded to 44D at the model input layer. See [`README_ACTION_SPACE.md`](scripts/README_ACTION_SPACE.md) for the full action vector layout and per-embodiment mapping details.

## Data: Open-H Benchmark

Training uses **32 datasets across 9 embodiments** from the [Open-H](https://huggingface.co/datasets/nvidia/Open-H) collection, stored in [LeRobot](https://github.com/huggingface/lerobot) format. The data mixture allocates 50% of training compute to CMR Versius (4 surgical procedures) and 50% to all other embodiments (step-weighted by frame count).

| Group | Datasets | Approx. Frames | Training Share |
|---|---|---|---|
| CMR Versius | cholecystectomy, hysterectomy, inguinal hernia, prostatectomy | ~17M | 50% |
| dVRK JHU | porcine chole, electrocautery, suturebot (x3), tissue, needle, suturing | ~2.8M | 25% |
| Stanford Real | Needle Transfer, Tissue Retraction, Peg Transfer | ~874K | 8% |
| Hamlyn Centre | Suturing (x2), peg transfer, needle, knot tying, tissue retraction | ~545K | 5% |
| Turin MITIC | mitic_lerobot_ex_vivo | ~389K | 3.5% |
| UCSD | surgical_learning_dataset (x2) | ~315K | 3% |
| UCB | debridement_lerobot | ~222K | 2% |
| USTC Torin | knot tying, needle handover, needle pickup | ~185K | 1.7% |
| LSCR ARCADE | Cholecystectomy, cautery | ~183K | 1.7% |
| Moon Surgical | moon | ~12K | 0.1% |

Training configuration: 8 nodes / 64 GPUs, batch size 16 per GPU (global 1024), 512x288 resolution, 13 video frames (1 context + 12 prediction), 12 action timesteps per sample.

## Setup

Follow the [Setup Guide](docs/setup.md) to install dependencies and download base checkpoints.

**Quick start:**

```bash
git clone <this-repo>
cd medtech-cosmos-h-surgical-simulator
git lfs pull

# Install via uv
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --extra=cu128
source .venv/bin/activate

# Configure Hugging Face for checkpoint downloads
hf auth login
export HF_HOME=/path/to/hf/cache
```

## Post-Training

### Open-H Multi-Embodiment (recommended)

Train across all 9 embodiments with the unified 44D action space:

```bash
export IMAGINAIRE_OUTPUT_ROOT=/path/to/output

torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train \
  --config=cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py \
  -- experiment=cosmos_predict2p5_2B_action_conditioned_open_h-fixed_13frame_8nodes_release_oss \
  ~dataloader_train.dataloaders
```

### CMR Versius Only

Train on CMR Surgical Versius data exclusively (44D action space):

```bash
torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train \
  --config=cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py \
  -- experiment=cosmos_predict2p5_2B_action_conditioned_cmr_13frame_44D_8nodes_release_oss \
  ~dataloader_train.dataloaders
```

### Downstream Fine-Tuning (e.g., SutureBot)

Fine-tune the Open-H model on a single dataset. The 20D SutureBot actions are zero-padded to 44D automatically:

```bash
torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train \
  --config=cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py \
  -- experiment=cosmos_predict2p5_2B_action_conditioned_suturebot_13frame_4nodes_release_oss \
  ~dataloader_train.dataloaders
```

### Checkpoint Conversion

Training saves checkpoints in DCP format. Convert to PyTorch `.pt` for inference:

```bash
CHECKPOINTS_DIR=${IMAGINAIRE_OUTPUT_ROOT}/cosmos_predict2_action_conditioned/.../checkpoints
CHECKPOINT_ITER=$(cat $CHECKPOINTS_DIR/latest_checkpoint.txt)

python ./scripts/convert_distcp_to_pt.py $CHECKPOINTS_DIR/$CHECKPOINT_ITER/model $CHECKPOINTS_DIR/$CHECKPOINT_ITER
```

This produces `model_ema_bf16.pt` (recommended for inference).

## Inference

### Multi-Embodiment Inference (Open-H)

Run action-conditioned video generation on any supported embodiment:

```bash
# CMR Versius
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python cosmos_predict2/_src/predict2/action/inference/inference_open_h.py \
  --experiment cosmos_predict2p5_2B_action_conditioned_open_h-fixed_13frame_8nodes_release_oss \
  --ckpt_path /path/to/model_ema_bf16.pt \
  --dataset_path /CMR_Versius/cholecystectomy_480p \
  --embodiment cmr_versius \
  --episode_ids 0,1,2

# dVRK JHU (monocular)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python cosmos_predict2/_src/predict2/action/inference/inference_open_h.py \
  --experiment cosmos_predict2p5_2B_action_conditioned_open_h-fixed_13frame_8nodes_release_oss \
  --ckpt_path /path/to/model_ema_bf16.pt \
  --dataset_path /path/to/suturebot_2 \
  --embodiment jhu_dvrk_mono \
  --episode_ids 0,1,2

# Stanford Real (with split exclusions)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python cosmos_predict2/_src/predict2/action/inference/inference_open_h.py \
  --experiment cosmos_predict2p5_2B_action_conditioned_open_h-fixed_13frame_8nodes_release_oss \
  --ckpt_path /path/to/model_ema_bf16.pt \
  --dataset_path /path/to/Needle_Transfer \
  --embodiment dvrk_stanford_real \
  --exclude_splits fail bad_frames \
  --episode_ids 0,1,2
```

### CMR-Only Inference

For models trained exclusively on CMR Versius:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python cosmos_predict2/_src/predict2/action/inference/inference_cmr.py \
  --experiment cosmos_predict2p5_2B_action_conditioned_cmr_13frame_44D_8nodes_release_oss \
  --ckpt_path /path/to/model_ema_bf16.pt \
  --dataset_path /CMR_Versius/cholecystectomy_480p \
  --save_root results/cmr_eval/cholecystectomy \
  --data_split test \
  --episode_ids 0,1,2
```

## Evaluation

The quantitative evaluation pipeline benchmarks checkpoints across the full Open-H benchmark with three metrics:

| Metric | What it measures | Direction |
|---|---|---|
| **FDS** (Frame Decay Score) | Pixel-level fidelity degradation over time (mean L1) | Lower is better |
| **GATC** (GT-Anchored Tool Consistency) | Surgical tool visual consistency vs. ground truth | Higher is better |
| **TCD** (Tool Centroid Distance) | Spatial displacement of tool instances (pixels) | Lower is better |

GATC and TCD require [Medical-SAM3](https://github.com/AIM-Research-Lab/Medical-SAM3) fine-tuned on [CholecSeg8k](https://www.kaggle.com/datasets/newslab/cholecseg8k) for surgical tool segmentation. For detailed metric definitions and the full evaluation pipeline documentation, see [`README_EVALUATION.md`](scripts/README_EVALUATION.md).

### Running Evaluation

```bash
# Generate test episode index
python scripts/print_test_datasets_and_episodes.py --output output/open-h_test_episodes.json

# Run evaluation across all Open-H embodiments
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/cosmos_h_surgical_simulator_quant_eval.py \
  --ckpt_path /path/to/model_ema_bf16.pt \
  --sam3_checkpoint /path/to/medical_sam3_cholecseg8k.pt \
  --test_episodes_json output/open-h_test_episodes.json \
  --episodes_per_dataset 2 --num_seeds 2

# Compare multiple checkpoints
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/cosmos_h_surgical_simulator_quant_eval.py \
  --ckpt_path /path/to/ckpt_10k.pt /path/to/ckpt_20k.pt \
  --ckpt_labels "10k-steps" "20k-steps" \
  --sam3_checkpoint /path/to/medical_sam3_cholecseg8k.pt \
  --test_episodes_json output/open-h_test_episodes.json

# Plot results
python scripts/plot_quant_eval_results.py --json output/quant_eval/*/quant_eval_results.json
```

See [`README_EVALUATION.md`](scripts/README_EVALUATION.md) for full documentation of the evaluation pipeline, metric definitions, and output formats.

## User Guide

* [Setup Guide](docs/setup.md)
* [Troubleshooting](docs/troubleshooting.md)
* [Post-Training](docs/post-training.md)
  * [Action-Conditioned Post-Training](docs/post-training_video2world_action.md)
* [Inference](docs/inference_robot_action_cond.md)
* [Action Space Design](scripts/README_ACTION_SPACE.md)
* [Evaluation Pipeline](scripts/README_EVALUATION.md)

## Key Source Files

| Component | Path |
|---|---|
| Embodiment registry & dataset specs | [`groot_configs.py`](cosmos_predict2/_src/predict2/action/datasets/gr00t_dreams/groot_configs.py) |
| Multi-embodiment dataset & zero-padding | [`dataset.py`](cosmos_predict2/_src/predict2/action/datasets/gr00t_dreams/data/dataset.py) |
| CMR & generic action transforms | [`state_action.py`](cosmos_predict2/_src/predict2/action/datasets/gr00t_dreams/data/transform/state_action.py) |
| Open-H multi-embodiment inference | [`inference_open_h.py`](cosmos_predict2/_src/predict2/action/inference/inference_open_h.py) |
| CMR-only inference | [`inference_cmr.py`](cosmos_predict2/_src/predict2/action/inference/inference_cmr.py) |
| Experiment configs (CMR / Open-H / SutureBot) | [`exp_...gr00t.py`](cosmos_predict2/_src/predict2/action/configs/action_conditioned/experiment/exp_2B_action_conditioned_rectify_flow_gr00t.py) |
| Data config (dataloaders) | [`data.py`](cosmos_predict2/_src/predict2/action/configs/action_conditioned/data.py) |
| Evaluation script | [`cosmos_h_surgical_simulator_quant_eval.py`](scripts/cosmos_h_surgical_simulator_quant_eval.py) |

## Upstream

This repository is a domain-specific fork of [Cosmos-Predict2.5](https://github.com/nvidia-cosmos/cosmos-predict2.5). The base model documentation (Text2World, Image2World, Video2World, distillation, multiview) remains available in [docs/](docs/).

## Contributing

We thrive on community collaboration! [NVIDIA-Cosmos](https://github.com/nvidia-cosmos/) wouldn't be where it is without contributions from developers like you. Check out our [Contributing Guide](CONTRIBUTING.md) to get started, and share your feedback through issues.

Big thanks 🙏 to everyone helping us push the boundaries of open-source physical AI for surgical robotics!

## License and Contact

This project will download and install additional third-party open source software projects. Review the license terms of these open source projects before use.

NVIDIA Cosmos source code is released under the [Apache 2 License](https://www.apache.org/licenses/LICENSE-2.0).

NVIDIA Cosmos models are released under the [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license). For a custom license, please contact [cosmos-license@nvidia.com](mailto:cosmos-license@nvidia.com).
