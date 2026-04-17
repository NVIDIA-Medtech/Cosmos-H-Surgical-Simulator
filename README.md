# Cosmos-H-Surgical-Simulator

[![License](https://img.shields.io/badge/Code-Apache_2.0-blue.svg)](LICENSE)
[![Weights](https://img.shields.io/badge/Weights-NVIDIA_Open_Model-green.svg)](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-yellow)](https://huggingface.co/nvidia/Cosmos-H-Surgical-Simulator)
[![Paper](https://img.shields.io/badge/arXiv-2511.00062-red.svg)](https://arxiv.org/abs/2511.00062)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)

Action-conditioned world simulation for surgical robotics, built on NVIDIA Cosmos Predict2.5 and fine-tuned on the [Open-H](https://huggingface.co/datasets/nvidia/Open-H) multi-embodiment surgical benchmark.

<p align="center">
  <img src="assets/cosmos-predict-diagram.png" alt="Cosmos-H-Surgical-Simulator architecture" width="600"/>
</p>

## Overview

Cosmos-H-Surgical-Simulator is a surgical-domain variant of [Cosmos Predict2.5](https://github.com/nvidia-cosmos/cosmos-predict2.5) that generates future video frames conditioned on robot actions and an initial surgical scene. Given a context frame from a surgical procedure and a sequence of action vectors describing instrument kinematics, the model predicts realistic video of the resulting surgical environment — enabling applications in surgical simulation, pre-operative planning, and robotic training. For video prediction and control-conditioned transfer without action conditioning, see the companion repo [Cosmos-H-Surgical](https://github.com/NVIDIA-Medtech/Cosmos-H-Surgical).

The model supports **9 distinct robot embodiments** across **10+ institutions**, with all embodiment action spaces mapped into a unified 44-dimensional action vector. The CMR Versius system serves as the primary embodiment. See [README_ACTION_SPACE.md](scripts/README_ACTION_SPACE.md) for the full action space design.

The underlying architecture is a flow-based diffusion transformer that unifies Text2World, Image2World, and Video2World generation into a single model, using Cosmos-Reason1 as the text encoder. This surgical variant extends the base Predict2.5 capabilities with domain-specific fine-tuning on real surgical video data.

## News

- **[April 2026]** — Bugfix: properly initialize action embedder MLPs (`action_embedder_B_D` / `action_embedder_B_3D`) after loading the base checkpoint, so their weights are no longer left at zero under FSDP meta-device materialization.
- **[March 2026]** — Released Cosmos-H-Surgical-Simulator for the [Open-H benchmark](#open-h-benchmark)

For Cosmos-Predict2.5 updates, see the [upstream changelog](https://github.com/NVIDIA/Cosmos-Predict2.5).

## Model Variants

| Model | Capability | Input | HuggingFace | License |
|-------|-----------|-------|-------------|---------|
| Cosmos-H-Surgical-Simulator | Surgical video generation and action-conditioned simulation | text, image, video, or action vectors | [Weights](https://huggingface.co/nvidia/Cosmos-H-Surgical-Simulator) | [NVIDIA Open Model](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/) |

## Supported Embodiments

| Embodiment | Robot | Action Dim | Institution |
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

All embodiments are zero-padded to 44D at the model input layer. See [`README_ACTION_SPACE.md`](scripts/README_ACTION_SPACE.md) for per-embodiment mapping details.

## Open-H Benchmark

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

## Quick Start

```bash
# Clone and install
git clone git@github.com:nvidia-cosmos/Cosmos-H-Surgical-Simulator.git
cd Cosmos-H-Surgical-Simulator
git lfs pull

# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# Create environment and install dependencies
uv sync --extra=cu128
source .venv/bin/activate

# Configure Hugging Face for checkpoint downloads
uv tool install -U "huggingface_hub[cli]"
hf auth login
```

### Inference

```bash
# Run action-conditioned surgical simulation
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python cosmos_predict2/_src/predict2/action/inference/inference_open_h.py \
  --experiment cosmos_predict2p5_2B_action_conditioned_open_h-fixed_13frame_8nodes_release_oss \
  --ckpt_path /path/to/model_ema_bf16.pt \
  --dataset_path /path/to/dataset \
  --embodiment cmr_versius \
  --episode_ids 0,1,2
```

For detailed inference options and multi-embodiment examples, see the [Surgical Inference Guide](docs/inference_surgical.md).

## Documentation

| Guide | Description |
|-------|-------------|
| [Setup](docs/setup.md) | Full installation guide (uv, Docker, Blackwell) |
| [Surgical Inference](docs/inference_surgical.md) | Multi-embodiment and CMR-only inference with Open-H |
| [Surgical Post-Training](docs/post-training_surgical.md) | Open-H multi-embodiment, CMR-only, and downstream fine-tuning |
| [Evaluation](scripts/README_EVALUATION.md) | Quantitative evaluation pipeline (FDS, GATC, TCD metrics) |
| [Action Space](scripts/README_ACTION_SPACE.md) | Unified 44D action space for multi-embodiment training |
| [Codebase Guide](docs/codebase-guide.md) | Key source files, data flow, and adding new embodiments |
| [Distillation](docs/distillation.md) | Model compression via DMD2 distillation |
| [Troubleshooting](docs/troubleshooting.md) | Common issues and solutions |
| [Upstream Docs](docs/) | Base model guides (Text2World, Image2World, Video2World, multiview) |

## Evaluation

The quantitative evaluation pipeline benchmarks checkpoints across the full Open-H benchmark with three metrics:

| Metric | What it measures | Direction |
|---|---|---|
| FDS (Frame Decay Score) | Pixel-level fidelity degradation over time (mean L1) | Lower is better |
| GATC (GT-Anchored Tool Consistency) | Surgical tool visual consistency vs. ground truth | Higher is better (range: -1 to 1) |
| TCD (Tool Centroid Distance) | Spatial displacement of tool instances (pixels) | Lower is better |

GATC and TCD require [Medical-SAM3](https://github.com/AIM-Research-Lab/Medical-SAM3) fine-tuned on [CholecSeg8k](https://www.kaggle.com/datasets/newslab/cholecseg8k) for surgical tool segmentation. See [`README_EVALUATION.md`](scripts/README_EVALUATION.md) for full metric definitions and evaluation pipeline documentation.

## License

| Component | License |
|-----------|---------|
| Source code | [Apache 2.0](LICENSE) |
| Cosmos-H-Surgical-Simulator weights | [NVIDIA Open Model](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/) |

This project downloads and installs additional third-party open source software. Review the license terms of these projects before use. See [ATTRIBUTIONS.md](ATTRIBUTIONS.md).

## Resources

- [Cosmos Predict2.5 Paper](https://arxiv.org/abs/2511.00062) — Base model paper
- [Open-H Dataset](https://huggingface.co/datasets/nvidia/Open-H) — Multi-embodiment surgical benchmark
- [HuggingFace](https://huggingface.co/nvidia/Cosmos-H-Surgical-Simulator) — Model weights and checkpoints
- [Cosmos-H-Surgical](https://github.com/NVIDIA-Medtech/Cosmos-H-Surgical) — Sister repo (predict + transfer)
- [Cosmos Predict2.5](https://github.com/nvidia-cosmos/cosmos-predict2.5) — Upstream base model
- [NVIDIA Cosmos Platform](https://www.nvidia.com/en-us/ai/cosmos) — Product website

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on reporting bugs and submitting changes.
