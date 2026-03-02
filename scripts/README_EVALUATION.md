# Quantitative Evaluation for Cosmos-H-Surgical-Simulator

This document describes the quantitative evaluation pipeline implemented in
[`cosmos_h_surgical_simulator_quant_eval.py`](./cosmos_h_surgical_simulator_quant_eval.py).
The script benchmarks action-conditioned video generation checkpoints of the
Cosmos Predict-2.5 world model on CMR Surgical Versius robotic surgery data.

## Overview

For each checkpoint under evaluation the script:

1. **Generates videos** autoregressively (6 chunks of 12 frames = 72 generated
   frames per episode) conditioned on ground-truth actions and an initial frame.
2. **Computes three complementary metrics** that capture different quality
   dimensions of the generated video relative to the ground-truth.
3. **Reports aggregated results** across datasets, episodes, and seeds in a
   structured log and a JSON file suitable for downstream analysis and plotting.

### Evaluation Matrix (defaults)

| Parameter | Default |
|---|---|
| Datasets | 4 (prostatectomy, inguinal hernia, hysterectomy, cholecystectomy) |
| Episodes per dataset | 5 |
| Seeds per episode | 3 |
| **Total evaluations per checkpoint** | **Up to 60** (default configuration) |

### Two-Phase Execution

To avoid GPU memory pressure from loading both the Cosmos world model and the
Medical-SAM3 segmentation model simultaneously, the script operates in two
sequential phases:

- **Phase 1** — Load the Cosmos checkpoint, generate all videos, compute Frame
  Decay Score (FDS).  Store video arrays in CPU memory.
- **Phase 2** — Unload Cosmos, load Medical-SAM3, compute GATC and TCD on the
  stored video pairs.

---

## Metrics

Three metrics are computed for every episode evaluation (a ground-truth video
paired with a generated video).  Results are then aggregated across episodes
and seeds.

### 1. Frame Decay Score (FDS)

**Purpose.**  Measure pixel-level fidelity degradation over the generated
video horizon.

**Definition.**  Let \( G_t \) and \( \hat{G}_t \) denote the ground-truth and
generated frames at time \( t \), both normalised to \([-1, 1]\) (i.e.,
\( x \mapsto x / 127.5 - 1 \)).  The conditioning frame (\( t = 0 \)) is
excluded from scoring.

Per-frame L1 distance:

$$
\text{L1}_t = \frac{1}{H \cdot W \cdot 3} \sum_{i,j,c} \left| \hat{G}_t^{(i,j,c)} - G_t^{(i,j,c)} \right|, \qquad t = 1, \ldots, T{-}1
$$

The **FDS scalar** is the mean over all generated frames:

$$
\text{FDS} = \frac{1}{T{-}1} \sum_{t=1}^{T-1} \text{L1}_t
$$

Lower is better.  An accompanying Mean SSIM score (higher is better) is also
reported.

**Per-chunk breakdown.**  To diagnose whether quality differences between
checkpoints emerge early or are masked by late-frame saturation, FDS is also
reported for three temporal phases:

| Phase | Frames | Autoregressive chunks |
|---|---|---|
| Early | 1 – 12 | Chunk 1 |
| Mid | 13 – 36 | Chunks 2–3 |
| Late | 37 – 72 | Chunks 4–6 |

---

### 2. GT-Anchored Tool Consistency (GATC)

**Purpose.**  Measure whether surgical tools in the generated video remain
visually consistent with the ground-truth tool locations, tolerating small
spatial and temporal shifts.

**Requires:** Medical-SAM3 tool segmentation masks on ground-truth frames.

**Definition.**  For each frame \( t \):

1. Obtain the binary tool mask \( M_t \) from the ground-truth frame using
   Medical-SAM3, and dilate it by \( r \) pixels (default \( r = 10 \)):
   \( \tilde{M}_t = \text{dilate}(M_t, r) \).

2. Compute the representation for ZNCC comparison.  By default, grayscale
   intensity is used (configurable to gradient magnitude via `--gatc_use_grad`):

$$
X_t = \text{gray}(G_t), \qquad Y_t = \text{gray}(\hat{G}_t)
$$

3. Compute per-frame consistency as the best ZNCC over temporal offsets
   \( \Delta t \in \{-1, 0, +1\} \) and spatial translations
   \( (\Delta x, \Delta y) \in [-k, k]^2 \) (default \( k = 3 \)):

$$
s_t = \max_{\Delta t, \Delta x, \Delta y} \; \text{ZNCC}\!\left(X_t,\; \text{shift}(Y_{t+\Delta t}, \Delta x, \Delta y) \;\big|\; \tilde{M}_t\right)
$$

   where the masked ZNCC is:

$$
\text{ZNCC}(a, b \mid m) = \frac{\sum_{m} (a - \bar{a}_m)(b - \bar{b}_m)}{\sqrt{\sum_{m}(a - \bar{a}_m)^2} \cdot \sqrt{\sum_{m}(b - \bar{b}_m)^2} + \varepsilon}
$$

4. Compute the tool-presence penalty:

$$
p_t = \min\!\left(1,\; \frac{\mathbb{E}[\|\nabla \hat{G}_t\| \mid \tilde{M}_t]}{\mathbb{E}[\|\nabla G_t\| \mid \tilde{M}_t] + \varepsilon}\right)
$$

   This penalises frames where the generated image lacks edge activity in the
   tool region (i.e., the tool has "disappeared" or become blurry).

5. The per-frame GATC score is:

$$
s'_t = s_t \cdot p_t
$$

6. The **GATC scalar** is the median of \( s'_t \) over valid frames (frames
   where \( |\tilde{M}_t| \geq 50 \) pixels):

$$
\text{GATC} = \text{median}_{t \in \mathcal{V}} \; s'_t
$$

Score range: \([-1, 1]\).  Higher is better (\(1.0\) = perfect tool
consistency).

---

### 3. Tool Centroid Distance (TCD)

**Purpose.**  Measure the spatial displacement between ground-truth and
generated tool instances.

**Requires:** Medical-SAM3 tool instance segmentation on both ground-truth and
generated frames.

**Definition.**  For each frame \( t \):

1. Segment both the ground-truth and generated frames into tool instances using
   Medical-SAM3.  Compute the centroid of each instance:

$$
C_t^{\text{gt}} = \{c_1^{\text{gt}}, \ldots, c_N^{\text{gt}}\}, \qquad
C_t^{\text{gen}} = \{c_1^{\text{gen}}, \ldots, c_M^{\text{gen}}\}
$$

2. Build the \( N \times M \) cost matrix of Euclidean distances:

$$
D_{ij} = \|c_i^{\text{gt}} - c_j^{\text{gen}}\|_2
$$

3. Find the minimum-cost bipartite matching via the Hungarian algorithm over
   \( \min(N, M) \) pairs.

4. Penalise unmatched ground-truth tools with a miss penalty
   \( D_{\text{miss}} = 0.5 \cdot \sqrt{H^2 + W^2} \) (half the image
   diagonal):

$$
\text{TCD}_t = \frac{\sum_{\text{matched}} D_{ij} + (N - |\text{matched}|) \cdot D_{\text{miss}}}{N}
$$

   Frames where \( N = 0 \) (no GT tools) are excluded.

5. The **TCD scalar** is the median over valid frames:

$$
\text{TCD} = \text{median}_{t \in \mathcal{V}} \; \text{TCD}_t
$$

In pixels, lower is better (\(0\) = perfect spatial alignment of all tools).

---

## Medical-SAM3 Dependency

The GATC and TCD metrics require tool segmentation masks produced by
[Medical-SAM3](https://github.com/AIM-Research-Lab/Medical-SAM3), a foundation
model for prompt-driven medical image segmentation built on top of
[SAM3](https://github.com/AIM-Research-Lab/SAM3).

### Setup

Medical-SAM3 must be installed and configured separately:

1. Clone the repository:
   ```bash
   git clone https://github.com/AIM-Research-Lab/Medical-SAM3.git
   ```

2. Install dependencies as described in the Medical-SAM3 README.

3. Ensure the SAM3 backbone is available (Medical-SAM3 depends on SAM3).

4. **Fine-tune on CholecSeg8k.**  The pretrained Medical-SAM3 weights are
   trained on general medical imaging datasets and do not perform adequate tool
   segmentation on laparoscopic surgical video out of the box.  To obtain
   reliable tool masks on CMR Surgical Versius data, the model must be
   fine-tuned on the publicly available
   [CholecSeg8k](https://www.kaggle.com/datasets/newslab/cholecseg8k) dataset,
   which provides pixel-level annotations of surgical tools, anatomy, and
   background in cholecystectomy procedures.  After fine-tuning, the resulting
   checkpoint is passed to this evaluation script via `--sam3_checkpoint`.

5. Ensure `sam3_inference.py` (from the Medical-SAM3 `inference/` directory) is
   on `PYTHONPATH` at runtime.

6. Ensure the loaded Medical-SAM3 runtime exposes a compatible processor API
   with `processor.set_text_prompt` after model initialization (`load_model()`),
   which is required by this evaluation script.

---

## Usage

### Single checkpoint

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/cosmos_h_surgical_simulator_quant_eval.py \
  --ckpt_path /path/to/model_ema_bf16.pt \
  --sam3_checkpoint /path/to/medical_sam3_cholecseg8k.pt
```

### Multiple checkpoints (compared in final report)

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/cosmos_h_surgical_simulator_quant_eval.py \
  --ckpt_path /path/to/ckpt_10k.pt /path/to/ckpt_20k.pt \
  --ckpt_labels "10k-steps" "20k-steps" \
  --sam3_checkpoint /path/to/medical_sam3_cholecseg8k.pt
```

### Quick debug run

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/cosmos_h_surgical_simulator_quant_eval.py \
  --ckpt_path /path/to/model.pt \
  --sam3_checkpoint /path/to/medical_sam3_cholecseg8k.pt \
  --num_episodes 1 --num_seeds 1
```

### Key arguments

| Argument | Default | Description |
|---|---|---|
| `--ckpt_path` | *(required)* | One or more Cosmos checkpoint `.pt` files |
| `--sam3_checkpoint` | *(required)* | Medical-SAM3 checkpoint (fine-tuned on CholecSeg8k) |
| `--experiment` | `cosmos_predict2p5_2B_...` | Experiment config name |
| `--num_episodes` | 5 | Episodes per dataset (1–5) |
| `--num_seeds` | 3 | Seeds per evaluation |
| `--seed` | 0 | Base seed |
| `--gatc_k` | 3 | GATC translation search radius in pixels |
| `--gatc_use_grad` | off | Use gradient magnitude instead of grayscale for GATC |
| `--save_videos` | off | Save generated and comparison videos as MP4 |
| `--save_path` | `output/quant_eval_cmr` | Output directory |

---

## Output

### Log report

At the end of evaluation, a structured report is printed to the log containing:

- Per-checkpoint summary: FDS (mean L1 +/- std), GATC (median +/- std),
  TCD (median +/- std), per-chunk breakdowns.
- Per-dataset breakdown table.
- Cross-checkpoint comparison table and rankings (when multiple checkpoints
  are evaluated).
- CSV-formatted blocks for direct copy-paste into spreadsheets.

### JSON file

A timestamped JSON file is saved to `--save_path` containing per-episode
**aggregated** scores and checkpoint-level aggregated statistics for
programmatic analysis.

### Plotting

Use [`plot_quant_eval_results.py`](./plot_quant_eval_results.py)
to generate comparison plots from the JSON output or from hardcoded results:

```bash
python scripts/plot_quant_eval_results.py --json output/quant_eval_cmr/quant_eval_results_*.json
```

This produces:
- Metric-over-iteration line plots (FDS, GATC, TCD).
- Bar chart comparisons.
- Per-chunk (Early/Mid/Late) 3-panel plots for each metric.
