#!/bin/bash
# Reference stage 3: one-node, eight-rank Phase 0 teacher cache generation.
#SBATCH --job-name=tabletop-phase0-h73
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --time=4:00:00
#SBATCH --output=tabletop-phase0-h73_%A_%a.out
#SBATCH --error=tabletop-phase0-h73_%A_%a.out
#SBATCH --array=0-3%1
#SBATCH --dependency=singleton
#SBATCH --requeue

set -euo pipefail

: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to persistent storage}"
: "${CACHE_ROOT:?Set CACHE_ROOT to persistent cache storage}"
: "${TABLETOP_DATA_ROOT:?Set TABLETOP_DATA_ROOT to the LeRobot root}"
: "${CONTAINER_COSMOS25:?Set CONTAINER_COSMOS25 to the teacher container image}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TEACHER_RELATIVE="cosmos_predict2_action_conditioned/official_runs_vid2vid/cosmos_predict2p5_2B_action_conditioned_jhu_dvrk_mono_finetune_13frame_8nodes_release_oss_h73_tabletop/checkpoints/iter_000005000/model_ema_bf16.pt"
test -f "${OUTPUT_ROOT}/${TEACHER_RELATIVE}" || {
    echo "Missing consolidated teacher checkpoint: ${OUTPUT_ROOT}/${TEACHER_RELATIVE}" >&2
    exit 1
}

export TOTAL_SAMPLES="${TOTAL_SAMPLES:-10000}"
export SAMPLE_STRATEGY="${SAMPLE_STRATEGY:-random}"
export INDICES_SEED="${INDICES_SEED:-0}"
export TEACHER_CKPT="/imaginaire_output/${TEACHER_RELATIVE}"
export EXPERIMENT="cosmos_predict2p5_2B_action_conditioned_jhu_dvrk_mono_finetune_13frame_8nodes_release_oss_h73_tabletop"
export SAVE_ROOT="datasets/jhu_dvrk_mono_warmup_4step_h73_tabletop"

MOUNTS="${REPO_ROOT}:/workspace,${OUTPUT_ROOT}:/imaginaire_output"
MOUNTS="${MOUNTS},${CACHE_ROOT}:/imaginaire_cache"
MOUNTS="${MOUNTS},${TABLETOP_DATA_ROOT}:/datasets/jhu_tabletop"

srun --export=ALL \
    --container-image="$CONTAINER_COSMOS25" \
    --container-mounts="$MOUNTS" \
    --container-workdir=/workspace \
    bash -c '
        set -euo pipefail
        source .venv/bin/activate
        export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
        export JHU_TABLETOP_DATA_ROOT=/datasets/jhu_tabletop
        export IMAGINAIRE_CACHE_DIR=/imaginaire_cache
        export HF_HOME=/imaginaire_cache/huggingface

        N_RANKS=8
        SAMPLES_PER_RANK=$(( (TOTAL_SAMPLES + N_RANKS - 1) / N_RANKS ))
        START=$(( SLURM_LOCALID * SAMPLES_PER_RANK ))
        END=$(( (SLURM_LOCALID + 1) * SAMPLES_PER_RANK ))
        (( END > TOTAL_SAMPLES )) && END=$TOTAL_SAMPLES

        python cosmos_predict2/_src/predict2/action/inference/inference_jhu_dvrk_warmup.py \
            --experiment "$EXPERIMENT" \
            --ckpt_path "$TEACHER_CKPT" \
            --save_root "$SAVE_ROOT" \
            --resolution 288,512 \
            --guidance 0 \
            --num_frames 73 \
            --chunk_size 72 \
            --sample_strategy "$SAMPLE_STRATEGY" \
            --total_samples "$TOTAL_SAMPLES" \
            --indices_seed "$INDICES_SEED" \
            --start "$START" \
            --end "$END" \
            --query_steps 0,9,18,27,34
    '
