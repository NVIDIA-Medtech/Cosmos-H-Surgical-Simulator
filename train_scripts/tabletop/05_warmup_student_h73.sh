#!/bin/bash
# Reference stage 4: 8-node causal-student warmup, state_t=19.
#SBATCH --job-name=tabletop-warmup-h73
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --time=4:00:00
#SBATCH --output=tabletop-warmup-h73_%A_%a.out
#SBATCH --error=tabletop-warmup-h73_%A_%a.out
#SBATCH --array=0-9%1
#SBATCH --dependency=singleton
#SBATCH --requeue

set -euo pipefail

: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to persistent storage}"
: "${CACHE_ROOT:?Set CACHE_ROOT to persistent cache storage}"
: "${CONTAINER_CAUSAL:?Set CONTAINER_CAUSAL to an image containing NATTEN}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PHASE0_ROOT="${REPO_ROOT}/datasets/jhu_dvrk_mono_warmup_4step_h73_tabletop"
for artifact in latents images actions videos; do
    test -d "${PHASE0_ROOT}/${artifact}" || {
        echo "Missing Phase 0 artifact directory: ${PHASE0_ROOT}/${artifact}" >&2
        exit 1
    }
done
mkdir -p "$OUTPUT_ROOT" "$CACHE_ROOT"

export MASTER_ADDR
MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | sed -n '1p')"
export MASTER_PORT=25002
export WORLD_SIZE=$SLURM_NTASKS

MOUNTS="${REPO_ROOT}:/workspace,${OUTPUT_ROOT}:/imaginaire_output"
MOUNTS="${MOUNTS},${CACHE_ROOT}:/imaginaire_cache"

srun --export=ALL \
    --container-image="$CONTAINER_CAUSAL" \
    --container-mounts="$MOUNTS" \
    --container-workdir=/workspace \
    bash -c '
        set -euo pipefail
        source .venv/bin/activate
        export RANK=$SLURM_PROCID
        export LOCAL_RANK=$SLURM_LOCALID
        export IMAGINAIRE_OUTPUT_ROOT=/imaginaire_output
        export IMAGINAIRE_CACHE_DIR=/imaginaire_cache
        export HF_HOME=/imaginaire_cache/huggingface
        export TORCH_HOME=/imaginaire_cache/torch
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        python -c "import natten"
        python -m scripts.train \
            --config=cosmos_predict2/_src/predict2/interactive/configs/config_warmup.py \
            -- \
            experiment=cosmos_predict2p5_2B_action_jhu_dvrk_mono_tabletop_h73_warmup_no_s3_resumable \
            checkpoint.save_iter=200
    '
