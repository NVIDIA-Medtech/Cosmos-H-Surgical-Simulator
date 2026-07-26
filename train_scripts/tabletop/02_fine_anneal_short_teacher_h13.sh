#!/bin/bash
# Reference stage 1b: 4k cosine fine anneal from short-teacher iter 16,000.
#SBATCH --job-name=tabletop-teacher-h13-anneal
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --time=4:00:00
#SBATCH --output=tabletop-teacher-h13-anneal_%A_%a.out
#SBATCH --error=tabletop-teacher-h13-anneal_%A_%a.out
#SBATCH --array=0-3%1
#SBATCH --dependency=singleton
#SBATCH --requeue

set -euo pipefail

: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to persistent storage}"
: "${TABLETOP_DATA_ROOT:?Set TABLETOP_DATA_ROOT to the LeRobot root}"
: "${CONTAINER_COSMOS25:?Set CONTAINER_COSMOS25 to the teacher container image}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SHORT_DCP="${OUTPUT_ROOT}/cosmos_predict2_action_conditioned/official_runs_vid2vid/cosmos_predict2p5_2B_action_conditioned_jhu_dvrk_mono_finetune_13frame_8nodes_release_oss/checkpoints/iter_000016000"
test -d "$SHORT_DCP" || { echo "Missing short-teacher DCP: $SHORT_DCP" >&2; exit 1; }

export MASTER_ADDR
MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | sed -n '1p')"
MOUNTS="${REPO_ROOT}:/workspace,${OUTPUT_ROOT}:/imaginaire_output"
MOUNTS="${MOUNTS},${TABLETOP_DATA_ROOT}:/datasets/jhu_tabletop"

srun --export=ALL \
    --container-image="$CONTAINER_COSMOS25" \
    --container-mounts="$MOUNTS" \
    --container-workdir=/workspace \
    bash -c '
        set -euo pipefail
        source .venv/bin/activate
        export IMAGINAIRE_OUTPUT_ROOT=/imaginaire_output
        export JHU_TABLETOP_DATA_ROOT=/datasets/jhu_tabletop
        NODE_RANK=${SLURM_NODEID:-0}
        NNODES=${SLURM_JOB_NUM_NODES:-1}
        torchrun \
            --nnodes="$NNODES" \
            --nproc_per_node=8 \
            --master_port=25001 \
            --master_addr="$MASTER_ADDR" \
            --node_rank="$NODE_RANK" \
            -m scripts.train \
            --config=cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py \
            -- \
            experiment=cosmos_predict2p5_2B_action_conditioned_jhu_dvrk_mono_finetune_13frame_8nodes_release_oss_fine_anneal_4k \
            checkpoint.save_iter=200 \
            ~dataloader_train.dataloaders
    '
