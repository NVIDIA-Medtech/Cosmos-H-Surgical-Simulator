#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Reference stage 2: 73-frame teacher warm-started from the annealed h13 teacher.
#SBATCH --job-name=tabletop-teacher-h73
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --time=4:00:00
#SBATCH --output=tabletop-teacher-h73_%A_%a.out
#SBATCH --error=tabletop-teacher-h73_%A_%a.out
#SBATCH --array=0-9%1
#SBATCH --dependency=singleton
#SBATCH --requeue

set -euo pipefail

: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to persistent storage}"
: "${TABLETOP_DATA_ROOT:?Set TABLETOP_DATA_ROOT to the LeRobot root}"
: "${CONTAINER_COSMOS25:?Set CONTAINER_COSMOS25 to the teacher container image}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ANNEALED_DCP="${OUTPUT_ROOT}/cosmos_predict2_action_conditioned/official_runs_vid2vid/cosmos_predict2p5_2B_action_conditioned_jhu_dvrk_mono_finetune_13frame_8nodes_release_oss_fine_anneal_4k/checkpoints/iter_000004000"
test -d "$ANNEALED_DCP" || { echo "Missing annealed teacher DCP: $ANNEALED_DCP" >&2; exit 1; }
mkdir -p "$OUTPUT_ROOT"

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
            experiment=cosmos_predict2p5_2B_action_conditioned_jhu_dvrk_mono_finetune_13frame_8nodes_release_oss_h73_tabletop \
            checkpoint.save_iter=200 \
            ~dataloader_train.dataloaders
    '
