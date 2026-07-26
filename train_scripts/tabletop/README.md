# Tabletop teacher-to-student SLURM templates

These templates reproduce the tabletop reference pipeline described in [`docs/tutorial_teacher_training_and_self_forcing.md`](../../docs/tutorial_teacher_training_and_self_forcing.md).
They use Pyxis/Enroot-style `srun --container-*` options; adapt those options if your cluster uses another container runtime.

Before submitting, create the output directories and export host paths:

```bash
export OUTPUT_ROOT=/persistent/path/imaginaire/output
export CACHE_ROOT=/persistent/path/imaginaire/cache
export TABLETOP_DATA_ROOT=/persistent/path/jhu_tabletop_lerobot
export CHSS_CHECKPOINT_DIR=/persistent/path/cosmos_h_surgical_simulator_dcp
export CONTAINER_COSMOS25=/path/to/cosmos-predict-2.5.sqsh
export CONTAINER_CAUSAL=/path/to/image-with-natten.sqsh
mkdir -p "$OUTPUT_ROOT" "$CACHE_ROOT"
```

`TABLETOP_DATA_ROOT` must contain the nine directories listed by `JHU_DVRK_MONO_FINETUNE_TRAIN_DATASET_SPECS`. Add your cluster's `#SBATCH --account` and `#SBATCH --partition` directives locally if required.

Run order:

```bash
sbatch train_scripts/tabletop/01_train_short_teacher_h13.sh
sbatch train_scripts/tabletop/02_fine_anneal_short_teacher_h13.sh
sbatch train_scripts/tabletop/03_train_long_teacher_h73.sh

# Convert long-teacher iter_000005000/model to model_ema_bf16.pt first.
sbatch train_scripts/tabletop/04_phase0_teacher_cache_h73.sh

sbatch train_scripts/tabletop/05_warmup_student_h73.sh
sbatch train_scripts/tabletop/06_self_forcing_h73.sh
```

Keep these invariants synchronized:

- short teacher: 13 frames, `state_t=4`, selected iteration 16,000;
- fine anneal: 4,000 iterations;
- long teacher: 73 frames, `state_t=19`, selected iteration 5,000;
- Phase 0: 10,000 randomly selected samples, 72 actions, 73 frames;
- warmup: 20,000-iteration ceiling; reference SF initialization at 18,000;
- Self Forcing: 3,000 iterations.

The Phase 0 script is resumable because existing complete artifact quartets are skipped. Before warmup, verify that `latents/`, `images/`, `actions/`, and `videos/` contain the same index set; do not rely only on SLURM completion.
