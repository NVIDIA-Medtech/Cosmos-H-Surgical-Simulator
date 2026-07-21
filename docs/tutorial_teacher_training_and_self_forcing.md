# From a Cosmos-H-Surgical-Simulator Teacher to a Real-Time Causal Student

This tutorial describes how to adapt the bidirectional
Cosmos-H-Surgical-Simulator (C-H-S-S) model to a new action-conditioned video
dataset and then distill it into a causal, streaming student with Self Forcing.
The final student can be deployed with Cosmos-H-Dreams for low-latency,
closed-loop generation.

The recipe is based on the JHU dVRK tabletop experiment used to train a
73-frame causal student:

1. Fine-tune a short-horizon, bidirectional teacher from C-H-S-S.
2. Fine-tune a long-horizon, bidirectional teacher from the short-horizon
   checkpoint.
3. Generate a Phase 0 cache of teacher denoising trajectories.
4. Warm up a causal student against the cached trajectories.
5. Run Self Forcing distillation.
6. Convert and deploy the causal student with Cosmos-H-Dreams.


