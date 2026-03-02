# Architecture & Technical Details

Deep technical documentation for the GR00T BC + Isaac Lab evaluation pipeline. For setup and usage, see [README.md](README.md).

---

## BC Training Pipeline

1. `wandb.agent` starts a Bayesian sweep
2. Each trial runs `launch_finetune.py` from the Isaac-GR00T repo
3. Training uses DDP across 8 GPUs with gradient checkpointing
4. The `flash_attn` library is patched to use `sdpa` (L40 GPUs lack flash_attn_2)
5. Best checkpoint is uploaded as a W&B artifact

## Evaluation Pipeline

1. Eval container polls W&B for new `groot-bc-g1-trial` artifacts
2. Downloads the checkpoint and loads `Gr00tPolicy`
3. For each episode:
   - Sim resets the environment (G1 robot + table + cube)
   - At each step: `obs → G1JointMapper.obs_to_groot() → policy.get_action() → G1JointMapper.action_chunk_from_groot(actions, current_proprio) → sim.step()`
   - The mapper handles the RELATIVE/ABSOLUTE action split: arm and leg deltas are added to current joint positions before being sent as absolute targets
   - Camera frames are collected for video
4. Videos and metrics are logged back to the original sweep trial's W&B run

## GR00T Model Details

- **Architecture**: EagleBackbone (vision-language encoder) + DiT action head (flow matching)
- **Action head**: Flow matching with linear interpolation (`x_t = (1-t)*noise + t*actions`), velocity prediction, Euler integration at inference
- **Action horizon**: 16 timesteps per query
- **Custom embodiment**: Registered as `NEW_EMBODIMENT` via `register_modality_config()`

## G1 Robot Joint Mapping

The GR00T model was trained on 43-DOF teleop data, but Isaac Lab's G1 has 37 DOFs:

| Body Part | Training DOFs | Sim DOFs | Padding | Action Type |
|-----------|--------------|----------|---------|-------------|
| waist | 3 | 1 | 1→3 | ABSOLUTE |
| left_arm | 7 | 5 | 5→7 | RELATIVE |
| right_arm | 7 | 5 | 5→7 | RELATIVE |
| left_hand | 7 | 7 | — | ABSOLUTE |
| right_hand | 7 | 7 | — | ABSOLUTE |
| left_leg | 6 | 6 | — | RELATIVE |
| right_leg | 6 | 6 | — | RELATIVE |

`G1JointMapper` handles the padding (state) and trimming (actions) automatically.
