# Fine-Tuning GR00T N1.6 VLA with Isaac Lab Closed-Loop Evaluation on Kubernetes

This guide explains how to run an end-to-end **Behavioral Cloning (BC) sweep + Isaac Lab closed-loop evaluation** pipeline for NVIDIA's GR00T N1.6-3B Vision-Language-Action model on a Kubernetes GPU cluster with Weights & Biases tracking.

This setup supports:

- Bayesian hyperparameter sweep for BC fine-tuning (8× L40 GPUs)
- Automated closed-loop evaluation in Isaac Lab simulation (2× L40 GPUs)
- Two-container pod architecture (sim + eval) with gRPC communication
- Automatic W&B logging: training curves, rollout videos, model artifacts
- Continuous evaluation: new checkpoints are picked up and evaluated automatically

## See this pipeline running live on W&B: [GR00T VLA + Isaac Lab on CoreWeave](https://wandb.ai/wandb-smle/isaacsim-nvidia-vla-crwv)

---

# Documentation References

- GR00T (Isaac-GR00T) https://github.com/NVIDIA/Isaac-GR00T
- Isaac Lab https://github.com/isaac-sim/IsaacLab
- Isaac Sim https://docs.omniverse.nvidia.com/isaacsim/latest/index.html
- Weights & Biases https://docs.wandb.ai
- Kubernetes https://kubernetes.io/docs/home/
- NVIDIA NGC https://ngc.nvidia.com

---

# Architecture Overview

The pipeline has two stages that run concurrently on separate pods:

```
┌─────────────────────────────────────────────────────────────────────┐
│                     BC Sweep Training Pod                          │
│                     groot-bc-g1-0 (8× L40)                        │
│                                                                     │
│  wandb.agent (Bayesian sweep)                                       │
│    ├── Trial 1: lr=1.09e-4, grad_accum=64, steps=20000             │
│    ├── Trial 2: lr=2.3e-4,  grad_accum=32, steps=15000             │
│    └── ...                                                          │
│                                                                     │
│  Each trial:                                                        │
│    1. Fine-tune GR00T N1.6-3B on 311 G1 teleop episodes            │
│    2. Upload best checkpoint as W&B artifact (groot-bc-g1-trial)    │
└────────────────────────┬────────────────────────────────────────────┘
                         │ W&B Artifacts
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│               Isaac Lab Evaluation Pod                              │
│               groot-isaaclab-eval-0 (2× L40)                        │
│                                                                     │
│  ┌──────────────────┐    gRPC    ┌────────────────────────┐         │
│  │  Sim Container   │◄─────────►│   Eval Container       │         │
│  │  (Isaac Lab 2.3.2)│  :7000    │   (GR00T + eval)       │         │
│  │                  │            │                        │         │
│  │  G1 robot        │  obs/act   │  Poll W&B for new      │         │
│  │  + table + cube  │◄──────────►│  groot-bc-g1-trial     │         │
│  │  + camera        │            │  artifacts             │         │
│  │                  │            │                        │         │
│  │  JointPosition   │            │  Load GR00T policy     │         │
│  │  ActionCfg       │            │  Run 3 rollout eps     │         │
│  │                  │            │  Log videos to W&B     │         │
│  └──────────────────┘            └────────────────────────┘         │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

- **Two-container pods**: Isaac Lab requires Python 3.11 (bundled in `nvcr.io/nvidia/isaac-lab:2.3.2`), while GR00T requires Python 3.10. The containers share a pod and communicate over localhost gRPC.
- **Fixed-base robot**: The G1 robot's root link is fixed (`fix_root_link = True`) for tabletop manipulation evaluation.
- **Joint position control**: GR00T outputs per-body-part joint positions/deltas, which are mapped to Isaac Lab's flat 37-DOF action space via `G1JointMapper`.
- **RELATIVE vs ABSOLUTE actions**: Arms and legs use relative (delta) actions, while waist and hands use absolute targets. The mapper converts relative deltas to absolute positions by adding them to the current joint state.

---

# 1. Prerequisites

## Kubernetes Cluster

A GPU cluster with NVIDIA L40 (or equivalent) GPUs. This guide uses CoreWeave.

```bash
kubectl config get-contexts
kubectl config use-context <your-cluster-context>
```

## NVIDIA NGC Access

Container image: `nvcr.io/nvidia/isaac-lab:2.3.2`

```bash
kubectl create secret docker-registry nvcr-secret \
  --docker-server=nvcr.io \
  --docker-username='$oauthtoken' \
  --docker-password='<YOUR_NGC_API_KEY>'
```

## Weights & Biases Setup

```bash
kubectl create secret generic wandb-api-key \
  --from-literal=WANDB_API_KEY=<YOUR_WANDB_API_KEY>
```

## Base Model

Upload the GR00T N1.6-3B base model as a W&B artifact:

```
wandb-smle/isaacsim-nvidia-vla-crwv/groot-n1.6-3b:v1
```

## Teleop Dataset

Upload the G1 teleop dataset (311 episodes) as a W&B artifact:

```
wandb-smle/isaacsim-nvidia-vla-crwv/groot-teleop-unitree-g1:v0
```

---

# 2. Running the Pipeline

## Stage 1: BC Sweep Training

Launches a Bayesian hyperparameter sweep that fine-tunes GR00T on G1 teleop data.

```bash
kubectl apply -f groot-bc-unitree-g1.yaml
```

This creates a single pod (`groot-bc-g1-0`) with 8× L40 GPUs running DDP training. The sweep explores:

| Parameter | Range |
|-----------|-------|
| Learning rate | 5e-5 — 3e-4 |
| Gradient accumulation | 16, 32, 64 |
| Max training steps | 10,000 — 30,000 |

Each trial:
1. Fine-tunes GR00T N1.6-3B with LoRA (PEFT)
2. Logs training loss to W&B
3. Uploads the best checkpoint as artifact `groot-bc-g1-trial`

## Stage 2: Isaac Lab Closed-Loop Evaluation

Automatically evaluates every BC checkpoint in simulation.

```bash
kubectl apply -f groot-isaaclab-eval.yaml
```

This creates a two-container pod (`groot-isaaclab-eval-0`) with 2× L40 GPUs:

- **Sim container** (GPU 0): Runs Isaac Lab with the `Isaac-G1-ManipJointCtrl-v0` environment — a Unitree G1 robot at a table with a red cube
- **Eval container** (GPU 1): Polls W&B for new `groot-bc-g1-trial` artifacts, downloads each checkpoint, runs 3 rollout episodes (3000 steps each), and logs videos + metrics back to W&B

The eval loop runs continuously — as the BC sweep produces new checkpoints, they are automatically picked up and evaluated.

---

# 3. Monitoring

## Pod Status

```bash
kubectl get pods
```

Expected output:
```
NAME                    READY   STATUS    AGE
groot-bc-g1-0           1/1     Running   10d
groot-isaaclab-eval-0   2/2     Running   4d
```

## Logs

```bash
# BC sweep training
kubectl logs groot-bc-g1-0 --tail=20

# Eval — sim container
kubectl logs groot-isaaclab-eval-0 -c sim --tail=20

# Eval — eval container
kubectl logs groot-isaaclab-eval-0 -c eval --tail=20
```

## W&B Dashboard

- **Project**: [`wandb-smle/isaacsim-nvidia-vla-crwv`](https://wandb.ai/wandb-smle/isaacsim-nvidia-vla-crwv)
- **Sweep runs**: Training loss curves for each trial
- **Eval metrics** (logged to each trial's original run):
  - `eval/mean_reward` — mean episode reward
  - `eval/mean_episode_length` — mean episode length
  - `eval/termination_rate` — fraction of episodes terminated early
  - `eval/checkpoint` — which checkpoint was evaluated
  - Rollout videos attached as `wandb.Video` media

---

# 4. How It Works

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

---

# 5. Configuration

## BC Sweep (`groot-bc-unitree-g1.yaml`)

| Variable | Default | Description |
|----------|---------|-------------|
| `WANDB_ENTITY` | `wandb-smle` | W&B entity |
| `WANDB_PROJECT` | `isaacsim-nvidia-vla-crwv` | W&B project |
| `BASE_MODEL_ARTIFACT` | `groot-n1.6-3b:v1` | Base GR00T model |
| `DATASET_ARTIFACT` | `groot-teleop-unitree-g1:v0` | Teleop training data |
| GPU allocation | 8× L40 | DDP training |

## Eval (`groot-isaaclab-eval.yaml`)

| Variable | Default | Description |
|----------|---------|-------------|
| `ISAAC_TASK` | `Isaac-G1-ManipJointCtrl-v0` | Isaac Lab environment |
| `TASK_DESCRIPTION` | `Pick up the apple and place it on the plate` | Language prompt |
| `N_EPISODES` | `3` | Rollout episodes per checkpoint |
| `MAX_STEPS` | `3000` | Max steps per episode |
| Sim GPU | 1× L40 | Isaac Lab simulation |
| Eval GPU | 1× L40 | GR00T inference |

---

# 6. Cleanup

```bash
# Stop the BC sweep
kubectl delete -f groot-bc-unitree-g1.yaml

# Stop the eval pipeline
kubectl delete -f groot-isaaclab-eval.yaml

# Or scale down without deleting (preserves config)
kubectl scale statefulset groot-bc-g1 --replicas=0
kubectl scale statefulset groot-isaaclab-eval --replicas=0
```

---

# 7. Files

| File | Description |
|------|-------------|
| [`groot-bc-unitree-g1.yaml`](groot-bc-unitree-g1.yaml) | BC sweep training (8× L40) |
| [`groot-isaaclab-eval.yaml`](groot-isaaclab-eval.yaml) | Isaac Lab closed-loop eval (2× L40) |
| [`groot-rft-g1.yaml`](groot-rft-g1.yaml) | GRPO/ReST reinforcement finetuning (not active) |
| [`groot-rft-two-container.yaml`](groot-rft-two-container.yaml) | Original RFT reference (MLP policy, not GR00T) |

---

# 8. Troubleshooting

### Pod not starting

```bash
kubectl describe pod groot-bc-g1-0
```

Common issues:
- GPU resources not available → check node GPU allocation
- Image pull errors → verify NGC secret exists
- W&B secret missing → create `wandb-api-key` secret

### Eval not picking up new checkpoints

The eval container tracks evaluated artifacts in memory (cleared on pod restart). If checkpoints are being skipped:

```bash
# Restart to re-evaluate all checkpoints
kubectl delete pod groot-isaaclab-eval-0
```

### W&B timeout from cluster

CoreWeave nodes can have intermittent W&B connectivity. The eval pipeline handles this gracefully — it runs rollouts first, then attempts W&B logging. Videos are saved locally even if W&B fails.

### flash_attn errors

L40 GPUs don't support `flash_attn_2`. The entrypoint scripts automatically patch Eagle backbone to use `sdpa` and create a `flash_attn` stub package.

### Robot not moving in rollouts

Verify the RELATIVE→ABSOLUTE action conversion is active. Check eval logs for:

```
RELATIVE parts (delta→absolute): {'left_arm', 'left_leg', 'right_arm', 'right_leg'}
ABSOLUTE parts (direct target): {'waist', 'left_hand', 'right_hand'}
```

If missing, the `g1_joint_mapper.py` needs the `RELATIVE_PARTS`/`ABSOLUTE_PARTS` constants and the `current_proprio` parameter in `action_chunk_from_groot()`.

---

# Workflow Summary

1. Apply BC sweep YAML → starts hyperparameter search
2. Apply eval YAML → starts polling for checkpoints
3. As each sweep trial completes, it uploads a checkpoint artifact to W&B
4. The eval pod downloads the checkpoint, runs 3 rollout episodes in Isaac Lab, and logs videos + metrics back to W&B
5. Monitor everything in the W&B dashboard

---

# References

- GR00T N1.6: https://github.com/NVIDIA/Isaac-GR00T
- Isaac Lab: https://isaac-sim.github.io/IsaacLab/
- Isaac Sim: https://docs.omniverse.nvidia.com/isaacsim/latest/
- Weights & Biases: https://docs.wandb.ai/
- CoreWeave Kubernetes: https://docs.coreweave.com/
