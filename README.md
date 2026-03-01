<p align="center">
  <img src="./assets/nvidia-isaac.webp" width="400" alt="NVIDIA Isaac Lab" style="vertical-align: middle;" />
  &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="./assets/Endorsed_primary_goldwhite.png#gh-dark-mode-only" width="600" alt="Weights & Biases" style="vertical-align: middle;" />
  <img src="./assets/Endorsed_primary_goldblack.png#gh-light-mode-only" width="600" alt="Weights & Biases" style="vertical-align: middle;" />
</p>

# Running NVIDIA Isaac Lab for Reinforcement Learning on Kubernetes with Weights & Biases Tracking

This guide explains how to run distributed NVIDIA Isaac Lab training on
a Kubernetes GPU cluster using the provided StatefulSet YAML
configuration. We are using a L40 (2 nodes x 4 GPU cluster from Coreweave here, please update the hardware configs to match your cluster)

This setup supports:

-   Multi-node distributed training (2 nodes × 4 GPUs each)
-   Headless Isaac Sim execution
-   Automatic Weights & Biases (W&B) logging
-   Artifact tracking
-   Checkpoint and video upload after training
-   Persistent shared storage for logs and models


<p align="center">
  <img src="./assets/W&B_Demo_gif.gif" alt="Weights & Biases" />
</p>

## See this blueprint running live on W&B: [Isaac Lab + W&B on Coreweave](https://wandb.ai/wandb-smle/isaaclab-wandb-crwv?nw=nwuseranushravvatsa)

------------------------------------------------------------------------

# Documentation References

-   Isaac Lab GitHub
    https://github.com/isaac-sim/IsaacLab

-   Isaac Sim Documentation
    https://docs.omniverse.nvidia.com/isaacsim/latest/index.html

-   Weights & Biases Documentation
    https://docs.wandb.ai

-   Kubernetes Documentation
    https://kubernetes.io/docs/home/

-   NVIDIA NGC (Container Registry)
    https://ngc.nvidia.com

------------------------------------------------------------------------

# Architecture Overview

The provided YAML deploys:

-   A StatefulSet with 2 replicas
-   4 GPUs per pod
-   torch.distributed multi-node training
-   Headless Isaac Sim
-   A 400Gi PersistentVolumeClaim (ReadWriteMany)
-   Automatic W&B artifact download before training
-   Automatic checkpoint + video upload after training

Training task:

    Isaac-Dexsuite-Kuka-Allegro-Lift-v0

Other avaialble tasks and enviroments that ship with Isaac Sim: https://isaac-sim.github.io/IsaacLab/main/source/overview/environments.html

------------------------------------------------------------------------

# 1. Prerequisites

## Select the Correct Kubernetes Context for your cluster

``` bash
kubectl config get-contexts
kubectl config use-context <your-cluster-context>
```

------------------------------------------------------------------------

# 2. NVIDIA Container Registry (NGC) Setup

Container image used:

    nvcr.io/nvidia/isaac-lab:2.3.2

Create an NGC account:

https://ngc.nvidia.com

Generate an NGC API key and create a Kubernetes image pull secret:

``` bash
kubectl create secret docker-registry nvcr-secret   --docker-server=nvcr.io   --docker-username='$oauthtoken'   --docker-password='<YOUR_NGC_API_KEY>'  
```

If required, add to your pod spec:

``` yaml
imagePullSecrets:
  - name: nvcr-secret
```

------------------------------------------------------------------------

# 3. Weights & Biases Setup

Create a W&B account:

https://wandb.ai

Generate an API key:

https://wandb.ai/authorize

Store it as a Kubernetes secret:

``` bash
kubectl create secret generic wandb-api-key   --from-literal=WANDB_API_KEY=<YOUR_WANDB_API_KEY>
```

Referenced in YAML as:

``` yaml
- name: WANDB_API_KEY
  valueFrom:
    secretKeyRef:
      name: wandb-api-key
      key: WANDB_API_KEY
```

------------------------------------------------------------------------

# 4. Running Training

Three workload configurations are available:

### Training Only

Runs distributed training with no W&B tracking or artifact uploads — useful for validating that training starts and runs correctly on your cluster.

``` bash
kubectl apply -f isaaclab-workload-only-train.yaml
```

### End-to-End Test

Runs the full pipeline (training, W&B tracking, checkpoint and video upload) at minimal scale (3 iterations, 32 envs, 1 GPU per node) as a smoke test.

``` bash
kubectl apply -f isaaclab-workload-end-to-end-test.yaml
```

### Full Training

Runs the complete pipeline at full scale (4 GPUs/node, 1500 iterations, 8192 envs) with W&B tracking and artifact uploads.

``` bash
kubectl apply -f isaaclab-workload-max.yaml
```

### Monitoring

Check pods:

``` bash
kubectl get pods
```

View logs:

``` bash
kubectl logs -f isaaclab-workers-0
kubectl logs -f isaaclab-workers-1
```

Currently W&B logging is enabled from node rank 0 only. W&B also supports logging from all nodes if needed.

------------------------------------------------------------------------

# 5. Cleanup

Delete deployment:

``` bash
kubectl delete -f <your-workload-file>.yaml
```

Delete PVC (optional):

``` bash
kubectl delete pvc isaaclab-logs-pvc
```

------------------------------------------------------------------------

# Workflow Summary:

1.  Select cluster context
2.  Ensure NGC secret exists
3.  Ensure W&B secret exists
4.  Apply YAML
5.  Monitor logs

Training runs distributed across 8 GPUs. 
Training is being tracked in Weights &
Biases.
Checkpoints and videos upload automatically to Weights &
Biases.
