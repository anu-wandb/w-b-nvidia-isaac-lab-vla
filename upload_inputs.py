#!/usr/bin/env python3
"""
Upload GR00T N1.6-3B base model and G1 teleop dataset to W&B as artifacts.

These artifacts are required by the Kubernetes training and evaluation pipelines.

Usage:
    pip install wandb huggingface_hub
    python upload_inputs.py \
        --entity <WANDB_ENTITY> \
        --project <WANDB_PROJECT>

This will:
  1. Download GR00T N1.6-3B from HuggingFace (nvidia/GR00T-N1.6-3B)
  2. Download G1 teleop dataset from HuggingFace (nvidia/PhysicalAI-Robotics-GR00T-Teleop-G1)
  3. Upload both as W&B artifacts
"""

import argparse
import os
from pathlib import Path

import wandb
from huggingface_hub import snapshot_download


def download_and_upload_model(entity: str, project: str, download_dir: str) -> None:
    """Download GR00T N1.6-3B from HuggingFace and upload as a W&B artifact."""
    model_dir = os.path.join(download_dir, "GR00T-N1.6-3B")
    print(f"Downloading GR00T N1.6-3B to {model_dir} ...")
    snapshot_download(
        repo_id="nvidia/GR00T-N1.6-3B",
        local_dir=model_dir,
    )
    print("Download complete. Uploading to W&B ...")

    run = wandb.init(entity=entity, project=project, job_type="upload-model")
    artifact = wandb.Artifact("groot-n1.6-3b", type="model")
    artifact.add_dir(model_dir)
    run.log_artifact(artifact)
    run.finish()
    print(f"Model artifact uploaded: {entity}/{project}/groot-n1.6-3b:v0")


def download_and_upload_dataset(entity: str, project: str, download_dir: str) -> None:
    """Download G1 teleop dataset from HuggingFace and upload as a W&B artifact."""
    dataset_dir = os.path.join(download_dir, "PhysicalAI-Robotics-GR00T-Teleop-G1")
    print(f"Downloading G1 teleop dataset to {dataset_dir} ...")
    snapshot_download(
        repo_id="nvidia/PhysicalAI-Robotics-GR00T-Teleop-G1",
        repo_type="dataset",
        local_dir=dataset_dir,
    )
    print("Download complete. Uploading to W&B ...")

    run = wandb.init(entity=entity, project=project, job_type="upload-dataset")
    artifact = wandb.Artifact("groot-teleop-unitree-g1", type="dataset")
    artifact.add_dir(dataset_dir)
    run.log_artifact(artifact)
    run.finish()
    print(f"Dataset artifact uploaded: {entity}/{project}/groot-teleop-unitree-g1:v0")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download GR00T model & G1 teleop dataset from HuggingFace and upload to W&B."
    )
    parser.add_argument("--entity", required=True, help="W&B entity (team or username)")
    parser.add_argument("--project", required=True, help="W&B project name")
    parser.add_argument(
        "--download-dir",
        default="./downloads",
        help="Local directory for downloaded files (default: ./downloads)",
    )
    parser.add_argument(
        "--skip-model",
        action="store_true",
        help="Skip model download/upload (if already uploaded)",
    )
    parser.add_argument(
        "--skip-dataset",
        action="store_true",
        help="Skip dataset download/upload (if already uploaded)",
    )
    args = parser.parse_args()

    Path(args.download_dir).mkdir(parents=True, exist_ok=True)

    if not args.skip_model:
        download_and_upload_model(args.entity, args.project, args.download_dir)
    else:
        print("Skipping model upload.")

    if not args.skip_dataset:
        download_and_upload_dataset(args.entity, args.project, args.download_dir)
    else:
        print("Skipping dataset upload.")

    print("\nDone! Both artifacts are now available in your W&B project.")
    print("Update the YAML env vars to point to your entity/project if they differ from the defaults.")


if __name__ == "__main__":
    main()
