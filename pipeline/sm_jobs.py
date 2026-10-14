# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""SageMaker training-job specs and submission.

Defaults every job gets: network isolation ON, the SageMaker profiler OFF, the image pinned by digest, checkpoints
under /opt/ml/checkpoints mirrored to S3, hyperparameters as strings, project tags. The hyperparameter contract is
defined by trainer/sagemaker/train_entrypoint.sh (see its header comment); run.py's policy_hps renders it.
"""
from __future__ import annotations

import datetime as dt
import re
from typing import Optional

HP_MAX_LEN = 2500            # SageMaker rejects longer hyperparameter values
JOB_NAME_MAX = 63
SMOKE_PREFIX = "jobs/smoke"  # matches the bucket lifecycle rule (expires after 14 days)
TRAIN_PREFIX = "jobs/train"


def _stamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S")


def job_name(run: str, policy: str, kind: str, suffix: str = "") -> str:
    """<run>-<policy>-<kind>[-<suffix>]-<utc stamp>, sanitised to SageMaker's ^[a-zA-Z0-9](-*[a-zA-Z0-9]){0,62}$."""
    parts = [run, policy, kind] + ([suffix] if suffix else []) + [_stamp()]
    name = re.sub(r"[^a-zA-Z0-9-]", "-", "-".join(parts)).strip("-")
    name = re.sub(r"-{2,}", "-", name)
    if len(name) > JOB_NAME_MAX:
        head = name[: JOB_NAME_MAX - 16].rstrip("-")
        name = f"{head}-{name[-15:]}"
    return name


def hyperparameters(**kv) -> dict:
    """All values become strings; None drops the key; anything over 2500 chars is refused before the API does."""
    out = {}
    for k, v in kv.items():
        if v is None:
            continue
        s = str(v).lower() if isinstance(v, bool) else str(v)
        if len(s) > HP_MAX_LEN:
            raise ValueError(f"hyperparameter {k} is {len(s)} chars; SageMaker allows {HP_MAX_LEN}. "
                             f"Use 'heldout' (the short list) instead of 'episodes'.")
        out[k] = s
    return out


def build_spec(*, name: str, image_uri: str, role_arn: str, dataset_s3: str, output_s3: str, checkpoint_s3: str,
               instance_type: str, hps: dict, max_runtime_s: int, volume_gb: int = 100, spot: bool = False, training_plan_arn: str = None,
               max_wait_s: Optional[int] = None, tags: Optional[dict] = None, hub_s3: Optional[str] = None,
               network_isolation: bool = True, input_mode: str = "File") -> dict:
    if "@sha256:" not in image_uri:
        raise ValueError(f"image must be pinned by digest (…@sha256:…), got {image_uri}")
    if not re.fullmatch(r"[a-zA-Z0-9](-*[a-zA-Z0-9]){0,62}", name):
        raise ValueError(f"invalid TrainingJobName {name!r}")
    channels = [{"ChannelName": "train", "InputMode": input_mode,
                 "DataSource": {"S3DataSource": {"S3DataType": "S3Prefix", "S3Uri": dataset_s3.rstrip("/") + "/",
                                                 "S3DataDistributionType": "FullyReplicated"}}}]
    if hub_s3:
        channels.append({"ChannelName": "hub", "InputMode": input_mode,
                         "DataSource": {"S3DataSource": {"S3DataType": "S3Prefix", "S3Uri": hub_s3.rstrip("/") + "/",
                                                         "S3DataDistributionType": "FullyReplicated"}}})
    spec = {
        "TrainingJobName": name,
        "AlgorithmSpecification": {"TrainingImage": image_uri, "TrainingInputMode": input_mode},
        "RoleArn": role_arn,
        "HyperParameters": dict(hps),
        "InputDataConfig": channels,
        "OutputDataConfig": {"S3OutputPath": output_s3.rstrip("/") + "/"},
        "CheckpointConfig": {"S3Uri": checkpoint_s3.rstrip("/") + "/", "LocalPath": "/opt/ml/checkpoints"},
        "ResourceConfig": {"InstanceType": instance_type, "InstanceCount": 1, "VolumeSizeInGB": int(volume_gb)},
        "StoppingCondition": {"MaxRuntimeInSeconds": int(max_runtime_s)},
        "EnableNetworkIsolation": bool(network_isolation),
        "ProfilerConfig": {"DisableProfiler": True},
        "Environment": {"PYTHONUNBUFFERED": "1"},
        "Tags": [{"Key": k, "Value": str(v)} for k, v in (tags or {}).items()],
    }
    if spot and training_plan_arn:
        raise ValueError("a training plan (reserved capacity) and managed Spot are exclusive")
    if training_plan_arn:
        spec["ResourceConfig"]["TrainingPlanArn"] = training_plan_arn
    if spot:
        spec["EnableManagedSpotTraining"] = True
        spec["StoppingCondition"]["MaxWaitTimeInSeconds"] = int(max_wait_s or max_runtime_s * 2)
    return spec


def max_runtime_seconds(steps: int, step_rate: float, *, multiplier: float = 3.0, overhead_s: int = 15 * 60,
                        eval_s: int = 20 * 60) -> int:
    """Derived cap, never a flat 8 h: image pull + downscale + first-batch warm-up (~15 min) + steps/rate x multiplier + eval.
    The smoke rate (200 steps, no checkpoint writes) is only slightly optimistic: measured VLA training runs held 96-99 % of it in
    steady state and 85-95 % per billable second, using 59-69 % of their x1.5 caps. The 3.0 default multiplier is headroom for slower
    policies, eval and publish, not a measured loss."""
    if step_rate <= 0:
        raise ValueError("step_rate must be > 0 steps/s (from the smoke SUMMARY or the registry table)")
    return int(overhead_s + eval_s + (steps / step_rate) * multiplier)


def volume_gb(staged_bytes: int, *, image_gb: int = 30, n_checkpoints: int = 20, ckpt_gb: float = 0.7, hub_gb: float = 0.0, floor: int = 100) -> int:
    """Room for the image, the raw + downscaled dataset copies, the staged hub channel, the kept checkpoints (weights + optimizer
    state) and publish.py's copy of the newest one. A FastWAM checkpoint is ~31 GB with its optimizer state, a pi0 one ~19 GB:
    registry hardware.checkpoint_gb / hardware.hub_gb feed this; 20 GB headroom; never below `floor`."""
    return int(max(floor, image_gb + 2 * staged_bytes / 1e9 + hub_gb + (n_checkpoints + 1) * ckpt_gb + 20))


def submit(sm, spec: dict) -> str:
    sm.create_training_job(**spec)
    return spec["TrainingJobName"]


def console_url(region: str, job: str) -> str:
    return f"https://{region}.console.aws.amazon.com/sagemaker/home?region={region}#/jobs/{job}"
