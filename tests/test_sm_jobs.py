# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline import sm_jobs  # noqa: E402

IMG = "123456789012.dkr.ecr.ap-northeast-1.amazonaws.com/x-sagemaker-training@sha256:" + "a" * 64
ROLE = "arn:aws:iam::123456789012:role/x"


def test_job_name_is_valid_and_bounded():
    n = sm_jobs.job_name("my run!", "act", "train", "g6e4x")
    assert re.fullmatch(r"[a-zA-Z0-9](-*[a-zA-Z0-9]){0,62}", n) and "my-run-act-train-g6e4x-" in n
    long = sm_jobs.job_name("x" * 80, "diffusion", "smoke")
    assert len(long) <= 63 and re.fullmatch(r"[a-zA-Z0-9](-*[a-zA-Z0-9]){0,62}", long)


def test_hyperparameters_are_strings_and_bounded():
    hps = sm_jobs.hyperparameters(steps=200, force=True, heldout="0 13", skip=None)
    assert hps == {"steps": "200", "force": "true", "heldout": "0 13"}
    with pytest.raises(ValueError, match="2500"):
        sm_jobs.hyperparameters(episodes="[" + ",".join(str(i) for i in range(900)) + "]")


def _spec(**kw):
    base = dict(name="run-act-train-20260910-120000", image_uri=IMG, role_arn=ROLE, dataset_s3="s3://b/sagemaker/datasets/d",
                output_s3="s3://b/sagemaker/jobs/train", checkpoint_s3="s3://b/sagemaker/jobs/train/run/checkpoints",
                instance_type="ml.g6e.4xlarge", hps={"steps": "100000"}, max_runtime_s=17100, tags={"project": "p"})
    base.update(kw)
    return sm_jobs.build_spec(**base)


def test_spec_defaults_match_the_validated_shape_plus_hardening():
    s = _spec()
    assert s["EnableNetworkIsolation"] is True and s["ProfilerConfig"] == {"DisableProfiler": True}
    assert s["InputDataConfig"][0]["ChannelName"] == "train"
    assert s["InputDataConfig"][0]["DataSource"]["S3DataSource"]["S3Uri"].endswith("/")
    assert s["CheckpointConfig"]["LocalPath"] == "/opt/ml/checkpoints"
    assert s["ResourceConfig"] == {"InstanceType": "ml.g6e.4xlarge", "InstanceCount": 1, "VolumeSizeInGB": 100}
    assert s["Tags"] == [{"Key": "project", "Value": "p"}] and "EnableManagedSpotTraining" not in s
    assert s["Environment"] == {"PYTHONUNBUFFERED": "1"}


def test_spot_hub_and_digest_rules():
    s = _spec(spot=True, hub_s3="s3://b/sagemaker/hub")
    assert s["EnableManagedSpotTraining"] is True and s["StoppingCondition"]["MaxWaitTimeInSeconds"] == 2 * 17100
    assert [c["ChannelName"] for c in s["InputDataConfig"]] == ["train", "hub"]
    with pytest.raises(ValueError, match="digest"):
        _spec(image_uri=IMG.split("@")[0] + ":v1")
    with pytest.raises(ValueError, match="TrainingJobName"):
        _spec(name="bad name")


def test_derived_runtime_and_volume():
    assert 8.5 * 3600 < sm_jobs.max_runtime_seconds(100000, 10.0) < 9.5 * 3600      # L40S: 2.8 h of steps x 3.0 + 35 min overhead
    assert 4.5 * 3600 < sm_jobs.max_runtime_seconds(100000, 10.0, multiplier=1.5) < 5 * 3600   # the 2026-09-10 ACT re-validation cap
    assert sm_jobs.max_runtime_seconds(100000, 1.74) > 24 * 3600                     # T4 would die at a flat 8 h
    with pytest.raises(ValueError):
        sm_jobs.max_runtime_seconds(100, 0)
    assert sm_jobs.volume_gb(553_000_000) == 100                                       # small dataset -> floor
    assert sm_jobs.volume_gb(40_000_000_000) == int(30 + 80 + 21 * 0.7 + 20)          # 40 GB recording, 20 kept + 1 published copy
    assert sm_jobs.volume_gb(3_300_000_000, n_checkpoints=3, ckpt_gb=31, hub_gb=46) == int(30 + 6.6 + 46 + 4 * 31 + 20)   # FastWAM (31 GB checkpoints, 46 GB staged weights)


def test_training_plan_goes_into_resource_config_and_excludes_spot():
    arn = "arn:aws:sagemaker:ap-northeast-1:123456789012:training-plan/lerobot-fastwam-smoke"
    s = _spec(training_plan_arn=arn)
    assert s["ResourceConfig"]["TrainingPlanArn"] == arn and "EnableManagedSpotTraining" not in s
    assert "TrainingPlanArn" not in _spec()["ResourceConfig"]
    with pytest.raises(ValueError, match="exclusive"):
        _spec(training_plan_arn=arn, spot=True)
