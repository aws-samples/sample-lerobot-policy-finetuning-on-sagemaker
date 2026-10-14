# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
import copy
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from pipeline.config import ConfigError, enabled_policies, load_project_config  # noqa: E402


def _write(tmp_path, cfg):
    p = tmp_path / "project_config.json"
    p.write_text(json.dumps(cfg))
    return p


@pytest.fixture
def shipped():
    return json.loads((REPO / "project_config.json").read_text())


def test_shipped_config_is_valid():
    cfg = load_project_config()
    assert enabled_policies(cfg) == {"act": cfg["policies"]["act"]}
    assert cfg["dataset"]["name"] == "svla_so101_pickplace"
    assert "_comment" not in cfg and "_comment" not in cfg["compute"]


def test_unknown_key_rejected(tmp_path, shipped):
    cfg = copy.deepcopy(shipped)
    cfg["compute"]["instnace_type"] = "ml.g6e.4xlarge"
    with pytest.raises(ConfigError, match="instnace_type"):
        load_project_config(_write(tmp_path, cfg))


def test_duplicate_key_rejected(tmp_path):
    p = tmp_path / "project_config.json"
    p.write_text('{"project": "a-b-c", "project": "d-e-f"}')
    with pytest.raises(ConfigError, match="duplicate key"):
        load_project_config(p)


def test_project_name_pattern(tmp_path, shipped):
    cfg = copy.deepcopy(shipped)
    cfg["project"] = "Bad Name"
    with pytest.raises(ConfigError, match="project"):
        load_project_config(_write(tmp_path, cfg))


def test_smoke_steps_above_training_steps(tmp_path, shipped):
    cfg = copy.deepcopy(shipped)
    cfg["policies"]["act"]["steps"] = 50
    with pytest.raises(ConfigError, match="below smoke.steps"):
        load_project_config(_write(tmp_path, cfg))


def test_owned_and_forbidden_overrides(tmp_path, shipped):
    cfg = copy.deepcopy(shipped)
    cfg["policies"]["act"]["overrides"] = ["--steps=5", "--policy.dtype=bfloat16", "--policy.kl_weight=20"]
    with pytest.raises(ConfigError) as e:
        load_project_config(_write(tmp_path, cfg))
    assert "--steps" in str(e.value) and "bfloat16" in str(e.value) and "kl_weight" not in str(e.value)


def test_legacy_aws_block_rejected_with_migration_hint(tmp_path, shipped):
    cfg = copy.deepcopy(shipped)
    cfg["aws"] = {"region": "ap-northeast-1", "profile": None, "retain_data_on_destroy": False}
    with pytest.raises(ConfigError) as e:
        load_project_config(_write(tmp_path, cfg))
    msg = str(e.value)
    assert "'aws'" in msg and "--region" in msg and "retainData=true" in msg


def test_no_policy_enabled(tmp_path, shipped):
    cfg = copy.deepcopy(shipped)
    cfg["policies"]["act"]["enabled"] = False
    with pytest.raises(ConfigError, match="no policy is enabled"):
        load_project_config(_write(tmp_path, cfg))


def test_cli_check():
    from pipeline.config import main
    assert main(["--check"]) == 0


def test_training_plan_is_reserved_capacity_not_spot(tmp_path, shipped):
    cfg = copy.deepcopy(shipped)
    cfg["compute"]["training_plan_arn"] = "arn:aws:sagemaker:us-east-1:123456789012:training-plan/p5-smoke"
    assert load_project_config(_write(tmp_path, cfg))["compute"]["training_plan_arn"].endswith("p5-smoke")
    cfg["compute"]["spot"] = True
    with pytest.raises(ConfigError, match="training_plan_arn and compute.spot"):
        load_project_config(_write(tmp_path, cfg))
    cfg["compute"]["spot"] = False
    cfg["compute"]["training_plan_arn"] = "not-an-arn"
    with pytest.raises(ConfigError, match="training_plan_arn"):
        load_project_config(_write(tmp_path, cfg))


def test_instance_type_auto_is_accepted_in_compute_and_per_policy(tmp_path, shipped):
    cfg = copy.deepcopy(shipped)
    cfg["compute"]["instance_type"] = "auto"
    assert load_project_config(_write(tmp_path, cfg))["compute"]["instance_type"] == "auto"
    cfg["compute"]["instance_type"] = "ml.g6e.4xlarge"
    cfg["policies"]["act"]["instance_type"] = "auto"
    assert load_project_config(_write(tmp_path, cfg))["policies"]["act"]["instance_type"] == "auto"
    for bad in ("Auto", "automatic", "g6e.4xlarge"):
        cfg["policies"]["act"]["instance_type"] = bad
        with pytest.raises(ConfigError, match="instance_type"):
            load_project_config(_write(tmp_path, cfg))
