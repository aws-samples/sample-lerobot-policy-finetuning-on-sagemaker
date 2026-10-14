# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
import sys
from pathlib import Path

import boto3
import pytest
from botocore.stub import Stubber

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline import aws  # noqa: E402

PARAMS = {"project": "lerobot-finetune", "region": "ap-northeast-1", "bucket": "b", "sagemaker_role_arn": "arn:aws:iam::123456789012:role/r",
          "ecr_repo_uri": "123456789012.dkr.ecr.ap-northeast-1.amazonaws.com/lerobot-finetune-sagemaker-training"}


def _stub(sess, found: dict, monkeypatch):
    ssm = sess.client("ssm")
    st = Stubber(ssm)
    st.add_response("get_parameters", {"Parameters": [{"Name": f"/lerobot-finetune/{k}", "Value": v} for k, v in found.items()]}, expected_params={"Names": [f"/lerobot-finetune/{k}" for k in aws.STACK_KEYS + aws.BUILD_KEYS]})
    st.activate()
    monkeypatch.setattr(aws, "client", lambda s, name: ssm)
    return st


def test_ambient_region_precedence(monkeypatch):
    monkeypatch.setenv("AWS_DEFAULT_REGION", "ap-northeast-1")
    monkeypatch.setenv("AWS_REGION", "us-west-2")
    assert aws.ambient_region("eu-central-1") == "eu-central-1"      # an explicit --region beats everything
    assert aws.ambient_region(None) == "ap-northeast-1"              # boto3's variable beats the CDK CLI's
    monkeypatch.delenv("AWS_DEFAULT_REGION")
    assert aws.ambient_region(None) == "us-west-2"
    monkeypatch.delenv("AWS_REGION")
    assert aws.ambient_region(None) is None                          # session() then fails with the readable error


def test_session_requires_region():
    with pytest.raises(ValueError):
        aws.session("")
    assert aws.session("ap-northeast-1").region_name == "ap-northeast-1"


def test_resolve_stack_and_cache(tmp_path, monkeypatch):
    sess = boto3.Session(region_name="ap-northeast-1", aws_access_key_id="AKIAIOSFODNN7EXAMPLE", aws_secret_access_key="x")  # pragma: allowlist secret
    _stub(sess, {**PARAMS, "image_uri": PARAMS["ecr_repo_uri"] + "@sha256:" + "a" * 64}, monkeypatch)
    out = aws.resolve_stack(sess, "lerobot-finetune", cache_dir=tmp_path)
    assert out["bucket"] == "b" and out["image_uri"].endswith("a" * 64)
    assert (tmp_path / "stack.lerobot-finetune.ap-northeast-1.json").is_file()
    # second call is served from the cache: no stubbed response left, so an API call would raise
    assert aws.resolve_stack(sess, "lerobot-finetune", cache_dir=tmp_path)["bucket"] == "b"


def test_missing_stack_and_region_mismatch(tmp_path, monkeypatch):
    sess = boto3.Session(region_name="ap-northeast-1", aws_access_key_id="AKIAIOSFODNN7EXAMPLE", aws_secret_access_key="x")  # pragma: allowlist secret
    _stub(sess, {"project": "lerobot-finetune"}, monkeypatch)
    with pytest.raises(aws.StackNotFound, match="cdk deploy"):
        aws.resolve_stack(sess, "lerobot-finetune", cache_dir=tmp_path, use_cache=False)
    _stub(sess, {**PARAMS, "region": "eu-central-1"}, monkeypatch)
    with pytest.raises(aws.StackNotFound, match="eu-central-1"):
        aws.resolve_stack(sess, "lerobot-finetune", cache_dir=tmp_path, use_cache=False)
