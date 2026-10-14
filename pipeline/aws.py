# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""boto3 sessions with an explicit region, and the deployed stack's facts resolved from SSM.

The CDK stack writes /<project>/{project,region,bucket,sagemaker_role_arn,ecr_repo_uri}; the CodeBuild image factory
writes /<project>/image_uri (digest-pinned) and /<project>/image_src_hash. Nothing account-specific is ever
committed: the pipeline resolves these at runtime and caches them in .finetune/stack.<project>.<region>.json (git-ignored) for 10 minutes once the image is published.

client() is the one place laptop-side code builds a boto3 client: it attaches the repository name and the solution
user-agent token (pipeline/solution.py; README "Anonymous usage metrics") to every call. A bare boto3 client still works
and is simply never attributed, so new AWS calls go through client() too.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional

import boto3
import botocore.config

from pipeline import solution

REPO = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO / ".finetune"
STACK_KEYS = ("project", "region", "bucket", "sagemaker_role_arn", "ecr_repo_uri")
BUILD_KEYS = ("image_uri", "image_src_hash")
USER_AGENT = "sample-lerobot-policy-finetuning-on-sagemaker"


class StackNotFound(RuntimeError):
    pass


def ambient_region(explicit: Optional[str] = None) -> Optional[str]:
    """--region beats AWS_DEFAULT_REGION beats AWS_REGION; the config carries no region on purpose (the stack
    lives where you deployed it, and boto3 and the CDK CLI read different variables -- pin one explicitly)."""
    return explicit or os.environ.get("AWS_DEFAULT_REGION") or os.environ.get("AWS_REGION")


def session(region: str, profile: Optional[str] = None) -> boto3.Session:
    """Always pin the region explicitly: boto3 ignores AWS_REGION and would otherwise follow the profile."""
    if not region:
        raise ValueError("region is required: pass --region, or export AWS_DEFAULT_REGION (and AWS_REGION for the CDK CLI)")
    return boto3.Session(profile_name=profile, region_name=region)


def client(sess: boto3.Session, service: str):
    """A client on `sess` with this repo's retry policy; solution.get_client merges the AWSSOLUTION token into the user agent."""
    return solution.get_client(service, botocore.config.Config(user_agent_extra=USER_AGENT, retries={"max_attempts": 8, "mode": "standard"}), session=sess)


def resolve_stack(sess: boto3.Session, project: str, *, use_cache: bool = True, cache_dir: Path = CACHE_DIR) -> dict:
    """Return {project, region, bucket, sagemaker_role_arn, ecr_repo_uri, image_uri?, image_src_hash?, resolved_at}."""
    region = sess.region_name
    cache = cache_dir / f"stack.{project}.{region}.json"
    if use_cache and cache.is_file():
        data = json.loads(cache.read_text())
        if time.time() - data.get("resolved_at", 0) < 600 and all(k in data for k in STACK_KEYS):
            return data
    ssm = client(sess, "ssm")
    names = [f"/{project}/{k}" for k in STACK_KEYS + BUILD_KEYS]
    resp = ssm.get_parameters(Names=names)
    found = {p["Name"].rsplit("/", 1)[1]: p["Value"] for p in resp["Parameters"]}
    missing = [k for k in STACK_KEYS if k not in found]
    if missing:
        raise StackNotFound(f"stack '{project}' not found in {region}: SSM parameters missing {missing}. "
                            f"Run `cdk deploy` in cdk/ (region {region}) first, or pass --region <region>.")
    if found["region"] != region:
        raise StackNotFound(f"SSM says the stack lives in {found['region']} but this session targets {region}")
    found["resolved_at"] = time.time()
    if found.get("image_uri"):  # never cache a stack whose image is not published yet (an early --dry-run would keep failing)
        cache_dir.mkdir(exist_ok=True)
        cache.write_text(json.dumps(found, indent=2))
    return found


def s3_root(stack: dict) -> str:
    return f"s3://{stack['bucket']}/sagemaker"
