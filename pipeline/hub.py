# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Stage a policy's base weights into S3 with a CPU SageMaker job (network on), so training runs isolated.

    prefix = stage_hub(sess, stack, cfg, policy_name, spec, yes=..., dry=...)  ->  s3://<bucket>/sagemaker/hub/<policy>/
The job runs the training image with the container entrypoint overridden to trainer/hub_stage.py; its /opt/ml/checkpoints
mirrors to the prefix as plain objects, which the training job later mounts as channel "hub". A _STAGED.json marker
with the same artefact request and the current manifest schema skips the job; any other manifest (different repos or
revisions, or an older schema whose staging was unfiltered) means the stale prefix is deleted (an unpinned staging whose
resolved sha equals the revision the registry now pins is kept: same bytes), after the confirm prompt
and never under --dry, before the job runs, so a re-stage can shrink the prefix.
"""
from __future__ import annotations

import json
from typing import Optional

from pipeline import sm_jobs, solution
from pipeline.aws import client, s3_root
from pipeline.wait import wait_for_job

STAGE_INSTANCE = "ml.m5.4xlarge"
# 2 = snapshots materialised (regular files; older stagings lost their symlinked snapshots in the S3 mirror)
# 3 = per-role download patterns recorded and the output directory cleared before downloading (hub_stage.py)
MANIFEST_SCHEMA = 3


def hub_prefix(stack: dict, policy: str) -> str:
    return f"{s3_root(stack)}/hub/{policy}/"


def read_manifest(s3, bucket: str, policy: str) -> Optional[dict]:
    """The prefix's _STAGED.json, or None when there is none (or it cannot be read)."""
    key = f"sagemaker/hub/{policy}/_STAGED.json"
    try:
        return json.loads(s3.get_object(Bucket=bucket, Key=key)["Body"].read())
    except Exception:
        return None


def manifest_matches(m: Optional[dict], artifacts: list) -> bool:
    if not m or m.get("schema", 1) < MANIFEST_SCHEMA or not m.get("materialized"):
        return False
    req, staged = m.get("request") or [], m.get("artifacts") or []
    if req == artifacts:
        return True
    # an earlier staging with `revision: null` that resolved to the sha the registry now pins: same bytes, no re-stage
    return len(req) == len(staged) == len(artifacts) > 0 and all(
        r.get("revision") is None and r.get("repo") == s.get("repo") == a.get("repo")
        and s.get("resolved_sha") == a.get("revision") and {**r, "revision": a.get("revision")} == a
        for r, s, a in zip(req, staged, artifacts))


def already_staged(s3, bucket: str, policy: str, artifacts: list) -> Optional[dict]:
    m = read_manifest(s3, bucket, policy)
    return m if manifest_matches(m, artifacts) else None


def list_prefix(s3, bucket: str, key_prefix: str) -> tuple:
    """(keys, total bytes) of every object under the key prefix."""
    keys, total, token = [], 0, None
    while True:
        kw = {"Bucket": bucket, "Prefix": key_prefix}
        if token:
            kw["ContinuationToken"] = token
        r = s3.list_objects_v2(**kw)
        for o in r.get("Contents") or []:
            keys.append(o["Key"])
            total += int(o.get("Size") or 0)
        token = r.get("NextContinuationToken") if r.get("IsTruncated") else None
        if not token:
            return keys, total


def delete_keys(s3, bucket: str, keys: list) -> int:
    for i in range(0, len(keys), 1000):
        s3.delete_objects(Bucket=bucket, Delete={"Objects": [{"Key": k} for k in keys[i:i + 1000]], "Quiet": True})
    return len(keys)


def staging_spec(stack: dict, cfg: dict, policy: str, artifacts: list, image: str, *, run_name: str, dataset_s3: str, instance: str = STAGE_INSTANCE) -> dict:
    name = sm_jobs.job_name(run_name, policy, "hub")
    hps = sm_jobs.hyperparameters(hub_artifacts=json.dumps(artifacts), hf_token_secret=cfg.get("secrets", {}).get("hf_token"))
    # SageMaker requires at least one input channel; the (already staged) dataset prefix is the cheapest existing one
    spec = sm_jobs.build_spec(name=name, image_uri=image, role_arn=stack["sagemaker_role_arn"], dataset_s3=dataset_s3,
                              output_s3=f"{s3_root(stack)}/jobs/hub/", checkpoint_s3=hub_prefix(stack, policy), instance_type=instance, hps=hps,
                              max_runtime_s=2 * 3600, volume_gb=200, network_isolation=False,
                              tags={"project": cfg["project"], "policy": policy, "kind": "hub", **cfg.get("extra_job_tags", {})})
    spec["AlgorithmSpecification"]["ContainerEntrypoint"] = ["python3", "/opt/ml/code/hub_stage.py"]
    # the staging job is the one job that calls an AWS API itself (Secrets Manager, from hub_stage.py); there is no repo
    # checkout inside the image, so the launcher hands it the solution user-agent token (pipeline/solution.py)
    token = solution.user_agent_string()
    if token:
        spec["Environment"]["USER_AGENT_STRING"] = token
    return spec


def stage_hub(sess, stack: dict, cfg: dict, policy: str, spec: dict, image: str, *, run_name: str, dataset_s3: str, yes: bool, dry: bool, log=print, confirm=None) -> Optional[str]:
    artifacts = spec.get("hub_artifacts") or []
    if not artifacts:
        return None
    gated = [a["repo"] for a in artifacts if a.get("gated")]
    if gated and not cfg.get("secrets", {}).get("hf_token"):  # decided before any AWS call
        raise RuntimeError(f"{policy}: gated artefacts {gated} need the Hugging Face token secret (run.py stores it as <project>/hf_token after prompting)")
    s3 = client(sess, "s3")
    prefix = hub_prefix(stack, policy)
    manifest = read_manifest(s3, stack["bucket"], policy)
    if manifest_matches(manifest, artifacts):
        log(f"  base weights already staged: {prefix}")
        return prefix
    js = staging_spec(stack, cfg, policy, artifacts, image, run_name=run_name, dataset_s3=dataset_s3)
    log(f"  will stage {len(artifacts)} artefact(s) for {policy} with a {STAGE_INSTANCE} job (network on) -> {prefix}")
    for a in artifacts:
        log(f"    {a['repo']}@{a.get('revision') or 'main'} ({a.get('role')}){' [gated]' if a.get('gated') else ''}")
    stale = []
    if manifest is not None:   # an earlier staging with another request or an older (unfiltered) schema: cleared before the job
        stale, stale_bytes = list_prefix(s3, stack["bucket"], f"sagemaker/hub/{policy}/")
        why = "manifest schema below the current one" if manifest.get("schema", 1) < MANIFEST_SCHEMA else "different artefact request"
        log(f"    stale staging ({why}): {len(stale)} object(s), {stale_bytes / 1e9:.2f} GB under {prefix} deleted before the job"
            + (" (dry run: kept)" if dry else ""))
    if dry:
        return prefix
    if confirm:
        confirm(f"[{policy}] " + (f"delete {len(stale)} stale object(s) under {prefix} and " if stale else "") + "run the base-weight staging job?", yes)
    if stale:
        log(f"  deleted {delete_keys(s3, stack['bucket'], stale)} stale object(s) under {prefix}")
    sm_jobs.submit(client(sess, "sagemaker"), js)
    log(f"  submitted {js['TrainingJobName']}")
    wait_for_job(lambda: client(sess, "sagemaker"), js["TrainingJobName"], log=log)
    return prefix
