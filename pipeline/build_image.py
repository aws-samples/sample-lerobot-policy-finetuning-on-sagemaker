# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Build the training image on AWS CodeBuild from the current trainer/ tree (no Docker on this machine).

    python3 -m pipeline.build_image --region <region> [--tag dev-<gitsha>] [--publish-ssm] [--lock-mode enforce|generate]
    python3 -m pipeline.build_image --region <region> --refresh-lock      # generate mode -> trainer/sagemaker/requirements.lock

Zips trainer/ (same excludes as the CDK asset), uploads it to s3://<bucket>/sagemaker/build-context/<hash>.zip and starts
the stack's CodeBuild project with a source override, so a code change does not need `cdk deploy`. Developer builds tag
dev-<gitsha> and never touch the SSM image pointer unless --publish-ssm is given. Prints the digest-pinned image URI.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import io
import json
import subprocess
import sys
import time
import zipfile
from typing import Optional

from botocore.exceptions import ClientError
from pipeline.aws import CACHE_DIR, REPO, ambient_region, client, resolve_stack, session
from pipeline.config import load_project_config

TRAINER = REPO / "trainer"
EXCLUDE_DIRS = {"__pycache__", ".pytest_cache"}
EXCLUDE_SUFFIXES = (".pyc", ".egg-info")
LOCK_PATH = TRAINER / "sagemaker" / "requirements.lock"


def trainer_files():
    for p in sorted(TRAINER.rglob("*")):
        rel = p.relative_to(TRAINER)
        if not p.is_file() or any(part in EXCLUDE_DIRS for part in rel.parts) or p.name.startswith("README"):
            continue
        if p.name.endswith(EXCLUDE_SUFFIXES):
            continue
        yield rel, p


def zip_trainer() -> tuple[bytes, str]:
    """Deterministic zip (fixed timestamps) + sha256 of the file contents => the build's DOCKER_SRC_HASH."""
    h = hashlib.sha256()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for rel, p in trainer_files():
            data = p.read_bytes()
            h.update(str(rel).encode() + b"\0" + data + b"\0")
            info = zipfile.ZipInfo(str(rel), date_time=(1980, 1, 1, 0, 0, 0))
            info.external_attr = (0o755 if p.suffix == ".sh" else 0o644) << 16
            z.writestr(info, data)
    return buf.getvalue(), h.hexdigest()


def git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short=10", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "nogit"


def start_build(cb, project: str, bucket: str, key: str, *, tag: str, src_hash: str, publish_ssm: bool, lock_mode: str) -> str:
    resp = cb.start_build(
        projectName=f"{project}-image-build",
        sourceTypeOverride="S3", sourceLocationOverride=f"{bucket}/{key}",
        environmentVariablesOverride=[
            {"name": "IMAGE_TAG", "value": tag, "type": "PLAINTEXT"},
            {"name": "DOCKER_SRC_HASH", "value": src_hash, "type": "PLAINTEXT"},
            {"name": "PUBLISH_SSM", "value": "1" if publish_ssm else "0", "type": "PLAINTEXT"},
            {"name": "LOCK_MODE", "value": lock_mode, "type": "PLAINTEXT"},
        ],
    )
    return resp["build"]["id"]


def wait_build(cb, build_id: str, *, poll_s: int = 20, log=print) -> dict:
    last = None
    while True:
        b = cb.batch_get_builds(ids=[build_id])["builds"][0]
        phase, status = b.get("currentPhase"), b.get("buildStatus")
        if (phase, status) != last:
            log(f"  build {build_id.split(':')[-1][:8]}: {phase} / {status}")
            last = (phase, status)
        if status != "IN_PROGRESS":
            return b
        time.sleep(poll_s)


def _publishes(build: dict) -> bool:
    """True for a build that writes /<project>/image_uri: the project default is PUBLISH_SSM=1, dev builds override it to 0."""
    env = {v.get("name"): v.get("value") for v in ((build.get("environment") or {}).get("environmentVariables") or [])}
    return env.get("PUBLISH_SSM", "1") == "1"


def newest_publishing_build(cb, cb_project: str, *, lookback: int = 10) -> Optional[dict]:
    """The newest build of `cb_project` that publishes to SSM, or None when there is none among the last `lookback`."""
    ids = cb.list_builds_for_project(projectName=cb_project, sortOrder="DESCENDING").get("ids") or []
    if not ids:
        return None
    by_id = {b["id"]: b for b in cb.batch_get_builds(ids=ids[:lookback])["builds"]}
    for i in ids[:lookback]:   # list order is newest first; batch_get_builds does not promise an order
        b = by_id.get(i)
        if b is not None and _publishes(b):
            return b
    return None


def ensure_image(sess, stack: dict, project: str, *, log=print, poll_s: int = 30, resolve=None) -> dict:
    """Return `stack` with the image the run should use, following a running image build first.

    `cdk deploy` starts the image build and returns before it finishes (about 15 min): on the first deploy nothing is
    published yet, and after a redeploy that changed trainer/ or image.tag the *previous* digest is still published
    while the new one builds. Both cases are the same to the operator: the newest publishing build is running, so wait
    for it and re-read the SSM facts. Dev builds (PUBLISH_SSM=0) are ignored. Raises RuntimeError (exit 4 in run.py)
    when nothing is published and there is no publishing build, or that build failed or published nothing.
    """
    from pipeline.aws import resolve_stack
    resolve = resolve or resolve_stack
    published = stack.get("image_uri")
    region = sess.region_name
    cb_project = f"{project}-image-build"
    console = f"https://{region}.console.aws.amazon.com/codesuite/codebuild/projects/{cb_project}/history?region={region}"
    rebuild = f"python3 -m pipeline.build_image --region {region} --tag <image.tag> --publish-ssm"
    try:
        cb = client(sess, "codebuild")
        b = newest_publishing_build(cb, cb_project)
    except ClientError as e:
        if published:   # cannot read CodeBuild (permissions): the published image is what the run gets
            log(f"  NOTE: cannot read the image builds ({e.response.get('Error', {}).get('Code', 'ClientError')}); using the published image as is")
            return stack
        raise
    if b is None:
        if published:
            return stack
        raise RuntimeError(f"no image: the stack in {region} has no publishing image build. `cdk deploy` starts one; or run {rebuild}")
    status = b.get("buildStatus")
    if status == "IN_PROGRESS":
        started = b.get("startTime")
        since = f" (started {started.astimezone(dt.timezone.utc):%H:%M} UTC, about 15 min in total)" if hasattr(started, "astimezone") else ""
        what = "a new image build" if published else "the deploy-time image build"
        log(f"image: {what} is still running{since}; waiting. Ctrl-C leaves it running.\n  console: {console}")
        b = wait_build(cb, b["id"], poll_s=poll_s, log=log)
        status = b.get("buildStatus")
    elif published:
        if status != "SUCCEEDED":
            log(f"  WARN image: the newest image build {status}; this run uses the previously published image. Console: {console}")
        return stack
    if status != "SUCCEEDED":
        raise RuntimeError(f"image build {status} (phase {b.get('currentPhase')}); see {console} then re-run `cdk deploy` or {rebuild}")
    stack = resolve(sess, project, use_cache=False)
    if not stack.get("image_uri"):
        raise RuntimeError(f"the image build succeeded but /{project}/image_uri is still unset; publish it with {rebuild}")
    return stack


def image_digest_uri(ecr, repo_uri: str, tag: str) -> str:
    repo_name = repo_uri.split("/", 1)[1]
    d = ecr.describe_images(repositoryName=repo_name, imageIds=[{"imageTag": tag}])["imageDetails"][0]
    return f"{repo_uri}@{d['imageDigest']}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--region"); ap.add_argument("--profile"); ap.add_argument("--project")
    ap.add_argument("--tag", help="ECR tag (default dev-<gitsha>)")
    ap.add_argument("--publish-ssm", action="store_true", help="also move /<project>/image_uri to this build (release builds)")
    ap.add_argument("--lock-mode", choices=["enforce", "generate"], default="enforce")
    ap.add_argument("--refresh-lock", action="store_true", help="generate mode, then write the produced lock to trainer/sagemaker/requirements.lock")
    ap.add_argument("--no-wait", action="store_true")
    a = ap.parse_args(argv)
    cfg = load_project_config()
    project = a.project or cfg["project"]
    region = ambient_region(a.region)
    if not region:
        sys.exit("ERROR: region is required: pass --region, or export AWS_DEFAULT_REGION (and AWS_REGION for the CDK CLI)")
    sess = session(region, a.profile)
    stack = resolve_stack(sess, project)
    lock_mode = "generate" if a.refresh_lock else a.lock_mode
    tag = a.tag or f"dev-{git_sha()}"
    if a.publish_ssm and tag.startswith("dev-"):
        sys.exit("refusing --publish-ssm with a dev- tag; pass --tag <release tag>")
    if a.publish_ssm:   # the SSM pointer will move: drop the 10-minute stack cache now so no exit path (incl. --no-wait) keeps the old digest
        (CACHE_DIR / f"stack.{project}.{region}.json").unlink(missing_ok=True)
    data, src_hash = zip_trainer()
    key = f"sagemaker/build-context/{src_hash[:16]}.zip"
    client(sess, "s3").put_object(Bucket=stack["bucket"], Key=key, Body=data)
    print(f"source: {len(data) / 1e3:.0f} kB -> s3://{stack['bucket']}/{key} (hash {src_hash[:12]})")
    cb = client(sess, "codebuild")
    build_id = start_build(cb, project, stack["bucket"], key, tag=tag, src_hash=src_hash, publish_ssm=a.publish_ssm, lock_mode=lock_mode)
    print(f"build: {build_id} tag={tag} lock_mode={lock_mode} publish_ssm={a.publish_ssm}")
    print(f"  console: https://{region}.console.aws.amazon.com/codesuite/codebuild/projects/{project}-image-build/build/{build_id}?region={region}")
    if a.no_wait:
        return 0
    b = wait_build(cb, build_id)
    if b["buildStatus"] != "SUCCEEDED":
        print(f"ERROR: build {b['buildStatus']} (phase {b.get('currentPhase')}); see the console link above", file=sys.stderr)
        return 4
    if lock_mode == "generate":
        lock_key = f"sagemaker/build-artifacts/{build_id}/requirements.lock"
        body = client(sess, "s3").get_object(Bucket=stack["bucket"], Key=lock_key)["Body"].read()
        LOCK_PATH.write_bytes(body)
        print(f"lock: {len(body.splitlines())} pinned packages -> {LOCK_PATH.relative_to(REPO)} (commit it)")
        return 0
    uri = image_digest_uri(client(sess, "ecr"), stack["ecr_repo_uri"], tag)
    CACHE_DIR.mkdir(exist_ok=True)
    (CACHE_DIR / "last_build.json").write_text(json.dumps({"build_id": build_id, "tag": tag, "image_uri": uri, "src_hash": src_hash, "region": region}, indent=2))
    print(f"image: {uri}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
