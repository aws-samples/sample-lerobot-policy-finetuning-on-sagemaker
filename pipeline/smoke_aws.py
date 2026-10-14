# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""The AWS-side correctness smoke: a 20-step CPU training job on the real image and a real dataset (~$0.30, 12-18 min).

    python3 -m pipeline.smoke_aws --region <region> [--image <ecr-uri@sha256:...>] [--dataset-s3 s3://...]

This is the per-change gate for anything under trainer/: it proves the image starts, reads the dataset, downscales,
trains, evaluates and publishes SUMMARY.json. Numbers are meaningless at 20 steps; only completion and a finite
joints MAE are asserted. Defaults: the image the stack published to SSM, the dataset named in project_config.json
staged under s3://<bucket>/sagemaker/datasets/<name>/, ml.c5.4xlarge.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import sys
import tarfile

from pipeline.aws import ambient_region, client, resolve_stack, s3_root, session
from pipeline.config import load_project_config
from pipeline.sm_jobs import SMOKE_PREFIX, build_spec, hyperparameters, job_name, submit
from pipeline.wait import JobFailed, wait_for_job


def read_summary(s3, bucket: str, key: str) -> dict:
    body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    with tarfile.open(fileobj=io.BytesIO(body), mode="r:gz") as tf:
        member = next((m for m in tf.getmembers() if m.name.lstrip("./") == "SUMMARY.json"), None)
        if member is None:
            raise RuntimeError("model.tar.gz has no SUMMARY.json")
        return json.load(tf.extractfile(member))


EPILOG = """\
Fingerprint check (opt-in): --expect-joints-mae <value> [--mae-tol <eps>] turns the by-hand comparison of the printed
joints MAE into PASS/FAIL. A 20-step CPU ACT job on the default public dataset has produced the same MAE (17.3817) on every
validated release image, so a rebuilt image that still prints it has not changed the ACT path. The check is off by
default; the numbers are still meaningless as a measure of policy quality.
"""


def judge(mae, expect: float | None = None, tol: float = 1e-4) -> tuple[bool, str]:
    """Verdict on the published joints MAE: finite, and within tol of --expect-joints-mae when one is given."""
    if mae is None or not math.isfinite(float(mae)):
        return False, "no finite joints_mae in SUMMARY.json"
    if expect is None:
        return True, "finite"
    diff = abs(float(mae) - expect)
    if diff <= tol:
        return True, f"matches expected {expect:g} (|diff| {diff:.2e} <= {tol:g})"
    return False, f"differs from expected {expect:g} by {diff:.4g} (tolerance {tol:g}): the ACT path changed"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, epilog=EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--region"); ap.add_argument("--profile"); ap.add_argument("--project")
    ap.add_argument("--image", help="digest-pinned image URI (default: SSM /<project>/image_uri)")
    ap.add_argument("--dataset-s3", help="dataset prefix (default: s3://<bucket>/sagemaker/datasets/<dataset.name>/)")
    ap.add_argument("--instance-type", default="ml.c5.4xlarge")
    ap.add_argument("--steps", type=int, default=20); ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--heldout", default="0 1", help="held-out episode ids for the (meaningless) 20-step eval")
    ap.add_argument("--policy", default="act")
    ap.add_argument("--hp", action="append", default=[], metavar="KEY=VALUE",
                help="extra trainer hyperparameters, e.g. force_res=1 for a 4:3 dataset")
    ap.add_argument("--name", default="smoke")
    ap.add_argument("--max-minutes", type=int, default=90)
    ap.add_argument("--no-wait", action="store_true")
    ap.add_argument("--expect-joints-mae", type=float, metavar="VALUE",
                    help="FAIL unless the published joints MAE is within --mae-tol of VALUE (default: off, finite MAE only)")
    ap.add_argument("--mae-tol", type=float, default=1e-4, metavar="EPS",
                    help="tolerance for --expect-joints-mae (default 1e-4, the ledger records the MAE to four decimals)")
    return ap


def main(argv=None) -> int:
    ap = build_parser()
    a = ap.parse_args(argv)
    if a.mae_tol < 0:
        ap.error("--mae-tol must be >= 0")
    cfg = load_project_config()
    project = a.project or cfg["project"]
    region = ambient_region(a.region)
    if not region:
        sys.exit("ERROR: region is required: pass --region, or export AWS_DEFAULT_REGION (and AWS_REGION for the CDK CLI)")
    sess = session(region, a.profile)
    stack = resolve_stack(sess, project, use_cache=False)   # per-change gate: always read the live SSM image pointer
    image = a.image or stack.get("image_uri")
    if not image:
        sys.exit("no image: the stack has not published /<project>/image_uri yet (is the deploy-time build finished?) -- or pass --image")
    root = s3_root(stack)
    dataset_s3 = a.dataset_s3 or f"{root}/datasets/{cfg['dataset']['name']}/"
    extra = dict(kv.split("=", 1) for kv in a.hp)
    hps = hyperparameters(steps=a.steps, batch_size=a.batch_size, save_freq=max(10, a.steps // 2), log_freq=min(10, a.steps),
                          device="cpu", num_workers=2, eval_stride=60, heldout=a.heldout, policy=a.policy, **extra)
    name = job_name(a.name, a.policy, "cpu")
    tags = {"project": project, "kind": "smoke", "policy": a.policy, **cfg.get("extra_job_tags", {})}
    spec = build_spec(name=name, image_uri=image, role_arn=stack["sagemaker_role_arn"], dataset_s3=dataset_s3,
                      output_s3=f"{root}/{SMOKE_PREFIX}/", checkpoint_s3=f"{root}/{SMOKE_PREFIX}/{name}/checkpoints/",
                      instance_type=a.instance_type, hps=hps, max_runtime_s=a.max_minutes * 60, volume_gb=100, tags=tags)
    sm = client(sess, "sagemaker")
    submit(sm, spec)
    print(f"submitted {name} on {a.instance_type}\n  image   : {image}\n  dataset : {dataset_s3}\n  hps     : {hps}")
    if a.no_wait:
        return 0
    try:
        d = wait_for_job(lambda: client(sess, "sagemaker"), name)
    except JobFailed as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 5
    key = f"{d['ModelArtifacts']['S3ModelArtifacts'].split('/', 3)[3]}"
    summary = read_summary(client(sess, "s3"), stack["bucket"], key)
    evals = summary.get("evals", {})
    mae = next((v.get("joints_mae") for v in evals.values()), None)
    ok, why = judge(mae, a.expect_joints_mae, a.mae_tol)
    print(f"SUMMARY: evals={list(evals)} joints_mae={mae} billable={d.get('BillableTimeInSeconds')}s -> {'PASS' if ok else 'FAIL'} ({why})")
    return 0 if ok else 5


if __name__ == "__main__":
    sys.exit(main())
