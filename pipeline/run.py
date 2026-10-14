#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""The one command: fine-tune the policies enabled in project_config.json on the dataset in datasets/<name>/.

    ./finetune.sh                       # = python3 pipeline/run.py
    ./finetune.sh --dry-run             # everything except AWS writes; prints the plan and both job specs
    ./finetune.sh --policy act --yes    # one policy, no prompts
    ./finetune.sh --instance-type auto  # per policy: the first registry instance type with quota in your account
    ./finetune.sh --train-job <name>    # reattach to a running job (lost terminal)
    ./finetune.sh --resume-from <job>   # continue an interrupted job from its checkpoints
    ./finetune.sh --resume-from <job> --extend-steps 150000   # train a finished 100k run on to 150k

Stages: config -> dataset (local check + compatibility, no AWS call) -> stack (SSM; waits for a running image build)
-> preflight -> stage (S3 sync) -> smoke (200 steps on the training instance, gate) -> train (one job per enabled policy, one child process each in parallel; --sequential runs
them in order in this process) -> report (SUMMARY vs reference band,
MANIFEST.json). The plan is printed and confirmed once before any AWS write (upload, staging, smoke and training jobs);
--plan stops after the table, --yes skips the question.
Exit codes: 0 ok | 2 config | 3 dataset | 4 preflight/image | 5 smoke gate | 6 training | 7 report band | 130 Ctrl-C
| 128+signal: a parallel training child killed by a signal (SIGKILL -> 137), see docs/ARCHITECTURE.md, fan-out.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline import hub, instance_table, logs, report, sm_jobs  # noqa: E402
from pipeline import compat, hf_token, plan  # noqa: E402
from pipeline.aws import StackNotFound, ambient_region, client, resolve_stack, s3_root, session  # noqa: E402
from pipeline.build_image import ensure_image  # noqa: E402
from botocore.exceptions import BotoCoreError, ClientError, EndpointConnectionError, NoCredentialsError  # noqa: E402
from pipeline.config import ConfigError, enabled_policies, load_project_config  # noqa: E402
from pipeline.preflight import PreflightError, resolve_auto_instances, run_preflight  # noqa: E402
from pipeline.wait import JobFailed, wait_for_job  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
TRAINER = REPO / "trainer"
RUNS = REPO / "runs"
sys.path.insert(0, str(TRAINER))
import render_flags  # noqa: E402

STAGES = ["config", "dataset", "stack", "preflight", "stage", "smoke", "train", "report"]


class Abort(SystemExit):
    def __init__(self, code: int, msg: str):
        print(f"ERROR: {msg}", file=sys.stderr)
        super().__init__(code)


def log(msg: str = "") -> None:
    print(msg, flush=True)


def confirm(question: str, yes: bool) -> None:
    if yes:
        return
    if not sys.stdin.isatty():
        raise Abort(2, f"{question} -- no terminal to confirm; pass --yes")
    if input(f"{question} [y/N] ").strip().lower() not in ("y", "yes"):
        raise Abort(130, "aborted by user")


# ---------------------------------------------------------------- dataset -------------------------------------------
def dataset_fingerprint(root: Path) -> tuple:
    """sha256 over meta/info.json + every file's relative path and size; also returns (count, bytes)."""
    h = hashlib.sha256()
    h.update((root / "meta" / "info.json").read_bytes())
    count, total = 0, 0
    for p in sorted(root.rglob("*")):
        if p.is_file() and ".cache" not in p.parts:
            st = p.stat(); count += 1; total += st.st_size
            h.update(f"{p.relative_to(root)}:{st.st_size}".encode())
    return h.hexdigest(), count, total


def ensure_dataset(cfg: dict) -> Path:
    ds = cfg["dataset"]
    root = REPO / "datasets" / ds["name"]
    if ds["source"] == "local":
        if not (root / "meta" / "info.json").is_file():
            raise Abort(3, f"datasets/{ds['name']}/meta/info.json not found: put your LeRobot v3.0 dataset folder under datasets/ (see datasets/README.md)")
    elif ds["source"] == "huggingface":
        if not (root / "meta" / "info.json").is_file():
            log(f"  downloading {ds['hf_repo_id']}@{ds.get('hf_revision') or 'main'} -> datasets/{ds['name']}/")
            try:
                from huggingface_hub import snapshot_download
            except ImportError:
                raise Abort(3, "pip install huggingface_hub (or download the dataset into datasets/<name>/ yourself)")
            # revision is pinned in project_config.json (dataset.hf_revision); bandit B615 wants a literal revision
            snapshot_download(repo_id=ds["hf_repo_id"], repo_type="dataset", revision=ds.get("hf_revision"), local_dir=str(root))  # nosec B615
    return root


def check_dataset(root: Path, cfg: dict, require_flat: bool) -> dict:
    ev = cfg.get("evaluation", {})
    args = [sys.executable, str(TRAINER / "check_dataset.py"), str(root), "--emit-split"]
    heldout = ev.get("heldout", "auto")
    if heldout != "auto":
        args += ["--heldout", " ".join(map(str, heldout)) if isinstance(heldout, list) else str(heldout)]
    if require_flat:
        args.append("--require-flat-names")
    r = subprocess.run(args, capture_output=True, text=True)
    for line in r.stderr.splitlines():
        log(f"  {line}")
    if r.returncode != 0:
        raise Abort(3, "dataset check failed (see report above)")
    return json.loads(r.stdout)


def _s3_dataset_facts(s3, s3_uri: str):
    """fps and a fingerprint for a dataset that already lives in S3 (source=s3): read its meta/info.json."""
    bucket, prefix = s3_uri[5:].split("/", 1)
    key = prefix.rstrip("/") + "/meta/info.json"
    try:
        body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    except ClientError as e:
        raise Abort(3, f"dataset.s3_uri has no meta/info.json at s3://{bucket}/{key}: {e.response.get('Error', {}).get('Code')}") from None
    info = json.loads(body)
    if str(info.get("codebase_version")) != "v3.0":
        raise Abort(3, f"dataset at {s3_uri} is {info.get('codebase_version')}, not v3.0")
    count, nbytes = 0, 0   # the job volume is sized from the dataset bytes
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix.rstrip("/") + "/"):
        for o in page.get("Contents", []):
            count += 1; nbytes += int(o.get("Size", 0))
    return {"fps": info.get("fps"), "image_keys": _image_keys(info), "_info": info}, "s3-" + hashlib.sha256(body).hexdigest(), count, nbytes


def _image_keys(info: dict) -> list:
    return sorted(k for k, v in info.get("features", {}).items() if v.get("dtype") in ("video", "image"))


def _camera_notes(pols: dict, specs: dict, keys: list, log) -> None:
    """Single-camera policies train on one image feature: say which, and refuse a camera the dataset does not have."""
    for name, pol in pols.items():
        cam = pol.get("camera")
        if not specs[name].get("single_camera_required"):
            if cam:
                log(f"  NOTE: policies.{name}.camera is ignored ({name} uses every camera)")
            continue
        if cam and cam not in keys:
            raise Abort(3, f"policies.{name}.camera = {cam!r} is not a camera of this dataset: {keys}")
        if len(keys) > 1:
            keep = cam or keys[0]
            log(f"  NOTE: {name} accepts one camera: the job trains on {keep} and ignores {[k for k in keys if k != keep]}"
                f"{'' if cam else f' (default: first in sorted order; set policies.{name}.camera to choose)'}")


def _child_argv(argv: list, policy: str, run_name: str, instance_type: str = None) -> list:
    """The CLI of one child process: the parent's arguments minus policy selection, plan and naming, plus this policy.
    With instance_type given, the parent's --instance-type (possibly 'auto') is replaced by the type it resolved and
    printed in the plan, so a child never re-resolves 'auto' against the quotas and cannot pick a different type."""
    out, skip = [], False
    for tok in argv:
        if skip:
            skip = False; continue
        if tok in ("--policy", "--name") or (instance_type and tok == "--instance-type"):
            skip = True; continue
        if tok.startswith("--policy=") or tok.startswith("--name=") or (instance_type and tok.startswith("--instance-type=")):
            continue
        if tok in ("--yes", "--plan", "--sequential", "--no-plan"):
            continue
        out.append(tok)
    extra = ["--instance-type", instance_type] if instance_type else []
    return [sys.executable, "-m", "pipeline.run", *out, *extra, "--policy", policy, "--name", run_name, "--yes", "--sequential", "--no-plan"]


def _child_exit_code(code: int) -> int:
    """Popen reports a signal-killed child as -signal; the shell convention 128 + signal keeps it non-zero (SIGKILL -> 137)."""
    return 128 - code if code < 0 else code


def _run_parallel(a, names: list, run_name: str, instances: dict = None) -> int:
    """One child process per policy (each is its own SageMaker smoke + training job); logs under runs/<run>-<policy>.log.

    Returns the highest child exit code; a child killed by a signal counts as 128 + the signal number."""
    RUNS.mkdir(exist_ok=True)
    procs = {}
    for name in names:
        logf = RUNS / f"{run_name}-{name}.log"
        fh = open(logf, "w")
        procs[name] = (subprocess.Popen(_child_argv(sys.argv[1:], name, run_name, (instances or {}).get(name)), stdout=fh, stderr=subprocess.STDOUT, cwd=str(REPO)), logf, fh)
        log(f"[{name}] started (pid {procs[name][0].pid}); log: {logf}")
    rc = 0
    for name, (p, logf, fh) in procs.items():
        code = p.wait()
        fh.close()
        tail = logf.read_text().splitlines()[-8:] if logf.is_file() else []
        log(f"\n[{name}] finished with exit {code}:\n  " + "\n  ".join(tail))
        rc = max(rc, _child_exit_code(code))
    return rc


def _keep_checkpoints(cfg: dict, pol: dict) -> int:
    """policies.<name>.keep_checkpoints overrides compute.keep_checkpoints (default 3); null means no override."""
    keep = pol.get("keep_checkpoints")
    if keep is None:
        keep = cfg["compute"].get("keep_checkpoints", 3)
    return keep


def _requested_instance(a, cfg: dict, pol: dict) -> str:
    """The instance type asked for: --instance-type, then policies.<name>.instance_type, then compute.instance_type ('auto' allowed in each)."""
    return a.instance_type or pol.get("instance_type") or cfg["compute"]["instance_type"]


def _policy_spot(a, cfg: dict, pol: dict) -> bool:
    """--spot, then policies.<name>.spot, then compute.spot."""
    if a.spot is not None:
        return bool(a.spot)
    if pol.get("spot") is not None:
        return bool(pol["spot"])
    return bool(cfg["compute"].get("spot", False))


def _dataset_cfg(cfg: dict, pol: dict) -> dict:
    """The dataset block for one policy: the shared block with the policy's `dataset` keys laid over it."""
    over = {k: v for k, v in (pol.get("dataset") or {}).items() if not k.startswith("_")}
    return {**cfg["dataset"], **over} if over else cfg["dataset"]


def _dataset_groups(cfg: dict, pols: dict) -> dict:
    """dataset name -> [policy names] in config order (the shared dataset first when any policy uses it)."""
    groups = {}
    for name, pol in pols.items():
        groups.setdefault(_dataset_cfg(cfg, pol)["name"], []).append(name)
    return groups


def _prepare_dataset(a, pcfg: dict, sess, stack: dict, specs: dict, pols: dict, all_specs: dict = None, do_stage: bool = True) -> dict:
    """check -> fingerprint (-> stage) for one dataset block; returns what the per-policy loop needs (dataset_s3 only after staging)."""
    all_specs = all_specs or specs
    root = ensure_dataset(pcfg) if pcfg["dataset"]["source"] != "s3" else None
    if root is not None:
        split = check_dataset(root, pcfg, any(s.get("requires_flat_action_names") for s in specs.values()))
        split["image_keys"] = _image_keys(json.loads((root / "meta" / "info.json").read_text()))
        fingerprint, count, nbytes = dataset_fingerprint(root)
    else:
        split, fingerprint, count, nbytes = _s3_dataset_facts(client(sess, "s3"), pcfg["dataset"]["s3_uri"])
    facts = compat.dataset_facts(root) if root is not None else compat.facts_from_info(split.get("_info") or {})
    for line in compat.report(facts, all_specs, list(pols)):
        log("  " + line)
    blocked = compat.blocked_enabled(facts, all_specs, list(pols), pols)
    if blocked:
        raise Abort(3, "; ".join(f"{n} cannot fine-tune on {pcfg['dataset']['name']}: {'; '.join(w)}" for n, w in blocked.items()) + " (disable it or give it its own policies.<name>.dataset)")
    _camera_notes(pols, specs, split.get("image_keys") or [], log)
    _quantile_notes(specs, root, log)
    prep = {"root": root, "split": split, "fingerprint": fingerprint, "count": count, "nbytes": nbytes, "dataset_s3": None, "pcfg": pcfg}
    return _stage_group(a, sess, stack, prep) if do_stage else prep


def _stage_group(a, sess, stack: dict, prep: dict) -> dict:
    pcfg, root = prep["pcfg"], prep["root"]
    log("stage:")
    if root is not None:
        prep["dataset_s3"] = stage_dataset(sess, stack, root, pcfg["dataset"]["name"], prep["fingerprint"], prep["count"], prep["nbytes"], yes=a.yes, dry=a.dry_run, profile=a.profile)
    else:
        prep["dataset_s3"] = pcfg["dataset"]["s3_uri"]
    return prep


def _quantile_notes(specs: dict, root, log) -> None:
    """QUANTILES policies need q01/q99 in stats.json; the job completes them, say so instead of failing after the base load."""
    need = [n for n, s in specs.items() if s.get("requires_quantile_stats")]
    if not need or root is None:
        return
    try:
        stats = json.loads((root / "meta" / "stats.json").read_text())
    except (OSError, json.JSONDecodeError):
        return
    lacking = [k for k in ("action", "observation.state") if any(q not in (stats.get(k) or {}) for q in ("q01", "q99"))]
    if lacking:
        log(f"  NOTE: {', '.join(need)} normalise with QUANTILES and stats.json lacks q01/q99 for {lacking}: the job computes them from the data (ensure_quantiles.py)")


def _stale_remote(s3, bucket: str, prefix: str, root: Path) -> list:
    """Keys under the dataset prefix without a local counterpart: `aws s3 sync --delete` removes them."""
    local = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and ".cache" not in p.parts}
    stale = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for o in page.get("Contents", []):
            if o["Key"][len(prefix):] not in local:
                stale.append(o["Key"])
    return stale


# ---------------------------------------------------------------- staging -------------------------------------------
def stage_dataset(sess, stack: dict, root: Path, name: str, fingerprint: str, count: int, nbytes: int, *, yes: bool, dry: bool, profile: str = None) -> str:
    s3 = client(sess, "s3")
    prefix = f"sagemaker/datasets/{name}/"
    uri = f"s3://{stack['bucket']}/{prefix}"
    marker = f"sagemaker/datasets/_staged/{name}.json"   # outside the synced prefix (sync --delete would remove it)
    try:
        staged = json.loads(s3.get_object(Bucket=stack["bucket"], Key=marker)["Body"].read())
        if staged.get("fingerprint") == fingerprint:
            log(f"  already staged (fingerprint {fingerprint[:12]}): {uri}")
            return uri
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") not in ("NoSuchKey", "404", "NotFound"):
            raise
    stale = _stale_remote(s3, stack["bucket"], prefix, root)
    log(f"  will upload {count} files, {nbytes / 1e6:.1f} MB -> {uri}" + (f"; the mirror deletes {len(stale)} stale object(s) under that prefix" if stale else ""))
    if dry:
        return uri
    confirm("Upload the dataset to your bucket" + (f" and delete {len(stale)} stale object(s) under {uri}" if stale else "") + "?", yes)
    if shutil.which("aws") is None:
        raise Abort(3, "the AWS CLI v2 is required for `aws s3 sync` (pip-free install: https://docs.aws.amazon.com/cli/)")
    # --delete: a re-exported dataset with fewer files must not leave stale parquet/mp4 objects for the job to glob
    cmd = ["aws", "s3", "sync", str(root) + "/", uri, "--delete", "--exclude", ".cache/*", "--only-show-errors", "--region", sess.region_name]
    if profile:  # only an explicitly chosen profile; boto3 reports 'default' even for env credentials
        cmd += ["--profile", profile]
    try:
        subprocess.run(cmd, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        raise Abort(3, f"aws s3 sync failed ({e}); check the AWS CLI v2 install and your credentials") from None
    s3.put_object(Bucket=stack["bucket"], Key=marker, Body=json.dumps({"fingerprint": fingerprint, "files": count, "bytes": nbytes, "staged_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}).encode())
    return uri


# ---------------------------------------------------------------- jobs ----------------------------------------------
def policy_hps(cfg: dict, name: str, pol: dict, spec: dict, split: dict, *, steps: int, smoke: bool, image_digest: str, fingerprint: str) -> dict:
    d = spec.get("defaults", {})
    ev = cfg.get("evaluation", {})
    fps = split.get("fps") or 25
    chunk = pol.get("chunk") or (max(1, round(float(pol.get("horizon_seconds", 1.0)) * fps)) if pol.get("horizon_seconds") else d.get("chunk", 24))
    heldout = ev.get("heldout", "auto")
    return sm_jobs.hyperparameters(
        policy=name, steps=steps, batch_size=pol.get("batch_size", d.get("batch_size", 8)), chunk=chunk,
        save_freq=min(pol.get("save_freq", d.get("save_freq", 10000)), steps), log_freq=(spec.get("smoke", {}).get("log_freq", 50) if smoke else d.get("log_freq", 200)),
        seed=pol.get("seed", d.get("seed", 1000)),
        heldout=None if heldout == "auto" else (" ".join(map(str, heldout)) if isinstance(heldout, list) else heldout),
        eval_stride=(60 if smoke else ev.get("stride", 1)), eval_ckpts=" ".join(ev.get("checkpoints", ["last"])),
        downscale_height=cfg["dataset"].get("downscale_height"), downscale=None if cfg["dataset"].get("downscale_height") else 0,
        force_res=None if cfg["dataset"].get("downscale_height") else 1, overrides=" ".join(pol.get("overrides", [])) or None,
        accept_licence=1 if pol.get("accept_licence") else None, approval_ref=pol.get("approval_ref"),
        smoke=1 if smoke else None, image_digest=image_digest, dataset_fingerprint=fingerprint,
        require_flat_names=1 if spec.get("requires_flat_action_names") else None,
        camera=pol.get("camera"), cameras=",".join(pol.get("cameras") or []) or None,
        keep_checkpoints=None if smoke else _keep_checkpoints(cfg, pol),
    )


def job_spec(cfg: dict, stack: dict, name: str, pol: dict, spec: dict, hps: dict, *, kind: str, instance: str, spot: bool,
             image: str, dataset_s3: str, max_runtime_s: int, volume_gb: int, run_name: str, hub_s3: str = None) -> dict:
    tplan = cfg["compute"].get("training_plan_arn")   # not `plan`: that is the pipeline.plan module
    root = s3_root(stack)
    prefix = sm_jobs.SMOKE_PREFIX if kind == "smoke" else sm_jobs.TRAIN_PREFIX
    job = sm_jobs.job_name(run_name, name, kind)
    tags = {"project": cfg["project"], "policy": name, "kind": kind, **cfg.get("extra_job_tags", {})}
    return sm_jobs.build_spec(name=job, image_uri=image, role_arn=stack["sagemaker_role_arn"], dataset_s3=dataset_s3,
                              output_s3=f"{root}/{prefix}/", checkpoint_s3=f"{root}/{prefix}/{job}/checkpoints/", instance_type=instance,
                              hps=hps, max_runtime_s=max_runtime_s, volume_gb=volume_gb, spot=spot and not tplan, training_plan_arn=tplan, tags=tags, hub_s3=hub_s3)


def read_summary_from_checkpoints(s3, bucket: str, checkpoint_s3: str, policy: str) -> dict:
    key = checkpoint_s3.split("/", 3)[3].rstrip("/") + f"/{policy}/SUMMARY.json"
    return json.loads(s3.get_object(Bucket=bucket, Key=key)["Body"].read())


def smoke_gate(summary: dict, spec: dict) -> tuple:
    """(ok, reasons, step_rate). Never uses 'beats hold': a 200-step policy is worse than hold and that is expected."""
    reasons = []
    e = report.first_eval(summary)
    if not e or e.get("joints_mae") is None or not math.isfinite(float(e["joints_mae"])):
        reasons.append("no finite joints MAE in SUMMARY.evals")
    if summary.get("train_exitcode") not in (0, None):
        reasons.append(f"trainer exit code {summary.get('train_exitcode')}")
    sm = summary.get("smoke") or {}
    if sm.get("error"):
        reasons.append(f"smoke checks errored: {sm['error']}")
    else:
        if sm.get("weights_moved_fraction") is not None and sm["weights_moved_fraction"] < 0.5:
            reasons.append(f"only {sm['weights_moved_fraction']:.0%} of trainable tensors moved: the run trained nothing or loaded nothing")
        if sm.get("bf16_trainable_params"):
            reasons.append(f"{sm['bf16_trainable_params']} trainable parameters stored in bf16 (updates below the bf16 ULP are discarded)")
    rc = summary.get("reload_check") or {}
    if rc and not rc.get("ok", True):
        reasons.append(f"published checkpoint failed the offline reload check: {rc.get('error') or rc.get('max_abs_diff_vs_eval')}")
    return (not reasons), reasons, summary.get("step_rate")


def _state_for_job(job: str):
    for p in RUNS.glob("*.json"):
        try:
            st = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        if st.get("train_job") == job:
            return st
    return None


def _passed_smoke(policy: str, hps: dict, instance: str):
    """A previous run's smoke that PASSED the gate with identical image digest, dataset fingerprint, hyperparameters and instance type."""
    for p in sorted(RUNS.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            st = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        if st.get("policy") == policy and st.get("smoke_hps") == hps and st.get("instance_type") == instance and st.get("smoke_passed") is True and st.get("smoke_summary"):
            return {"smoke_job": st.get("smoke_job"), "step_rate": st["smoke_summary"].get("step_rate")}
    return None


def save_run(state: dict) -> Path:
    RUNS.mkdir(exist_ok=True)
    p = RUNS / f"{state['name']}.json"
    p.write_text(json.dumps(state, indent=2, default=str))
    return p


def _safe_extract(tf: "tarfile.TarFile", dest: Path) -> None:
    """Extract with the stdlib 'data' filter (Python >= 3.9.17 / 3.12); older interpreters get an explicit member check."""
    try:
        tf.extractall(dest, filter="data")
        return
    except TypeError:  # no filter= support
        pass
    root = dest.resolve()
    for m in tf.getmembers():
        target = (dest / m.name).resolve()
        if m.islnk() or m.issym() or m.isdev() or not str(target).startswith(str(root) + os.sep):
            raise RuntimeError(f"refusing archive member {m.name!r}")
    tf.extractall(dest)  # nosec B202 - members validated above


def download_outputs(s3, bucket: str, describe: dict, dest: Path) -> dict:
    dest.mkdir(parents=True, exist_ok=True)
    art = describe["ModelArtifacts"]["S3ModelArtifacts"]
    key = art.split("/", 3)[3]
    body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    with tarfile.open(fileobj=io.BytesIO(body), mode="r:gz") as tf:
        _safe_extract(tf, dest / "model")
    out_key = key.replace("model.tar.gz", "output.tar.gz")   # optional: SageMaker writes it only when the job left files in /opt/ml/output
    try:
        body = s3.get_object(Bucket=bucket, Key=out_key)["Body"].read()
        with tarfile.open(fileobj=io.BytesIO(body), mode="r:gz") as tf:
            _safe_extract(tf, dest / "output")
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") not in ("NoSuchKey", "404", "NotFound"):
            log(f"  NOTE: s3://{bucket}/{out_key} not downloaded: {e}")
    except (BotoCoreError, tarfile.TarError, RuntimeError, OSError) as e:   # a transport error, a corrupt archive or a refused member: the report needs model.tar.gz only
        log(f"  NOTE: s3://{bucket}/{out_key} not downloaded or unpacked: {e}")
    return json.loads((dest / "model" / "SUMMARY.json").read_text())


def download_summary_only(s3, bucket: str, describe: dict, policy: str, dest: Path) -> dict:
    """SUMMARY.json and eval/ from the job's mirrored checkpoint prefix (a few MB), leaving model.tar.gz in S3."""
    (dest / "model" / "eval").mkdir(parents=True, exist_ok=True)
    prefix = describe["CheckpointConfig"]["S3Uri"].split("/", 3)[3].rstrip("/") + f"/{policy}/"
    body = s3.get_object(Bucket=bucket, Key=prefix + "SUMMARY.json")["Body"].read()
    (dest / "model" / "SUMMARY.json").write_bytes(body)
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix + "eval/"):
        for o in page.get("Contents", []):
            name = o["Key"][len(prefix) + len("eval/"):]
            if name and "/" not in name:
                (dest / "model" / "eval" / name).write_bytes(s3.get_object(Bucket=bucket, Key=o["Key"])["Body"].read())
    return json.loads(body)


# ---------------------------------------------------------------- main -----------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="alternative project_config.json (default: the repo root one)")
    ap.add_argument("--policy", action="append", help="only this enabled policy (repeatable)")
    ap.add_argument("--region", help="AWS region of the deployed stack (default: AWS_DEFAULT_REGION, then AWS_REGION)")
    ap.add_argument("--profile", help="AWS CLI profile for the credentials (default: the ambient credentials)")
    ap.add_argument("--name", help="run name (default: dataset name)")
    ap.add_argument("--instance-type", metavar="TYPE", help="SageMaker instance type for every job of this run, or 'auto': per policy, the first entry of the registry's "
                    "hardware.instance_types that the instance rules accept and whose quota (Spot or on-demand, as the policy runs) is above 0; "
                    "overrides compute.instance_type and policies.<name>.instance_type")
    ap.add_argument("--spot", action="store_true", default=None, help="run the GPU jobs on managed Spot (checkpoint resume on interruption); overrides compute.spot")
    ap.add_argument("--training-plan", metavar="ARN", help="run the GPU jobs on this SageMaker training plan (reserved capacity) instead of on-demand/Spot; overrides compute.training_plan_arn")
    ap.add_argument("--image", help="digest-pinned image URI to use instead of the stack's published one (development)")
    ap.add_argument("--compat", action="store_true", help="only print which registry policies can fine-tune on the configured dataset(s) (no AWS calls), then exit")
    ap.add_argument("--dry-run", action="store_true", help="everything except AWS writes: no upload, no jobs; prints the plan and both job specs")
    ap.add_argument("--yes", action="store_true", help="skip the plan confirmation and every later prompt (uploads, stale-object deletes, job submits)")
    ap.add_argument("--skip-smoke", action="store_true", help="submit the training job without the 200-step smoke gate (the step rate then comes from the registry table, not from a measurement)")
    ap.add_argument("--no-wait", action="store_true", help="return after the training job is submitted; reattach later with --train-job <name>")
    ap.add_argument("--plan", action="store_true", help="print the submission plan (policies, datasets, instances, estimated hours and dollars) and exit")
    ap.add_argument("--sequential", action="store_true", help="run several policies one after another in this process (default: one child process per policy, submitted in parallel)")
    ap.add_argument("--no-plan", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--no-download", action="store_true", help="report from the mirrored SUMMARY.json and eval files only; do not download model.tar.gz (about 8.5 GB compressed, 16 GB unpacked, for the pi0 family)")
    ap.add_argument("--allow-multigpu", action="store_true", help="accept a multi-GPU instance type (the trainer uses one GPU; the others are billed idle)")
    ap.add_argument("--allow-stubs", action="store_true", help="run registry entries with status stub (the smoke that promotes them)")
    ap.add_argument("--stop-after", choices=STAGES[:STAGES.index("train")], help="exit 0 after this stage (config, dataset, stack, preflight, stage or smoke)")
    ap.add_argument("--train-job", help="reattach to this training job")
    ap.add_argument("--resume-from", help="continue from this job's checkpoints (new job, resume=1)")
    ap.add_argument("--extend-steps", type=int, help="with --resume-from: continue the finished run to this total step count "
                    "(ACT: constant LR, a true continuation; decay-scheduler policies: a warm restart, reported as two phases)")
    a = ap.parse_args(argv)
    if a.extend_steps and not a.resume_from:
        print("ERROR: --extend-steps needs --resume-from <job>", file=sys.stderr); return 2
    try:
        return _main(a)
    except Abort as e:
        return int(e.code)
    except JobFailed as e:
        print(f"ERROR: {e}", file=sys.stderr); return 6
    except StackNotFound as e:
        print(f"ERROR: {e}", file=sys.stderr); return 4
    except json.JSONDecodeError as e:    # a malformed JSON object (dataset meta, staged marker, SUMMARY.json)
        print(f"ERROR: malformed JSON: {e}", file=sys.stderr); return 3
    except ValueError as e:               # e.g. no region anywhere
        print(f"ERROR: {e}", file=sys.stderr); return 2
    except (NoCredentialsError, ClientError, BotoCoreError) as e:   # BotoCoreError covers EndpointConnectionError and the other transport failures
        print(f"ERROR: AWS call failed: {e}", file=sys.stderr); return 4
    except KeyboardInterrupt:
        print("\ninterrupted; running jobs keep running -- reattach with --train-job <name>", file=sys.stderr)
        return 130


def _main(a) -> int:
    # -- config
    try:
        cfg = load_project_config(Path(a.config)) if a.config else load_project_config()
    except ConfigError as e:
        raise Abort(2, str(e))
    pols = enabled_policies(cfg)
    if a.policy:   # --policy selects these policies whether or not they are enabled in the file
        allp = {k: v for k, v in cfg["policies"].items() if not k.startswith("_")}
        missing = [p for p in a.policy if p not in allp]
        if missing:
            raise Abort(2, f"--policy {missing}: not in project_config.json (known policies: {sorted(allp)})")
        pols = {k: allp[k] for k in a.policy}
        for k in a.policy:
            if not allp[k].get("enabled"):
                log(f"  NOTE: {k} is enabled: false in project_config.json; --policy selects it anyway")
    if a.resume_from and len(pols) != 1:   # a checkpoint prefix belongs to one job; the parent would hand it to every child
        raise Abort(2, f"--resume-from continues one job, but {len(pols)} policies are selected ({', '.join(pols)}): add --policy <name>")
    specs = {k: render_flags.load_policy(k) for k in pols}
    all_specs = {p.stem: render_flags.load_policy(p.stem) for p in sorted(render_flags.POLICIES_DIR.glob("*.yaml"))}
    if a.compat:   # local only: download/check the dataset(s) and print the compatibility table for every registry policy
        rc = 0
        for ds_name, group in _dataset_groups(cfg, pols).items():
            pcfg = {**cfg, "dataset": _dataset_cfg(cfg, pols[group[0]])}
            if pcfg["dataset"]["source"] == "s3":
                log(f"dataset {ds_name}: source=s3, compatibility needs the meta/ folder locally (put a copy under datasets/{ds_name}/)"); continue
            root = ensure_dataset(pcfg)
            facts = compat.dataset_facts(root)
            for line in compat.report(facts, all_specs, list(group)):
                log(line)
            for name, why in compat.blocked_enabled(facts, all_specs, list(group), pols).items():
                log(f"  ERROR: enabled policy {name} cannot fine-tune on {ds_name}: {'; '.join(why)}"); rc = 3
        return rc
    cfg = dict(cfg); cfg["policies"] = {k: cfg["policies"][k] for k in pols}   # --policy narrows every later stage, incl. preflight
    run_name = a.name or cfg["dataset"]["name"]
    region = ambient_region(a.region)
    sess = session(region, a.profile)
    log(f"config: project={cfg['project']} region={region} dataset={cfg['dataset']['name']} policies={', '.join(pols)}")
    if a.stop_after == "config":
        return 0
    if a.train_job:  # reattach: reuse the state run.py saved when it submitted the job
        stack = resolve_stack(sess, cfg["project"])
        state = _state_for_job(a.train_job) or {"name": f"{run_name}-reattached", "policy": None, "dataset_name": cfg["dataset"]["name"]}
        d = wait_for_job(lambda: client(sess, "sagemaker"), a.train_job, log=log)
        state["policy"] = state.get("policy") or d["HyperParameters"].get("policy", "act")
        return _report(a, cfg, sess, client(sess, "s3"), stack, state, d)
    # -- dataset: check + compatibility first, once per distinct dataset (policies.<name>.dataset overrides the shared
    #    block); nothing is uploaded yet, and no stack is needed, so a bad recording fails before any AWS call
    preps = {}
    for ds_name, group in _dataset_groups(cfg, pols).items():
        pcfg = {**cfg, "dataset": _dataset_cfg(cfg, pols[group[0]])}
        log(f"dataset ({', '.join(group)}): {ds_name} (per-policy override)" if pcfg["dataset"] is not cfg["dataset"] else "dataset:")
        preps[ds_name] = _prepare_dataset(a, pcfg, sess, None, {k: specs[k] for k in group}, {k: pols[k] for k in group}, all_specs, do_stage=False)
    if a.stop_after == "dataset":
        return 0
    # -- stack, then the image: `cdk deploy` returns before the image build (about 15 min) has finished, so a run that
    #    starts right after the deploy waits for that build instead of failing
    stack = resolve_stack(sess, cfg["project"], use_cache=not a.image)   # a fresh publish must not be hidden by the 10-minute cache
    if not a.image:
        try:
            stack = ensure_image(sess, stack, cfg["project"], log=log)
        except RuntimeError as e:
            raise Abort(4, str(e))
    image = a.image or stack["image_uri"]
    log(f"stack: bucket={stack['bucket']} image=@sha256:{image.split('@sha256:')[1][:12]}..." if "@sha256:" in image else f"stack: bucket={stack['bucket']} image={image}")
    if a.stop_after == "stack":
        return 0
    sm = client(sess, "sagemaker"); s3 = client(sess, "s3")
    spot = cfg["compute"].get("spot", False) if a.spot is None else a.spot
    tplan = a.training_plan or cfg["compute"].get("training_plan_arn")   # not `plan`: that is the pipeline.plan module
    if tplan:   # reserved capacity: the GPU jobs run on the plan, never on Spot
        if a.spot:
            raise Abort(2, "--spot and a training plan are exclusive (a plan is reserved capacity)")
        if spot or any(pol.get("spot") for pol in pols.values()):
            log(f"  NOTE: training plan {tplan.rsplit('/', 1)[-1]}: Spot is off for every policy of this run")
        spot = False
        cfg = {**cfg, "compute": {**cfg["compute"], "spot": False, "training_plan_arn": tplan}, "policies": {k: {**v, "spot": False} for k, v in cfg["policies"].items()}}
        pols = {k: cfg["policies"][k] for k in pols}
    # -- instance type per policy: CLI, then policies.<name>.instance_type, then compute.instance_type. 'auto' walks the
    #    registry's hardware.instance_types against the account's quotas (read-only call) so that preflight and the plan
    #    see the resolved type; a run that can get no instance stops here (exit 4) with the quota names to request
    instances = {name: _requested_instance(a, cfg, pol) for name, pol in pols.items()}
    auto = {name: _policy_spot(a, cfg, pols[name]) for name, it in instances.items() if it == "auto"}
    if auto:
        if tplan:
            raise Abort(2, f"instance type 'auto' and a training plan are exclusive: the plan reserves one instance type; name it for {', '.join(auto)}")
        try:
            instances.update(resolve_auto_instances(auto, specs, lambda n: client(sess, n), region, allow_multigpu=a.allow_multigpu, log=log))
        except PreflightError as e:
            raise Abort(4, str(e))
    # -- preflight
    cands = sorted(set(instances.values()))
    cands += [t for t in cfg["compute"].get("fallback_instance_types", []) if t not in cands]   # fallbacks get the same quota/GPU checks
    if any(specs[k].get("hub_artifacts") for k in pols):
        cands.append(hub.STAGE_INSTANCE)   # the base-weight staging job runs on a CPU instance (on-demand quota)
    log("preflight:")
    try:
        run_preflight(cfg, stack, sess, specs, candidates=cands, spot=spot, image_uri=image, allow_multigpu=a.allow_multigpu, allow_stubs=a.allow_stubs, log=log, training_plan_arn=tplan)
    except PreflightError as e:
        raise Abort(4, str(e))
    if a.stop_after == "preflight":
        return 0
    # -- plan: one table, one confirmation, before any upload or job
    if not a.no_plan:
        rows, uploads = [], []
        for name, pol in pols.items():
            inst, sp = instances[name], _policy_spot(a, cfg, pol)
            rows.append(plan.policy_row(name, pol, specs[name], cfg, _dataset_cfg(cfg, pol)["name"], inst, sp, instance_table.step_rate(specs[name], inst), instance_table.spec(inst)))
        for ds_name, prep in preps.items():
            if prep.get("root") is not None:
                uploads.append(f"{ds_name}: {prep['count']} files, {(prep['nbytes'] or 0) / 1e6:.0f} MB -> s3://{stack['bucket']}/sagemaker/datasets/{ds_name}/ (skipped when already staged)")
        log("\nplan:\n" + plan.table(rows, uploads))
        blocked = [r["policy"] for r in rows if r["blocked"]]
        if blocked:
            raise Abort(4, f"cannot submit {blocked}: see the licence / gpu columns above (accept_licence, approval_ref, a bigger instance type)")
        if a.plan:
            return 0
        if not a.dry_run:
            confirm(f"Submit {len(pols)} job(s) as planned ({'in parallel' if len(pols) > 1 and not a.sequential else 'in order'})?", a.yes)
            a.yes = True   # the plan was confirmed once; no per-upload / per-job prompts follow
    for ds_name in preps:
        preps[ds_name] = _stage_group(a, sess, stack, preps[ds_name])
    if a.stop_after == "stage":
        return 0
    if len(pols) > 1 and not a.sequential and not a.dry_run:
        return _run_parallel(a, list(pols), run_name, instances)
    # -- per policy: smoke -> train -> report
    rc = 0
    for name, pol in pols.items():
        spec = specs[name]
        pcfg = {**cfg, "dataset": _dataset_cfg(cfg, pol)}
        prep = preps[pcfg["dataset"]["name"]]
        split, fingerprint, count, nbytes, dataset_s3 = prep["split"], prep["fingerprint"], prep["count"], prep["nbytes"], prep["dataset_s3"]
        instance = instances[name]   # CLI wins, 'auto' already resolved
        pol_spot = _policy_spot(a, cfg, pol)
        steps = pol.get("steps", spec.get("defaults", {}).get("steps", 100000))
        state = {"name": f"{run_name}-{name}", "run": run_name, "policy": name, "dataset_name": pcfg["dataset"]["name"], "dataset_s3": dataset_s3,
                 "dataset_fingerprint": fingerprint, "dataset_objects": count, "dataset_bytes": nbytes, "instance_type": instance, "spot": pol_spot,
                 "image": image, "region": region, "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        step_rate = instance_table.step_rate(spec, instance)
        hw = spec.get("hardware") or {}
        vol = cfg["compute"].get("volume_gb") or sm_jobs.volume_gb(
            nbytes or 0, n_checkpoints=(_keep_checkpoints(cfg, pol) or 20), ckpt_gb=float(hw.get("checkpoint_gb", 0.7)), hub_gb=float(hw.get("hub_gb", 0)))
        hub_s3 = None
        if spec.get("hub_artifacts"):
            log(f"\n[{name}] base weights:")
            if any(art.get("gated") for art in spec["hub_artifacts"]):   # the token: stored once, validated, never in the repo
                sname = hf_token.secret_name(cfg)
                try:
                    hf_token.ensure(client(sess, "secretsmanager"), sname, log=log)
                except RuntimeError as e:
                    raise Abort(4, str(e))
                cfg = {**cfg, "secrets": {**(cfg.get("secrets") or {}), "hf_token": sname}}
            try:
                hub_s3 = hub.stage_hub(sess, stack, cfg, name, spec, image or "x@sha256:" + "0" * 64, run_name=run_name, dataset_s3=dataset_s3, yes=a.yes, dry=a.dry_run, log=log, confirm=confirm)
            except RuntimeError as e:
                raise Abort(4, str(e))
            state["hub_s3"] = hub_s3
        # smoke
        hps = None   # set only when a smoke job is actually needed
        smoke_cfg = cfg.get("smoke") or {}   # the block is optional: enabled, 200 steps, reuse
        if smoke_cfg.get("enabled", True) and not a.skip_smoke:
            s_steps = smoke_cfg.get("steps", spec.get("smoke", {}).get("steps", 200))
            hps = policy_hps(pcfg, name, pol, spec, split, steps=s_steps, smoke=True, image_digest=image.split("@")[1] if image else "", fingerprint=fingerprint)
            prior = _passed_smoke(name, hps, instance) if smoke_cfg.get("reuse", True) else None
            if prior:
                log(f"\n[{name}] smoke: reusing {prior['smoke_job']} (same image, dataset, hyperparameters, instance)")
                if prior.get("step_rate"):
                    step_rate = prior["step_rate"]
                hps = None
        if hps is not None:   # a smoke is wanted and no earlier PASS covers it
            sspec = job_spec(cfg, stack, name, pol, spec, hps, kind="smoke", instance=instance, spot=pol_spot, image=image or "x@sha256:" + "0" * 64,   # the smoke uses the same capacity as the training job
                             dataset_s3=dataset_s3, max_runtime_s=int(spec.get("smoke", {}).get("max_minutes", 60)) * 60, volume_gb=vol, run_name=run_name, hub_s3=hub_s3)
            log(f"\n[{name}] smoke: {s_steps} steps on {instance}{' (Spot)' if pol_spot else ''} (~<$1)")
            if a.dry_run:
                log(json.dumps(sspec, indent=2)); log("  (dry run: not submitted)")
            else:
                confirm(f"[{name}] submit the {s_steps}-step smoke job on {instance}?", a.yes)
                sm_jobs.submit(sm, sspec); state["smoke_job"] = sspec["TrainingJobName"]; state["smoke_hps"] = hps; save_run(state)
                log(f"  submitted {sspec['TrainingJobName']}  {sm_jobs.console_url(region, sspec['TrainingJobName'])}")
                try:
                    d = wait_for_job(lambda: client(sess, "sagemaker"), sspec["TrainingJobName"], log=log)
                except JobFailed as e:
                    raise Abort(5, f"smoke job failed: {e}")
                summary = read_summary_from_checkpoints(s3, stack["bucket"], sspec["CheckpointConfig"]["S3Uri"], name)
                ok, reasons, measured = smoke_gate(summary, spec)
                state["smoke_summary"] = summary; state["smoke_passed"] = bool(ok); save_run(state)   # a failed gate is never reusable
                if not ok:
                    raise Abort(5, "smoke gate failed:\n  " + "\n  ".join(reasons))
                if measured:
                    step_rate = measured
                log(f"  smoke gate PASS: step rate {step_rate} steps/s, billable {d.get('BillableTimeInSeconds')} s")
        if a.stop_after == "smoke":
            continue
        # train
        if step_rate is None:
            msg = f"no step rate for {instance} ({instance_table.spec(instance).gpu if instance_table.spec(instance) else 'unknown GPU'}): the smoke measures it; add the GPU to {name}.yaml hardware.step_rate to plan without one"
            if not a.dry_run:
                raise Abort(4, msg)
            log(f"  NOTE: {msg} -- dry run assumes 1.0 steps/s"); step_rate = 1.0
        cap = sm_jobs.max_runtime_seconds(steps, step_rate, multiplier=cfg["compute"].get("runtime_multiplier", 3.0))
        hps = policy_hps(pcfg, name, pol, spec, split, steps=steps, smoke=False, image_digest=image.split("@")[1] if image else "", fingerprint=fingerprint)
        if a.resume_from:
            hps["resume"] = "1"
            if a.extend_steps:
                if a.extend_steps <= steps:
                    raise Abort(2, f"--extend-steps {a.extend_steps} must exceed the configured steps {steps}")
                hps["extend_steps"] = str(a.extend_steps)
                steps = a.extend_steps
                cap = sm_jobs.max_runtime_seconds(steps, step_rate, multiplier=cfg["compute"].get("runtime_multiplier", 3.0))
        tspec = job_spec(cfg, stack, name, pol, spec, hps, kind="train", instance=instance, spot=pol_spot, image=image or "x@sha256:" + "0" * 64,
                         dataset_s3=dataset_s3, max_runtime_s=cap, volume_gb=vol, run_name=run_name, hub_s3=hub_s3)
        if a.resume_from:
            tspec["CheckpointConfig"]["S3Uri"] = f"{s3_root(stack)}/{sm_jobs.TRAIN_PREFIX}/{a.resume_from}/checkpoints/"
        est_h = (steps / step_rate) / 3600
        log(f"\n[{name}] train: {steps} steps on {instance}{' (Spot)' if pol_spot else ''}, ~{est_h:.1f} h at {step_rate} steps/s, runtime cap {cap / 3600:.1f} h, volume {vol} GB")
        log(f"  hyperparameters: {hps}")
        if a.dry_run:
            log(json.dumps(tspec, indent=2)); log("  (dry run: not submitted)")
            continue
        confirm(f"[{name}] submit the training job on {instance} (~{est_h:.1f} h billed)?", a.yes)
        try:
            sm_jobs.submit(sm, tspec)
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code")
            fallbacks = [t for t in cfg["compute"].get("fallback_instance_types", []) if t != instance]
            if code != "ResourceLimitExceeded" or not fallbacks:   # only a quota refusal at submit justifies another instance type
                raise Abort(6, f"create_training_job failed on {instance}: {e}")
            log(f"  submit on {instance} refused ({code}); trying fallbacks {fallbacks}")
            last = str(e)
            for fb in fallbacks:
                fb_rate = instance_table.step_rate(spec, fb)
                if fb_rate is None:   # the runtime cap is derived from a step rate, never a flat number
                    log(f"  {fb}: skipped, {name}.yaml hardware.step_rate has no entry for its GPU"); continue
                fb_cap = sm_jobs.max_runtime_seconds(steps, fb_rate, multiplier=cfg["compute"].get("runtime_multiplier", 3.0))
                tspec["ResourceConfig"]["InstanceType"] = fb
                tspec["StoppingCondition"]["MaxRuntimeInSeconds"] = fb_cap
                if "MaxWaitTimeInSeconds" in tspec["StoppingCondition"]:
                    tspec["StoppingCondition"]["MaxWaitTimeInSeconds"] = fb_cap * 2
                fb_h = (steps / fb_rate) / 3600
                log(f"  {fb}: ~{fb_h:.1f} h at {fb_rate} steps/s, runtime cap {fb_cap / 3600:.1f} h")
                confirm(f"[{name}] submit the training job on the fallback {fb} instead (~{fb_h:.1f} h billed)?", a.yes)
                try:
                    sm_jobs.submit(sm, tspec); instance = fb; step_rate = fb_rate; cap = fb_cap; state["instance_type"] = fb; break
                except ClientError as e2:
                    last = str(e2); log(f"  {fb}: {e2.response.get('Error', {}).get('Code')}")
            else:
                raise Abort(6, f"every fallback instance type was refused at submit; last error: {last}")
        state["train_job"] = tspec["TrainingJobName"]; save_run(state)
        log(f"  submitted {tspec['TrainingJobName']}  {sm_jobs.console_url(region, tspec['TrainingJobName'])}")
        log(f"  reattach any time with: ./finetune.sh --train-job {tspec['TrainingJobName']}")
        if a.no_wait:
            continue
        tail = logs.LogTail(s3, stack["bucket"], tspec["CheckpointConfig"]["S3Uri"].split("/", 3)[3] + f"{name}/train.log")
        def on_poll(_d, _tail=tail):   # every poll: print the newest metric record since the last poll
            m = logs.latest_metric(_tail.read())
            if m:
                log(f"    {m['line']}")
        try:
            d = wait_for_job(lambda: client(sess, "sagemaker"), tspec["TrainingJobName"], log=log, on_poll=on_poll)
        except JobFailed as e:
            raise Abort(6, str(e))
        rc = max(rc, _report(a, cfg, sess, s3, stack, state, d))
    return rc


def _report(a, cfg, sess, s3, stack, state: dict, describe: dict) -> int:
    log("\nreport:")
    dest = RUNS / state["name"]
    if getattr(a, "no_download", False):
        summary = download_summary_only(s3, stack["bucket"], describe, state["policy"], dest)
    else:
        summary = download_outputs(s3, stack["bucket"], describe, dest)
    summary["_eval_path"] = str(dest / "model" / "eval" / "eval_last.json")
    ref = report.load_reference(state["policy"], state.get("dataset_name", ""))
    log(report.table(summary, ref))
    v, reasons = report.verdict(summary, ref, state.get("dataset_name", ""))
    log(f"verdict: {v}\n  " + "\n  ".join(reasons))
    state.update({"summary": summary, "verdict": v, "billable_seconds": describe.get("BillableTimeInSeconds"), "job": describe.get("TrainingJobName")})
    m = report.manifest(state, describe, summary)
    (dest / "MANIFEST.json").write_text(json.dumps(m, indent=2))
    save_run(state)
    if getattr(a, "no_download", False):
        log(f"  artefacts: {dest}/model (SUMMARY.json, eval/ only; --no-download), model.tar.gz stays at {describe['ModelArtifacts']['S3ModelArtifacts']}")
    else:
        log(f"  artefacts: {dest}/model (pretrained_model/, eval/, SUMMARY.json, LICENCE_NOTICE.md), {dest}/MANIFEST.json")
    return 0 if v in ("PASS", "n/a") else 7


if __name__ == "__main__":
    sys.exit(main())
