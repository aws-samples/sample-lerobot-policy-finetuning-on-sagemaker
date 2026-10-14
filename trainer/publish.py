#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Copy the newest checkpoint, the evaluations, the logs and the licence notice to the SageMaker output folders.

    python3 publish.py --out-dir /opt/ml/checkpoints/act --model-dir /opt/ml/model --output-data /opt/ml/output/data \
        --policy act --status completed

Used by the normal exit path AND by the SIGTERM handler, so a stopped or capped job still ships model.tar.gz with the
last saved checkpoint (status=interrupted). Never resolves `last`: the newest numbered checkpoint directory is copied.

Optional adapter merge (LoRA policies): when the copied checkpoint is a PEFT adapter (adapter_config.json, no
model.safetensors) and either the registry entry says `publish.merge_adapter: true` or the environment has MERGE_ADAPTER=1
(hyperparameter merge_adapter), the adapter is merged into the registry's base_checkpoint on the CPU and the full
weights are saved next to it as <model-dir>/pretrained_model_merged/ (model.safetensors + config.json + the processor
files and train_config.json of the adapter). The merged config is made servable: use_peft false, pretrained_path null,
use_gradient_checkpointing false, device = the training device from the adapter's config.json (cuda when missing): a
merge saved with the CPU config used for merging made a cuda server fail with tensors on two devices.
The merge never fails the publish: a failure is logged and recorded as `merge_error` in SUMMARY.json, a success as
`merged_checkpoint` (path + model_sha256). Skipped for status interrupted (the SIGTERM grace period is too short for it).
"""
from __future__ import annotations

import argparse
import json
import os
import glob
import tempfile
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from summary import sha256_file  # noqa: E402  (stdlib only; the same digest SUMMARY.json records for the adapter)

# SageMaker's checkpoint sync agent marks every mirrored file with a zero-byte sibling; they are not part of the model
IGNORE = shutil.ignore_patterns("*.sagemaker-uploaded")
MERGED_DIRNAME = "pretrained_model_merged"   # read by reload_check.py (preferred over pretrained_model/ when present) and summary.py
MERGE_JSON = "merge.json"                    # <out-dir>/merge.json: what the merge did; summary.py copies it into SUMMARY.json


def prior_summary(model_dir: str) -> dict:
    """The SUMMARY.json a previous publish wrote into model_dir (the SIGTERM path may run after the normal one), else {}."""
    p = os.path.join(model_dir, "SUMMARY.json")
    if not os.path.isfile(p):
        return {}
    try:
        return json.load(open(p))
    except (json.JSONDecodeError, OSError):
        return {}


def newest_checkpoint(out_dir: str):
    root = os.path.join(out_dir, "train", "checkpoints")
    if not os.path.isdir(root):
        return None
    # complete checkpoints only: lerobot writes pretrained_model/ (model.safetensors, then train_config.json) before
    # training_state/; a directory without train_config.json is a save in progress
    dirs = sorted(d for d in os.listdir(root) if d.isdigit()
                  and any(os.path.isfile(os.path.join(root, d, "pretrained_model", w)) for w in ("model.safetensors", "adapter_model.safetensors"))
                  and os.path.isfile(os.path.join(root, d, "pretrained_model", "train_config.json")))
    return os.path.join(root, dirs[-1], "pretrained_model") if dirs else None


def is_adapter_checkpoint(ckpt_dir: str) -> bool:
    """A PEFT (LoRA) checkpoint: adapter_config.json and no full model.safetensors (same test as eval_openloop.load_policy)."""
    return os.path.isfile(os.path.join(ckpt_dir, "adapter_config.json")) and not os.path.isfile(os.path.join(ckpt_dir, "model.safetensors"))


def registry_spec(policy: str) -> dict:
    """The registry entry for the policy (trainer/policies/<policy>.yaml), or {} when it cannot be read here."""
    try:
        from render_flags import load_policy   # PyYAML; present in the image and in the developer venv
        return load_policy(policy)
    except Exception:
        return {}


def merge_requested(policy: str, env: dict = None) -> bool:
    """MERGE_ADAPTER=1 in the environment (hyperparameter merge_adapter) or `publish.merge_adapter: true` in the registry."""
    env = os.environ if env is None else env
    if str(env.get("MERGE_ADAPTER", "")).strip() == "1":
        return True
    return bool((registry_spec(policy).get("publish") or {}).get("merge_adapter"))


def merge_adapter(adapter_dir: str, merged_dir: str, lerobot_type: str, base_checkpoint: str) -> dict:
    """Merge the adapter into its base on the CPU and save full weights to merged_dir. Needs lerobot + peft (inside the image).

    Loads through eval_openloop.load_policy, the code path the evaluation already uses for adapter checkpoints, so the merged
    weights are exactly what the evaluation scored. The base is resolved offline from the staged Hub cache (HF_HOME)."""
    if not base_checkpoint:
        raise RuntimeError(f"registry entry has no base_checkpoint: cannot merge the adapter in {adapter_dir}")
    from lerobot.policies.factory import get_policy_class
    from eval_openloop import load_policy
    merged = load_policy(get_policy_class(lerobot_type), adapter_dir, "cpu", base_checkpoint)
    adapter_cfg_p = os.path.join(adapter_dir, "config.json")
    adapter_cfg = json.load(open(adapter_cfg_p)) if os.path.isfile(adapter_cfg_p) else {}
    # servable config: the merge ran on the CPU, the checkpoint is served where it was trained; no PEFT, no checkpointing
    for key, val in {"device": adapter_cfg.get("device") or "cuda", "use_peft": False, "pretrained_path": None, "use_gradient_checkpointing": False}.items():
        if hasattr(merged.config, key):
            setattr(merged.config, key, val)
    tmp = _scratch_dir(merged_dir)   # complete directories only: a killed merge must not leave a half-written copy in model.tar.gz
    merged.save_pretrained(tmp)
    for f in sorted(os.listdir(adapter_dir)):   # processor files carry the dataset statistics used in training
        if (f.startswith("policy_") or f == "train_config.json") and os.path.isfile(os.path.join(adapter_dir, f)):
            shutil.copy2(os.path.join(adapter_dir, f), os.path.join(tmp, f))
    os.replace(tmp, merged_dir)
    sizes = {f: os.path.getsize(os.path.join(merged_dir, f)) for f in sorted(os.listdir(merged_dir)) if os.path.isfile(os.path.join(merged_dir, f))}
    for f, n in sizes.items():
        print(f"publish: merged {f}: {n / 2**30:.2f} GiB" if n >= 2**30 else f"publish: merged {f}: {n} bytes", file=sys.stderr)
    return {"path": MERGED_DIRNAME + "/", "model_sha256": sha256_file(os.path.join(merged_dir, "model.safetensors")),
            "base_checkpoint": base_checkpoint, "sizes_bytes": sizes}


def _scratch_dir(merged_dir: str) -> str:
    """Empty scratch directory for the merge on the SAME filesystem as merged_dir, so the final os.replace is atomic.

    It lives inside the model directory as a dot-prefixed sibling (.pretrained_model_merged.<random>): /opt/ml/model is its own
    mount on SageMaker (and a bind mount on an EC2 host), so a scratch directory next to the model directory sits on another
    device and os.replace fails with EXDEV (seen on the 2026-09-21 p5 smoke). reload_check.py and summary.py look only for
    the exact MERGED_DIRNAME, and maybe_merge() removes leftovers from a killed merge before the next publish."""
    parent = os.path.dirname(os.path.abspath(merged_dir))
    os.makedirs(parent, exist_ok=True)
    return tempfile.mkdtemp(prefix="." + MERGED_DIRNAME + ".", dir=parent)


def maybe_merge(policy: str, model_dir: str, out_dir: str, status: str, env: dict = None) -> dict:
    """Decide and run the adapter merge for the checkpoint just copied to <model-dir>/pretrained_model. Never raises.

    Returns {} (nothing to do), {"merged_checkpoint": {...}} or {"merge_error": "..."}; the same dict is written to
    <out-dir>/merge.json for summary.py. A stale merged directory from an earlier publish is always removed first."""
    merged_dir = os.path.join(model_dir, MERGED_DIRNAME)
    model_parent = os.path.dirname(os.path.abspath(merged_dir))
    stale_dirs = [merged_dir, merged_dir + ".tmp"] + [os.path.join(model_parent, d) for d in os.listdir(model_parent)
                                                        if d.startswith("." + MERGED_DIRNAME + ".")] if os.path.isdir(model_parent) else []
    for stale in stale_dirs:
        if os.path.isdir(stale):
            shutil.rmtree(stale)
    merge_json = os.path.join(out_dir, MERGE_JSON)
    if os.path.isfile(merge_json):
        os.remove(merge_json)
    dst = os.path.join(model_dir, "pretrained_model")
    if not is_adapter_checkpoint(dst) or not merge_requested(policy, env):
        return {}
    if status != "completed":
        print(f"publish: adapter merge skipped (status {status})", file=sys.stderr)
        return {}
    spec = registry_spec(policy)
    try:
        info = merge_adapter(dst, merged_dir, spec.get("lerobot_type") or policy, spec.get("base_checkpoint"))
        print(f"publish: adapter merged into {merged_dir} (base {info['base_checkpoint']})", file=sys.stderr)
        result = {"merged_checkpoint": info}
    except (Exception, SystemExit) as e:   # load_policy exits when the base is missing; a merge failure never fails the publish
        print(f"publish: adapter merge failed: {type(e).__name__}: {e}", file=sys.stderr)
        result = {"merge_error": f"{type(e).__name__}: {e}"}
        for d in [merged_dir + ".tmp"] + glob.glob(os.path.join(os.path.abspath(model_dir), "." + MERGED_DIRNAME + ".*")):
            shutil.rmtree(d, ignore_errors=True)   # nothing half-written ships in model.tar.gz
    os.makedirs(out_dir, exist_ok=True)
    with open(merge_json, "w") as f:
        json.dump(result, f, indent=2)
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True); ap.add_argument("--model-dir", required=True); ap.add_argument("--output-data", required=True)
    ap.add_argument("--policy", default=os.environ.get("POLICY", "act"))
    ap.add_argument("--status", choices=["completed", "interrupted", "failed"], default="completed")
    ap.add_argument("--accept-licence", default=os.environ.get("ACCEPT_LICENCE", "0"))
    ap.add_argument("--approval-ref", default=os.environ.get("APPROVAL_REF", ""))
    ap.add_argument("--dataset-fingerprint", default=os.environ.get("DATASET_FINGERPRINT", ""), help="recorded in SUMMARY and the notice")
    ap.add_argument("--image-digest", default=os.environ.get("IMAGE_DIGEST", ""), help="recorded in SUMMARY and the notice")
    a = ap.parse_args(argv)
    os.makedirs(a.model_dir, exist_ok=True); os.makedirs(a.output_data, exist_ok=True)
    ckpt = newest_checkpoint(a.out_dir)
    if ckpt is None:
        print("publish: no checkpoint to publish", file=sys.stderr)
    else:
        dst = os.path.join(a.model_dir, "pretrained_model")
        if os.path.isdir(dst):
            shutil.rmtree(dst)
        shutil.copytree(ckpt, dst, ignore=IGNORE)
        print(f"publish: {ckpt} -> {dst}", file=sys.stderr)
        maybe_merge(a.policy, a.model_dir, a.out_dir, a.status)
    ev = os.path.join(a.out_dir, "eval")
    if os.path.isdir(ev):
        shutil.copytree(ev, os.path.join(a.model_dir, "eval"), dirs_exist_ok=True, ignore=IGNORE)
    for name in ("train.log", "losscurve.txt", "split.json", "exitcode"):
        src = os.path.join(a.out_dir, name)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(a.output_data, name))
    # provenance: the hyperparameters (exported by the entrypoint) or, on a re-run, what the previous publish recorded
    prior = prior_summary(a.model_dir)
    dataset_fp = a.dataset_fingerprint or prior.get("dataset_fingerprint") or ""
    image_digest = a.image_digest or prior.get("image_digest") or ""
    # the licence gate, exactly as train_policy.sh ran it before training; its outcome is stamped into SUMMARY and the notice
    accepted = str(a.accept_licence) == "1"
    lic = subprocess.run([sys.executable, os.path.join(HERE, "licence.py"), a.policy] + (["--accept"] if accepted else [])
                         + (["--approval-ref", a.approval_ref] if a.approval_ref else []), capture_output=True, text=True)
    licence_status = "accepted" if lic.returncode == 0 else "refused"
    # licence notice next to the weights, naming this run's dataset, image, status and gate outcome
    cmd = [sys.executable, os.path.join(HERE, "licence.py"), a.policy, "--notice", "--run-status", a.status, "--licence-status", licence_status]
    if dataset_fp:
        cmd += ["--dataset", dataset_fp]
    if image_digest:
        cmd += ["--image", image_digest]
    if accepted:
        cmd += ["--accepted-by", f"policies.{a.policy}.accept_licence: true" + (f", approval_ref {a.approval_ref}" if a.approval_ref else "")]
    notice = subprocess.run(cmd, capture_output=True, text=True)
    if notice.returncode == 0:
        with open(os.path.join(a.model_dir, "LICENCE_NOTICE.md"), "w") as f:
            f.write(notice.stdout)
    else:
        print(f"publish: licence notice failed: {notice.stderr}", file=sys.stderr)
    # the summary (also copied into out-dir so the laptop can read it from the checkpoint prefix without model.tar.gz)
    cmd = [sys.executable, os.path.join(HERE, "summary.py"), "--out-dir", a.out_dir, "--model-dir", a.model_dir,
           "--policy", a.policy, "--status", a.status, "--write", "--licence-status", licence_status,
           "--dataset-fingerprint", dataset_fp, "--image-digest", image_digest]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"publish: summary.py failed: {r.stderr}", file=sys.stderr)
        return 1
    print(r.stdout)
    return 0 if ckpt else 2


if __name__ == "__main__":
    sys.exit(main())
