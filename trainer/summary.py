#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Build SUMMARY.json for a finished (or interrupted) run: the one record every later stage reads.

    python3 summary.py --out-dir /opt/ml/checkpoints/act --model-dir /opt/ml/model --policy act --status completed \
        [--dataset-fingerprint SHA] [--image-digest sha256:...] [--licence-status clean --deliverable 1]

Keeps the keys the validated pipeline produced (checkpoint, split, evals{eval_<ckpt>: joints_mae, joints_hold,
grippers_mae, grippers_hold, gripper_timing, episodes, stride}) and adds: policy, status, checkpoint_step, model sha256,
beats_hold (per-dim MAE below the hold baseline), step_rate from the training log, versions, dataset fingerprint,
image digest, licence/deliverable, chunk, and merged_checkpoint / merge_error when publish.py merged a LoRA adapter. Python 3.9+, stdlib only (the registry, read for the deliverable flag when
--deliverable is not given, needs PyYAML; without it the flag falls back to the previous SUMMARY.json).
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import hashlib
import json
import os
import re
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
STEP_RE = re.compile(r"step:([0-9.]+)(K?) .*?updt_s:([0-9.]+) data_s:([0-9.]+) smp/s:([0-9.]+)")
TIME_RE = re.compile(r"INFO (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)")  # not anchored: a tqdm bar may precede INFO on the same line
# lerobot's tqdm bar shares the line with the INFO record when both streams land in the same log:
#   'Training:  53%|█████▎    | 10600/20000 [40:57<31:02,  5.05step/s]INFO 2026-... step:11K ...'
# its counter is the exact step, where the step:NK label is rounded to whole thousands from step 1000 on.
TQDM_RE = re.compile(r"\|\s*(\d+)/\d+ \[[^\]]*\]\s*INFO ")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_step(tok: str, k: str) -> int:
    return int(round(float(tok) * (1000 if k else 1)))


def exact_step(line: str, label: int) -> int:
    """The tqdm counter's step when the line carries one and it rounds to the printed label, else the label itself."""
    m = TQDM_RE.search(line)
    if not m:
        return label
    n = int(m.group(1))
    if n == label or (label >= 1000 and int(round(n / 1000.0)) * 1000 == label):
        return n
    return label


def step_rate_from_log(train_log: str, batch_size: int = None):
    """Median steps/s over the log's metric windows, from the INFO timestamps; None when there is nothing to measure.

    Two things about lerobot's records: the first window includes the dataloader warm-up (skipped), and from step 1000 on
    the step label is rounded to whole thousands (step:8K), so at log_freq 100 the same label repeats up to ten times.
    Windows are therefore taken between consecutive records with DISTINCT steps, using the exact step from the tqdm
    counter that shares the line when it is present (every SageMaker log so far) and the rounded label otherwise; a
    window whose step or time delta is not positive is dropped. Wall-clock windows include the checkpoint writes and the
    tracker's own overhead, and the median discards the windows that contain a save or a stall. When no record carries
    a timestamp the per-step timers stand in: 1 / (updt_s + data_s), which misses that overhead (typically 1-3 %).
    """
    if not os.path.isfile(train_log):
        return None, None
    records = []
    with open(train_log, errors="replace") as f:
        for line in f:
            m = STEP_RE.search(line)
            if not m:
                continue
            t = TIME_RE.search(line)
            step = exact_step(line, parse_step(m.group(1), m.group(2)))
            records.append((step, dt.datetime.strptime(t.group(1), "%Y-%m-%d %H:%M:%S") if t else None,
                            float(m.group(3)), float(m.group(4)), float(m.group(5))))
    last = records[-1] if records else None
    last_window = ({"updt_s": last[2], "data_s": last[3], "smp_per_s": last[4],
                    "timer_steps_per_s": round(1.0 / (last[2] + last[3]), 3) if last[2] + last[3] > 0 else None}
                   if last else None)
    if len(records) < 3:
        return None, ({"updt_s": last[2], "data_s": last[3]} if last else None)
    body = records[1:]                       # the first window is the warm-up
    distinct = [body[0]]
    for r in body[1:]:
        if r[0] != distinct[-1][0]:          # collapse records that repeat the rounded label
            distinct.append(r)
    rates = []
    for (s0, t0, *_), (s1, t1, *_) in zip(distinct, distinct[1:]):
        if t0 and t1 and s1 > s0 and (t1 - t0).total_seconds() > 0:
            rates.append((s1 - s0) / (t1 - t0).total_seconds())
    if not rates:                            # no usable timestamps: per-step timers
        rates = [1.0 / (u + d) for (_, _, u, d, _) in body if u + d > 0]
    return (statistics.median(rates) if rates else None), last_window


def resolve_checkpoint(out_dir: str):
    """(numbered dir, step) of the newest checkpoint; `last` is a symlink that may not survive an S3 round trip."""
    dirs = sorted(d for d in glob.glob(os.path.join(out_dir, "train", "checkpoints", "[0-9]*"))
                  if os.path.isfile(os.path.join(d, "pretrained_model", "train_config.json")))   # complete checkpoints only
    if not dirs:
        return None, None
    return dirs[-1], int(os.path.basename(dirs[-1]))


def eval_summary(path: str) -> dict:
    d = json.load(open(path))
    a, g = d["mae"]["all"], d["gripper_timing"]
    per_dim = d.get("per_dim", {}).get("all", {})
    beats = sum(1 for v in per_dim.values() if v.get("model") is not None and v.get("hold") is not None and v["model"] < v["hold"])
    return {"layout": d.get("layout"), "joints_mae": a["all_joints"]["model"], "joints_hold": a["all_joints"]["hold"],
            "grippers_mae": a["all_grippers"]["model"], "grippers_hold": a["all_grippers"]["hold"],
            "gripper_timing": f"{g['hits']}/{g['total']}" if g.get("total") else None,
            "beats_hold": f"{beats}/{len(per_dim)}" if per_dim else None, "episodes": d.get("episodes"), "stride": d.get("stride"),
            "chunk": d.get("chunk"), "n_samples": d.get("n_samples")}


def build(args) -> dict:
    out = {"schema": 1, "policy": args.policy, "status": args.status, "checkpoint": "pretrained_model/", "split": None, "evals": {},
           "generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    split_p = os.path.join(args.out_dir, "split.json")
    if os.path.isfile(split_p):
        out["split"] = json.load(open(split_p))
    ckpt_dir, step = resolve_checkpoint(args.out_dir)
    out["checkpoint_step"] = step
    model_file = next((os.path.join(args.model_dir, "pretrained_model", w) for w in ("model.safetensors", "adapter_model.safetensors")
                       if os.path.isfile(os.path.join(args.model_dir, "pretrained_model", w))), os.path.join(args.model_dir, "pretrained_model", "model.safetensors"))
    if os.path.isfile(model_file):
        out["model_sha256"] = sha256_file(model_file)
    for p in sorted(glob.glob(os.path.join(args.model_dir, "eval", "eval_*.json"))):
        try:
            out["evals"][os.path.basename(p)] = eval_summary(p)
        except (KeyError, json.JSONDecodeError) as e:
            out["evals"][os.path.basename(p)] = {"error": f"{type(e).__name__}: {e}"}
    rate, last = step_rate_from_log(os.path.join(args.out_dir, "train.log"))
    out["step_rate"] = rate
    out["last_window"] = last
    # a later re-run (e.g. after reload_check) must not erase what publish.py recorded: keep prior values when not given
    prior = {}
    prev_p = os.path.join(args.model_dir, "SUMMARY.json")
    if os.path.isfile(prev_p):
        try:
            prior = json.load(open(prev_p))
        except json.JSONDecodeError:
            prior = {}
    out["dataset_fingerprint"] = args.dataset_fingerprint or prior.get("dataset_fingerprint")
    out["image_digest"] = args.image_digest or prior.get("image_digest")
    out["licence_status"] = args.licence_status or prior.get("licence_status")
    if args.deliverable is not None:
        out["deliverable"] = bool(int(args.deliverable))
    else:                                    # the registry's flag, so the field is never null for a registry policy
        reg = registry_deliverable(args.policy)
        out["deliverable"] = reg if reg is not None else prior.get("deliverable")
    out["versions"] = versions()
    rc = os.path.join(args.out_dir, "reload_check.json")
    if os.path.isfile(rc):
        out["reload_check"] = json.load(open(rc))
    sc = os.path.join(args.out_dir, "smoke_checks.json")
    if os.path.isfile(sc):
        out["smoke"] = json.load(open(sc))
    mj = os.path.join(args.out_dir, "merge.json")   # publish.py: adapter merged into full weights (merged_checkpoint) or why not (merge_error)
    if os.path.isfile(mj):
        try:
            out.update({k: v for k, v in json.load(open(mj)).items() if k in ("merged_checkpoint", "merge_error")})
        except json.JSONDecodeError:
            out["merge_error"] = "merge.json unreadable"
    exit_p = os.path.join(args.out_dir, "exitcode")
    out["train_exitcode"] = int(open(exit_p).read().strip()) if os.path.isfile(exit_p) else None
    return out


def registry_deliverable(policy: str):
    """The registry's `deliverable` flag for the policy (bool), or None when the registry cannot be read here."""
    try:
        if HERE not in sys.path:
            sys.path.insert(0, HERE)
        from render_flags import load_policy   # PyYAML; present in the image and in the developer venv
        spec = load_policy(policy)
    except Exception:
        return None
    v = spec.get("deliverable")
    return bool(v) if v is not None else None


def versions() -> dict:
    v = {"python": sys.version.split()[0]}
    for mod in ("lerobot", "torch", "torchvision", "av", "numpy", "transformers", "diffusers"):
        try:
            v[mod] = __import__(mod).__version__
        except Exception:
            v[mod] = None
    return v


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True, help="the trainer's OUT_DIR (train.log, split.json, train/checkpoints)")
    ap.add_argument("--model-dir", required=True, help="/opt/ml/model: pretrained_model/ and eval/ already copied there")
    ap.add_argument("--policy", default=os.environ.get("POLICY", "act"))
    ap.add_argument("--status", choices=["completed", "interrupted", "failed"], default="completed")
    ap.add_argument("--dataset-fingerprint", default=os.environ.get("DATASET_FINGERPRINT", ""))
    ap.add_argument("--image-digest", default=os.environ.get("IMAGE_DIGEST", ""))
    ap.add_argument("--licence-status", default="")
    ap.add_argument("--deliverable", default=None)
    ap.add_argument("--write", action="store_true", help="write <model-dir>/SUMMARY.json and a copy into <out-dir>")
    a = ap.parse_args(argv)
    out = build(a)
    text = json.dumps(out, indent=2)
    if a.write:
        for d in (a.model_dir, a.out_dir):
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, "SUMMARY.json"), "w") as f:
                f.write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
