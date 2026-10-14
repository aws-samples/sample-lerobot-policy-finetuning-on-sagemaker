# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Report a finished run: SUMMARY vs the reference band, MANIFEST.json, and a table across runs.

    python3 -m pipeline.report --run runs/<name>.json            # after run.py finished (or reattach)
    python3 -m pipeline.report --compare runs/*.json              # leaderboard-style table
Verdicts (only when the dataset matches the reference's): PASS | REPEAT | FAIL | n/a.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
REFERENCE = REPO / "pipeline" / "reference_results.json"


def load_reference(policy: str, dataset_name: str = "") -> dict:
    """Reference block for (policy, dataset); {} when there is none. Adds dataset_name so verdict() can match."""
    per_policy = json.loads(REFERENCE.read_text()).get(policy) or {}
    block = per_policy.get(dataset_name)
    if block is None:
        return {}
    return {**{k: v for k, v in block.items() if not k.startswith("_")}, "dataset_name": dataset_name}


def first_eval(summary: dict) -> dict:
    evals = summary.get("evals") or {}
    return evals.get("eval_last.json") or (next(iter(evals.values())) if evals else {})


def verdict(summary: dict, ref: dict, dataset_name: str) -> tuple:
    """(verdict, reasons)."""
    band = ref.get("band")
    if not band or not ref.get("dataset_name") or ref["dataset_name"] != dataset_name:
        return "n/a", ["no reference band for this dataset; numbers reported without a verdict"]
    e = first_eval(summary)
    if not e or e.get("joints_mae") is None or not math.isfinite(float(e["joints_mae"])):
        return "FAIL", ["no finite joints MAE in SUMMARY.evals"]
    if e.get("stride") not in (None, 1):
        return "n/a", [f"evaluated at stride {e['stride']}; the reference band applies to stride-1 evaluations only"]
    j, jh = float(e["joints_mae"]), float(e.get("joints_hold") or 0)
    g, gh = e.get("grippers_mae"), e.get("grippers_hold")
    beats = int(str(e.get("beats_hold") or "0/0").split("/")[0])
    reasons, other_ok = [], True
    if jh and j / jh > band["joints_ratio_max"]:
        other_ok = False; reasons.append(f"joints ratio {j / jh:.3f} > {band['joints_ratio_max']}")
    if g is not None and float(g) > band["grippers_mae_max"]:
        other_ok = False; reasons.append(f"grippers MAE {float(g):.5f} > {band['grippers_mae_max']}")
    if g is not None and gh and float(g) / float(gh) > band["grippers_ratio_max"]:
        other_ok = False; reasons.append(f"grippers ratio {float(g) / float(gh):.3f} > {band['grippers_ratio_max']}")
    if e.get("gripper_timing") != band["gripper_timing"]:
        other_ok = False; reasons.append(f"gripper timing {e.get('gripper_timing')} != {band['gripper_timing']}")
    near_miss = False
    if beats < band["beats_hold_min"]:
        # a dim sitting within 5 % of the hold baseline flips between runs (the two reference runs differ by 0.0005 in
        # aggregate MAE); one such flip is a REPEAT, not a FAIL
        per_dim = (summary.get("per_dim") or {}).get("all") or _per_dim_from_eval(summary)
        worst_flips = sorted(v["model"] / v["hold"] for v in per_dim.values() if v.get("hold") and v.get("model") is not None and v["model"] >= v["hold"])
        near_hold = [x for x in worst_flips if x <= 1.05]
        if beats == band["beats_hold_min"] - 1 and near_hold:
            near_miss = True; reasons.append(f"{beats} dims beat hold (< {band['beats_hold_min']}): one dim within 5 % of hold ({near_hold[0]:.2f}x) -- repeat once")
        else:
            other_ok = False; reasons.append(f"only {beats} dims beat hold (< {band['beats_hold_min']})")
    if j < band.get("suspiciously_good_joints_mae", 0):
        reasons.append(f"joints MAE {j:.5f} is far below every reference run: check the split (leakage?)")
    if j <= band["joints_mae_pass"] and other_ok and not near_miss:
        return "PASS", reasons or [f"joints MAE {j:.5f} <= {band['joints_mae_pass']}, all other bounds met"]
    if j <= band["joints_mae_pass"] and other_ok and near_miss:
        return "REPEAT", reasons
    if j <= band["joints_mae_repeat"] and other_ok:
        return "REPEAT", [f"joints MAE {j:.5f} in ({band['joints_mae_pass']}, {band['joints_mae_repeat']}]: repeat once before concluding"] + reasons
    return "FAIL", reasons or [f"joints MAE {j:.5f} > {band['joints_mae_repeat']}"]


def _per_dim_from_eval(summary: dict) -> dict:
    """per-dim MAE when the run folder is available next to the summary (runs/<name>/model/eval/eval_last.json)."""
    p = summary.get("_eval_path")
    if p and Path(p).is_file():
        try:
            return (json.loads(Path(p).read_text()).get("per_dim") or {}).get("all") or {}
        except (OSError, json.JSONDecodeError):
            return {}
    return {}


def table(summary: dict, ref: dict) -> str:
    e = first_eval(summary)
    rows = [("this run", e.get("joints_mae"), (e.get("joints_mae") or 0) / e["joints_hold"] if e.get("joints_hold") else None,
             e.get("grippers_mae"), (e.get("grippers_mae") or 0) / e["grippers_hold"] if e.get("grippers_hold") else None,
             e.get("gripper_timing"), e.get("beats_hold"))]
    for r in ref.get("reference_runs", []):
        rows.append((r["label"], r["joints_mae"], r["joints_ratio"], r["grippers_mae"], r["grippers_ratio"], r["gripper_timing"], r["beats_hold"]))
    def f(x, n=5):
        return "-" if x is None else (f"{x:.{n}f}" if isinstance(x, float) else str(x))
    out = [f"{'run':58s} {'joints':>8s} {'x hold':>7s} {'grippers':>9s} {'x hold':>7s} {'timing':>7s} {'beats':>6s}"]
    for lab, j, jr, g, gr, t, b in rows:
        out.append(f"{lab[:58]:58s} {f(j):>8s} {f(jr, 3):>7s} {f(g):>9s} {f(gr, 3):>7s} {str(t or '-'):>7s} {str(b or '-'):>6s}")
    return "\n".join(out)


def git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def manifest(run: dict, describe: dict, summary: dict) -> dict:
    return {
        "schema": 1, "generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "run": run.get("name"), "policy": run.get("policy"), "job": describe.get("TrainingJobName"),
        "status": describe.get("TrainingJobStatus"), "instance_type": describe.get("ResourceConfig", {}).get("InstanceType"),
        "billable_seconds": describe.get("BillableTimeInSeconds"), "spot": describe.get("EnableManagedSpotTraining", False),
        "network_isolation": describe.get("EnableNetworkIsolation"), "image": describe.get("AlgorithmSpecification", {}).get("TrainingImage"),
        "hyperparameters": describe.get("HyperParameters"), "dataset_s3": run.get("dataset_s3"), "dataset_fingerprint": run.get("dataset_fingerprint"),
        "dataset_objects": run.get("dataset_objects"), "dataset_bytes": run.get("dataset_bytes"), "split": summary.get("split"),
        "checkpoint_step": summary.get("checkpoint_step"), "model_sha256": summary.get("model_sha256"), "versions": summary.get("versions"),
        "licence_status": summary.get("licence_status"), "deliverable": summary.get("deliverable"), "repo_git_sha": git_sha(),
        "model_artifact": describe.get("ModelArtifacts", {}).get("S3ModelArtifacts"),
    }


def compare(paths) -> str:
    rows = []
    for p in paths:
        run = json.loads(Path(p).read_text())
        s = run.get("summary") or {}
        e = first_eval(s)
        rows.append((run.get("name"), run.get("policy"), run.get("instance_type"), run.get("billable_seconds"), e.get("joints_mae"),
                     e.get("beats_hold"), run.get("verdict"), s.get("licence_status"), s.get("deliverable")))
    out = [f"{'run':28s} {'policy':10s} {'instance':16s} {'billable_s':>10s} {'joints':>8s} {'beats':>6s} {'verdict':>7s} {'licence':>9s} {'deliv':>5s}"]
    for r in rows:
        out.append(" ".join([f"{str(r[0])[:28]:28s}", f"{str(r[1]):10s}", f"{str(r[2]):16s}", f"{str(r[3]):>10s}",
                             f"{r[4]:.5f}" if isinstance(r[4], float) else f"{'-':>8s}", f"{str(r[5] or '-'):>6s}", f"{str(r[6] or '-'):>7s}",
                             f"{str(r[7] or '-'):>9s}", f"{str(r[8]):>5s}"]))
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", help="runs/<name>.json written by run.py")
    ap.add_argument("--compare", nargs="*", help="runs/*.json files")
    a = ap.parse_args(argv)
    if a.compare:
        print(compare(a.compare)); return 0
    run = json.loads(Path(a.run).read_text())
    s = run.get("summary") or {}
    eval_p = Path(a.run).with_suffix("") / "model" / "eval" / "eval_last.json"   # runs/<name>/model/eval/eval_last.json
    if eval_p.is_file():
        s["_eval_path"] = str(eval_p)
    ref = load_reference(run["policy"], run.get("dataset_name", ""))
    print(table(s, ref))
    v, reasons = verdict(s, ref, run.get("dataset_name", ""))
    print(f"\nverdict: {v}\n  " + "\n  ".join(reasons))
    return 0 if v in ("PASS", "n/a") else 7


if __name__ == "__main__":
    sys.exit(main())
