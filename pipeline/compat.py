# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Which registry policies can fine-tune on a given LeRobot v3.0 dataset, and why not.

Facts come from the dataset's meta/ (info.json, stats.json, tasks.parquet); requirements come from the registry YAMLs:
  requires_flat_action_names   action/state names must be a flat list (nested dicts break the policy's name scan)
  requires_language            the policy conditions on the task string (0 tasks = blocked, 1 task = language is constant)
  requires_reward              RL objective: the dataset must carry next.reward
  requires_quantile_stats      QUANTILES normalisation: q01/q99 are completed in-job when missing (note, not a blocker)
  single_camera_required       trains on one camera (select_camera.py picks it; note when the dataset has several)
  square_images                the kept camera must be square (TD-MPC's random-shift augmentation)
  max_action_dim               the policy pads actions to this width; wider datasets cannot be fine-tuned
`./finetune.sh --compat` prints the table without touching AWS; the dataset stage prints it on every run and refuses an
enabled policy that is blocked (exit 3) instead of letting the job fail after the base weights are loaded.
"""
import json
from pathlib import Path

RUNNABLE = ("validated", "trained", "smoke-tested")


def dataset_facts(root: Path) -> dict:
    root = Path(root)
    info = json.loads((root / "meta" / "info.json").read_text())
    feats = info.get("features", {})
    cams = {}
    for k, v in feats.items():
        if v.get("dtype") in ("video", "image"):
            hw = [d for d in v.get("shape", []) if d != 3]
            cams[k] = (hw[0], hw[1]) if len(hw) >= 2 else None
    names = (feats.get("action") or {}).get("names")
    facts = {
        "name": root.name, "fps": info.get("fps"),
        "action_dim": (feats.get("action") or {}).get("shape", [None])[0],
        "state_dim": (feats.get("observation.state") or {}).get("shape", [None])[0],
        "names_flat": None if names is None else isinstance(names, list),
        "cameras": cams, "has_reward": "next.reward" in feats,
        "n_tasks": None, "has_quantiles": None,
    }
    tasks = root / "meta" / "tasks.parquet"
    if tasks.is_file():
        try:
            import pyarrow.parquet as pq
            facts["n_tasks"] = pq.read_metadata(str(tasks)).num_rows
        except Exception:  # pyarrow missing or unreadable file: the rule is skipped with a note
            facts["n_tasks"] = None
    stats = root / "meta" / "stats.json"
    if stats.is_file():
        try:
            st = json.loads(stats.read_text())
            facts["has_quantiles"] = all(q in (st.get(k) or {}) for k in ("action", "observation.state") for q in ("q01", "q99"))
        except json.JSONDecodeError:
            facts["has_quantiles"] = None
    return facts


def facts_from_info(info: dict) -> dict:
    """The subset of facts available without the meta/ folder (source=s3): tasks and quantiles stay unknown."""
    tmp = {"features": info.get("features", {}), "fps": info.get("fps")}
    feats = tmp["features"]
    cams = {k: (lambda hw: (hw[0], hw[1]) if len(hw) >= 2 else None)([d for d in v.get("shape", []) if d != 3])
            for k, v in feats.items() if v.get("dtype") in ("video", "image")}
    names = (feats.get("action") or {}).get("names")
    return {"name": "s3", "fps": info.get("fps"), "action_dim": (feats.get("action") or {}).get("shape", [None])[0],
            "state_dim": (feats.get("observation.state") or {}).get("shape", [None])[0],
            "names_flat": None if names is None else isinstance(names, list), "cameras": cams,
            "has_reward": "next.reward" in feats, "n_tasks": None, "has_quantiles": None}


def evaluate(facts: dict, spec: dict, dataset_cameras_override: list = None) -> dict:
    """{'ok': bool, 'blockers': [...], 'notes': [...]} for one registry entry against one dataset (dataset_cameras_override =
    policies.<name>.cameras, the subset the job will train on)."""
    blockers, notes = [], []
    cams = facts.get("cameras") or {}
    if spec.get("requires_reward") and not facts.get("has_reward"):
        blockers.append("needs reward-labelled data (feature next.reward); imitation datasets have none")
    if spec.get("requires_flat_action_names") and facts.get("names_flat") is False:
        blockers.append("action names are a nested dict; this policy scans a flat names list")
    if spec.get("requires_language"):
        n = facts.get("n_tasks")
        if n == 0:
            blockers.append("no task strings in meta/tasks.parquet; the policy conditions on language")
        elif n == 1:
            notes.append("one task string: the language branch is a constant")
        elif n is None:
            notes.append("task strings not checked (meta/ not available)")
    if not cams:
        blockers.append("no camera feature (dtype video/image)")
    elif spec.get("single_camera_required"):
        cam = sorted(cams)[0]
        if len(cams) > 1:
            notes.append(f"trains on one camera ({cam} by default; policies.<name>.camera picks another)")
        hw = cams.get(cam)
        if spec.get("square_images") and hw and hw[0] != hw[1]:
            blockers.append(f"needs a square camera image; {cam} is {hw[0]}x{hw[1]} and downscaling keeps the aspect")
    if spec.get("requires_quantile_stats") and facts.get("has_quantiles") is False:
        notes.append("stats.json lacks q01/q99: completed in-job (ensure_quantiles.py)")
    div = spec.get("camera_count_divides")
    if div and cams:
        chosen = [k for k in (dataset_cameras_override or []) if k in cams] if dataset_cameras_override else list(cams)
        if len(chosen) and div % len(chosen) != 0:
            ok_counts = [n for n in range(1, len(cams) + 1) if div % n == 0]
            blockers.append(f"splits a {div}-px width across cameras: {len(chosen)} camera(s) do not divide it; use {ok_counts} of them via policies.<name>.cameras")
    mx = spec.get("max_action_dim")
    if mx and facts.get("action_dim") and facts["action_dim"] > mx:
        blockers.append(f"action has {facts['action_dim']} dims, the policy pads to at most {mx}")
    return {"ok": not blockers, "blockers": blockers, "notes": notes}


def report(facts: dict, specs: dict, enabled: list = ()) -> list:
    """Human-readable lines: how many registry policies fit this dataset, grouped, with reasons for the rest."""
    out_of_scope = sorted(n for n, s in specs.items() if s.get("out_of_scope"))
    specs = {n: s for n, s in specs.items() if not s.get("out_of_scope")}
    rows = {n: evaluate(facts, s) for n, s in specs.items()}
    runnable = [n for n, r in rows.items() if r["ok"] and specs[n].get("status") in RUNNABLE]
    stub_ok = [n for n, r in rows.items() if r["ok"] and specs[n].get("status") not in RUNNABLE]
    blocked = [n for n, r in rows.items() if not r["ok"]]
    cams = facts.get("cameras") or {}
    head = (f"compatibility of dataset {facts.get('name')} ({facts.get('action_dim')}-dim actions, {len(cams)} camera(s), "
            f"{facts.get('n_tasks') if facts.get('n_tasks') is not None else '?'} task string(s)): "
            f"{len(runnable) + len(stub_ok)} of {len(specs)} registry policies can fine-tune on it")
    lines = [head, f"  runnable now ({len(runnable)}): {', '.join(sorted(runnable)) or '-'}"]
    if stub_ok:
        lines.append(f"  compatible but stub in this release ({len(stub_ok)}, need --allow-stubs): {', '.join(sorted(stub_ok))}")
    for n in sorted(blocked):
        lines.append(f"  not compatible: {n}: " + "; ".join(rows[n]["blockers"]))
    if out_of_scope:
        lines.append(f"  out of scope ({len(out_of_scope)}, not offered by this repository): {', '.join(out_of_scope)}")
    for n in sorted(set(enabled) & set(rows)):
        for note in rows[n]["notes"]:
            lines.append(f"  note for {n}: {note}")
    return lines


def blocked_enabled(facts: dict, specs: dict, enabled: list, pols: dict = None) -> dict:
    cams = lambda n: ((pols or {}).get(n) or {}).get("cameras")
    return {n: (evaluate(facts, specs[n], cams(n))["blockers"] or ["out of scope: not offered by this repository"]) for n in enabled
            if n in specs and (specs[n].get("out_of_scope") or not evaluate(facts, specs[n], cams(n))["ok"])}
