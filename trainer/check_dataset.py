#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Validate a LeRobot v3.0 dataset and choose the train / held-out split. One tool for three places:
the robot PC on the day you record, the laptop before anything is uploaded, and the training job itself.

    python3 check_dataset.py datasets/my_task                       # report, exit 0 (ok/warnings) or 1 (a FAIL)
    python3 check_dataset.py datasets/my_task --heldout "0 13 26"   # explicit held-out ids
    python3 check_dataset.py <root> --emit-split > split.json       # what the trainer calls (report goes to stderr)

Dependencies: Python >= 3.9 with pyarrow and numpy only (no torch, no lerobot), so it runs on a recorder PC.

What it catches (each cost real time once):
  FAIL  not codebase_version v3.0 / missing meta files / camera features that are PNG frames instead of video
  FAIL  action == observation.state bit-exactly in ANY data file (a recording of nothing: the policy would learn "hold")
  FAIL  action.names length != action dim; fewer than 10 episodes without an explicit split; bad episode ids
  WARN  action is an exact k-frame copy of state (a delayed copy is still not a command signal)
  WARN  action.names is a dict {"axes": [...]} instead of a flat list (some policies silently break on it)
  WARN  cameras are not 16:9 (informational: the in-job downscale keeps the aspect ratio; widths other than 640 are fine)
  WARN  stats.json lacks q01/q99, or the open gripper sits far above q99 (QUANTILES normalisation saturates it)
  WARN  chunk horizon outside 0.5-2.0 s at this fps
Override a FAIL only when you know why: --force-res --force-action-eq-state --force-small (three separate flags).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

REPORT = []          # (level, message)
FAILS = 0


def say(level: str, msg: str) -> None:
    global FAILS
    REPORT.append((level, msg))
    if level == "FAIL":
        FAILS += 1
    print(f"{level:4s} {msg}", file=sys.stderr)


def load_json(path: str):
    with open(path) as f:
        return json.load(f)


def action_names(feature: dict):
    """Flat list of names, plus whether the source was the dict form {"axes": [...]}."""
    names = feature.get("names")
    if isinstance(names, dict):
        for v in names.values():
            if isinstance(v, list):
                return list(v), True
        return None, True
    if isinstance(names, list):
        return list(names), False
    return None, False


def check_meta(root: str, require_flat: bool):
    info_p = os.path.join(root, "meta", "info.json")
    if not os.path.isfile(info_p):
        say("FAIL", f"{info_p} missing: not a LeRobot dataset root (expect meta/, data/, videos/)")
        return None
    info = load_json(info_p)
    ver = str(info.get("codebase_version", ""))
    if ver != "v3.0":
        say("FAIL", f"codebase_version is {ver!r}; this pipeline needs v3.0 (convert with the LeRobot v3.0 tools first)")
    n, fps = info.get("total_episodes"), info.get("fps")
    if not isinstance(n, int) or n < 1 or not fps or fps <= 0:
        say("FAIL", f"total_episodes={n} fps={fps} in info.json")
        return None
    feats = info.get("features", {})
    if "action" not in feats or "observation.state" not in feats:
        say("FAIL", "features must contain 'action' and 'observation.state'")
        return None
    a_shape, s_shape = feats["action"].get("shape"), feats["observation.state"].get("shape")
    if feats["action"].get("dtype") != "float32" or a_shape != s_shape:
        say("FAIL", f"action {feats['action'].get('dtype')}{a_shape} vs observation.state {s_shape}: expected float32 and equal shapes")
    dim = int(a_shape[0]) if a_shape else 0
    names, was_dict = action_names(feats["action"])
    if was_dict:
        say("FAIL" if require_flat else "WARN",
            "action.names is a dict {'axes': [...]}, not a flat list; lerobot's list(names) yields ['axes'] and silently breaks "
            "policies that use joint names (relative actions, gripper exclusion). Flat list required for this policy." if require_flat else
            "action.names is a dict {'axes': [...]}, not a flat list; fine for ACT, breaks name-based policies (pi0.5, GR00T)")
    if names is None:
        say("WARN", "action.names missing: joints are scored generically, grippers cannot be identified")
    elif len(names) != dim:
        say("FAIL", f"action.names has {len(names)} entries but action has {dim} dims")
    cams = {k: v for k, v in feats.items() if v.get("dtype") in ("video", "image")}
    if not cams:
        say("FAIL", "no camera feature (dtype video) in features")
    for k, v in cams.items():
        if v.get("dtype") == "image":
            say("FAIL", f"{k} is dtype 'image' (PNG frames); only video datasets are supported")
        shape = v.get("shape") or []
        hw = [x for x in shape if x != 3]
        if len(hw) != 2:
            say("FAIL", f"{k}: cannot read height/width from shape {shape}")
            continue
        h, w = hw
        ratio = w / h
        tag = "16:9" if abs(ratio - 16 / 9) < 0.02 else ("4:3" if abs(ratio - 4 / 3) < 0.02 else f"{ratio:.3f}")
        say("OK", f"camera {k}: {h}x{w} ({tag})")
        if tag != "16:9":
            say("WARN", f"{k} is {tag}; the in-job downscale keeps the aspect ratio (a 4:3 camera becomes 360x480) and widths other than 640 are fine; reference bands in pipeline/reference_results.json are per dataset")
    for rel in ("meta/tasks.parquet", "meta/stats.json"):
        if not os.path.isfile(os.path.join(root, rel)):
            say("FAIL", f"{rel} missing")
    ep_files = sorted(glob.glob(os.path.join(root, "meta", "episodes", "**", "*.parquet"), recursive=True))
    if not ep_files:
        say("FAIL", "meta/episodes/chunk-*/file-*.parquet missing (v3.0 episode metadata)")
    data_files = sorted(glob.glob(os.path.join(root, "data", "**", "*.parquet"), recursive=True))
    if not data_files:
        say("FAIL", "no data/**/*.parquet files")
    for k in cams:
        vids = glob.glob(os.path.join(root, "videos", k, "**", "*.mp4"), recursive=True)
        if not vids:
            say("FAIL", f"no mp4 under videos/{k}/")
    return {"info": info, "n": n, "fps": fps, "dim": dim, "names": names, "cams": cams, "data_files": data_files, "ep_files": ep_files}


def check_data(root: str, m: dict, force_eq: bool):
    try:
        import numpy as np
        import pyarrow.parquet as pq
    except ImportError:
        say("FAIL", "pyarrow and numpy are required (pip install pyarrow numpy); the action==state check cannot run without them")
        return
    if m["ep_files"]:
        rows = sum(pq.ParquetFile(f).metadata.num_rows for f in m["ep_files"])
        (say("OK", f"episodes metadata rows = {rows} = total_episodes") if rows == m["n"]
         else say("FAIL", f"episodes metadata has {rows} rows but info.json says total_episodes={m['n']}"))
    if not m["data_files"] or m["dim"] == 0:
        return
    max_diff, n_rows, lag_hits = 0.0, 0, {}
    for f in m["data_files"]:
        cols = ["action", "observation.state", "episode_index"]
        t = pq.read_table(f, columns=cols).to_pydict()
        a = np.asarray(t["action"], dtype=np.float32)
        s = np.asarray(t["observation.state"], dtype=np.float32)
        ep = np.asarray(t["episode_index"])
        if a.shape != s.shape:
            say("FAIL", f"{os.path.basename(f)}: action {a.shape} vs state {s.shape}")
            return
        n_rows += len(a)
        file_diff = float(np.max(np.abs(a - s))) if len(a) else 0.0
        max_diff = max(max_diff, file_diff)
        if len(a) and file_diff == 0.0:
            say("WARN" if force_eq else "FAIL", f"{os.path.basename(f)}: action == observation.state bit-exactly in this whole file")
        # exact k-frame copy? compare action[t] with state[t+k] inside each episode
        for k in range(1, 11):
            same = (ep[:-k] == ep[k:])
            if same.sum() == 0:
                continue
            eq = np.all(a[:-k][same] == s[k:][same])
            lag_hits[k] = lag_hits.get(k, True) and bool(eq)
    say("OK", f"scanned {len(m['data_files'])} data file(s), {n_rows} rows; max |action - state| = {max_diff:.5f}")
    if max_diff == 0.0:
        say("WARN" if force_eq else "FAIL",
            "action == observation.state bit-exactly in every row: nothing to learn beyond 'hold' (a common recorder/export defect)"
            + (" -- proceeding because --force-action-eq-state" if force_eq else "; re-export the dataset or pass --force-action-eq-state"))
    for k, hit in lag_hits.items():
        if hit:
            say("WARN", f"action is an EXACT copy of state shifted by {k} frame(s): a delayed copy is not a command signal")
            break
    if m["dim"] == 16:
        _arm_report(root, m, np, pq)


def _arm_report(root, m, np, pq):
    """Which arm moves per episode, ASSUMING the 7+1+7+1 bimanual layout for 16-dim actions (dims 0-6 left joints, 7 left
    gripper, 8-14 right joints, 15 right gripper). Selected by the dim count alone, not by action.names."""
    arms = {}
    for f in m["data_files"]:
        t = pq.read_table(f, columns=["observation.state", "episode_index"]).to_pydict()
        s, ep = np.asarray(t["observation.state"], dtype=np.float32), np.asarray(t["episode_index"])
        for e in np.unique(ep):
            se = s[ep == e]
            arms[int(e)] = "L" if se[:, 0:7].std(axis=0).mean() > se[:, 8:15].std(axis=0).mean() else "R"
    m["arms"] = arms
    say("OK", f"bimanual layout: {sum(v == 'L' for v in arms.values())} left-arm and {sum(v == 'R' for v in arms.values())} right-arm episodes"
        " (assumed 7+1+7+1 layout for 16-dim actions: dims 0-7 left arm, 8-15 right arm, grippers 7 and 15)")


def check_stats(root: str, m: dict):
    p = os.path.join(root, "meta", "stats.json")
    if not os.path.isfile(p):
        return
    stats = load_json(p)
    cams_missing = [k for k in (m.get("cams") or {}) if k not in stats]
    if cams_missing:
        say("WARN", f"stats.json has no entry for camera(s) {cams_missing}: lerobot-train indexes stats[<camera>] (KeyError in make_dataset); the job adds ImageNet-shaped placeholders")
    for key in ("action", "observation.state"):
        st = stats.get(key)
        if not st or "mean" not in st or "std" not in st:
            say("FAIL", f"stats.json[{key}] lacks mean/std (lerobot normalises with them)")
            continue
        if len(st["mean"]) != m["dim"]:
            say("FAIL", f"stats.json[{key}].mean has {len(st['mean'])} entries, action dim is {m['dim']}")
        if "q01" not in st or "q99" not in st:
            say("WARN", f"stats.json[{key}] lacks q01/q99: policies using QUANTILES normalisation (pi0.5, MolmoAct2) cannot run")
    st = stats.get("action") or {}
    if m["names"] and all(k in st for k in ("q99", "max", "min")):
        for i, name in enumerate(m["names"]):
            if any(w in name.lower() for w in ("gripper", "finger")):
                rng = st["max"][i] - st["min"][i]
                if rng > 0 and (st["max"][i] - st["q99"][i]) / rng > 0.3:
                    say("WARN", f"{name}: max is far above q99 ({st['max'][i]:.4f} vs {st['q99'][i]:.4f}); under QUANTILES normalisation the open gripper saturates. MEAN_STD (ACT default) is unaffected")


def choose_split(m: dict, heldout: str, episodes: str, force_small: bool):
    n = m["n"]
    if heldout.strip():
        held = sorted({int(x) for x in heldout.replace(",", " ").split()})
        train = sorted(set(range(n)) - set(held))
    elif episodes.strip():
        train = sorted(set(json.loads(episodes)))
        held = sorted(set(range(n)) - set(train))
    else:
        if n < 10 and not force_small:
            say("FAIL", f"only {n} episodes: give --heldout/--episodes explicitly or --force-small (holds out episode 0)")
            return None
        k = min(n - 1, max(1, round(0.1 * n)))
        held = sorted({round(i * n / k) for i in range(k)}) if n > 1 else []
        train = sorted(set(range(n)) - set(held))
    oob = sorted(e for e in set(train) | set(held) if not 0 <= e < n)
    if oob:
        say("FAIL", f"episode ids out of range for {n} episodes: {oob}")
        return None
    if not train:
        say("FAIL", "no training episodes left")
        return None
    if set(train) & set(held):
        say("FAIL", f"held-out episodes also in the training list: {sorted(set(train) & set(held))}")
        return None
    arms = m.get("arms")
    if arms and held:
        la = sum(arms.get(e) == "L" for e in held)
        if la in (0, len(held)):
            say("WARN", f"held-out set is all {'left' if la else 'right'}-arm episodes; gripper-timing and idle-arm metrics will be one-sided")
    say("OK", f"split: {len(train)} training episodes, held-out {held or '(none)'}")
    return {"train": train, "held": held, "fps": m["fps"], "action_dim": m["dim"], "n_episodes": n, "names": m["names"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root")
    ap.add_argument("--heldout", default="", help="space/comma separated held-out episode ids")
    ap.add_argument("--episodes", default="", help="JSON list of TRAINING episode ids (held-out = the rest)")
    ap.add_argument("--horizon-seconds", type=float, default=1.0, help="action chunk horizon; chunk = round(horizon * fps)")
    ap.add_argument("--require-flat-names", action="store_true", help="FAIL (not WARN) on the dict form of action.names")
    ap.add_argument("--force-res", action="store_true", help="do not count the non-16:9 aspect warning in the summary (the trainer accepts any aspect; the line still prints)")
    ap.add_argument("--force-action-eq-state", action="store_true", help="proceed although action == state bit-exactly")
    ap.add_argument("--force-small", action="store_true", help="auto-split fewer than 10 episodes")
    ap.add_argument("--emit-split", action="store_true", help="print the split JSON on stdout (report stays on stderr)")
    a = ap.parse_args(argv)
    root = os.path.abspath(a.root)
    m = check_meta(root, a.require_flat_names)
    if m is None:
        return 1
    chunk = max(1, round(a.horizon_seconds * m["fps"]))
    say("OK", f"{m['n']} episodes, {m['info'].get('total_frames')} frames, {m['fps']} fps, {m['dim']} action dims; chunk {chunk} = {chunk / m['fps']:.2f} s")
    if not 0.5 <= chunk / m["fps"] <= 2.0:
        say("WARN", f"horizon {chunk / m['fps']:.2f} s is outside the 0.5-2.0 s range that worked in practice")
    if a.force_res:
        REPORT[:] = [(lvl, msg) for lvl, msg in REPORT if "the in-job downscale keeps the aspect ratio" not in msg]
    check_data(root, m, a.force_action_eq_state)
    check_stats(root, m)
    split = choose_split(m, a.heldout, a.episodes, a.force_small)
    warns = sum(1 for lvl, _ in REPORT if lvl == "WARN")
    print(f"{'FAIL' if FAILS else 'OK'}: {FAILS} fail(s), {warns} warning(s) for {root}", file=sys.stderr)
    if FAILS or split is None:
        return 1
    if a.emit_split:
        split["ds_root"] = root
        split["chunk"] = chunk
        print(json.dumps(split))
    return 0


if __name__ == "__main__":
    sys.exit(main())
