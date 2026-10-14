#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Complete meta/stats.json with q01/q10/q50/q90/q99 for policies that normalise with QUANTILES (registry
`requires_quantile_stats`: pi0.5, MolmoAct2). Datasets exported without quantiles make lerobot raise on the first batch,
after the multi-GB base weights are loaded.

Like select_camera.py this is a metadata-only view of the dataset: data/ and videos/ are symlinked, meta/ is copied and
stats.json gains the missing quantiles, computed over every data parquet file for each 1-D float feature that lacks them.

    python3 ensure_quantiles.py <root> [--out DIR]

Prints the root to train on: <root> itself when nothing is missing, else the derived directory.
"""
import argparse
import glob
import json
import os
import shutil
import sys

import numpy as np

MARKER = "_ENSURE_QUANTILES.json"
QUANTILES = {"q01": 1, "q10": 10, "q50": 50, "q90": 90, "q99": 99}


def missing_quantiles(root: str) -> list:
    with open(os.path.join(root, "meta", "info.json")) as f:
        info = json.load(f)
    with open(os.path.join(root, "meta", "stats.json")) as f:
        stats = json.load(f)
    out = []
    for key, ft in info.get("features", {}).items():
        if ft.get("dtype") not in ("float32", "float64") or len(ft.get("shape", [])) != 1:
            continue
        st = stats.get(key)
        if st is None or any(q not in st for q in ("q01", "q99")):
            out.append(key)
    return out


IMAGENET = {"mean": [[[0.485]], [[0.456]], [[0.406]]], "std": [[[0.229]], [[0.224]], [[0.225]]],
            "min": [[[0.0]], [[0.0]], [[0.0]]], "max": [[[1.0]], [[1.0]], [[1.0]]]}


def missing_image_stats(root: str) -> list:
    """Camera features with no stats.json entry: lerobot's make_dataset indexes stats[<camera>] to overwrite mean/std with
    the ImageNet values (use_imagenet_stats), so the key must exist even though its values are replaced."""
    with open(os.path.join(root, "meta", "info.json")) as f:
        info = json.load(f)
    with open(os.path.join(root, "meta", "stats.json")) as f:
        stats = json.load(f)
    return [k for k, v in info.get("features", {}).items() if v.get("dtype") in ("video", "image") and k not in stats]


def compute(root: str, keys: list) -> dict:
    import pyarrow.parquet as pq
    cols = {k: [] for k in keys}
    for f in sorted(glob.glob(os.path.join(root, "data", "**", "*.parquet"), recursive=True)):
        t = pq.read_table(f, columns=[k for k in keys if k in pq.read_schema(f).names])
        for k in t.schema.names:
            cols[k].append(np.asarray(t.column(k).to_pylist(), dtype=np.float64))
    result = {}
    for k in keys:
        if not cols[k]:
            continue
        arr = np.concatenate(cols[k], axis=0)
        arr = arr.reshape(arr.shape[0], -1)
        result[k] = {name: np.percentile(arr, p, axis=0).astype(np.float32).tolist() for name, p in QUANTILES.items()}
    return result


def derive(root: str, out: str = None, quantiles: bool = True) -> str:
    root = os.path.abspath(root)
    keys = missing_quantiles(root) if quantiles else []
    cams = missing_image_stats(root)
    if not keys and not cams:
        return root
    quant = compute(root, keys) if keys else {}
    if keys and not quant:
        sys.exit(f"FATAL: {keys} lack q01/q99 and are not columns of the data parquet files")
    with open(os.path.join(root, "meta", "info.json")) as f:
        total = int(json.load(f).get("total_frames") or 0)
    for cam in cams:   # ImageNet-shaped (3,1,1) placeholders; lerobot replaces mean/std with the ImageNet values anyway
        quant[cam] = {**IMAGENET, "count": [total]}
    out = os.path.abspath(out or f"{root}_q")
    if os.path.exists(out):
        if not os.path.isfile(os.path.join(out, "meta", MARKER)):
            sys.exit(f"FATAL: {out} exists and was not created by ensure_quantiles.py; refusing to overwrite")
        shutil.rmtree(out)
    os.makedirs(out)
    for d in ("data", "videos"):
        if os.path.isdir(os.path.join(root, d)):
            os.symlink(os.path.join(root, d), os.path.join(out, d))
    shutil.copytree(os.path.join(root, "meta"), os.path.join(out, "meta"), symlinks=False)
    stats_path = os.path.join(out, "meta", "stats.json")
    with open(stats_path) as f:
        stats = json.load(f)
    for k, q in quant.items():
        stats.setdefault(k, {}).update(q)
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=4)
    with open(os.path.join(out, "meta", MARKER), "w") as f:
        json.dump({"source": root, "completed": sorted(quant), "image_stats_added": cams}, f, indent=2)
    if keys:
        print(f"ensure_quantiles: added q01/q10/q50/q90/q99 for {sorted(keys)} (computed over the data files)", file=sys.stderr)
    if cams:
        print(f"ensure_quantiles: added ImageNet-shaped stats entries for cameras {cams} (lerobot requires the keys; it overwrites mean/std)", file=sys.stderr)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root")
    ap.add_argument("--out", default=None, help="where to write the view (default: <root>_q)")
    ap.add_argument("--quantiles", action="store_true", help="also complete q01/q99 (QUANTILES policies); without it only missing camera stats entries are added")
    a = ap.parse_args(argv)
    print(derive(a.root, a.out, a.quantiles))
    return 0


if __name__ == "__main__":
    sys.exit(main())
