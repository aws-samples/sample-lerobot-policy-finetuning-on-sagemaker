#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Derive a one-camera view of a LeRobot v3.0 dataset for single-image policies (registry `single_camera_required`).

lerobot 0.6.1 VQ-BeT and TD-MPC raise "You must provide only one image among the inputs." when the dataset has more than
one camera. lerobot derives the camera list from meta/info.json["features"], so videos of a camera that is not listed are
never opened: the one-camera view is metadata only. data/ and the kept camera's videos/ directory are symlinked, meta/ is
copied with the other image features removed from info.json and stats.json.

    python3 select_camera.py <root> [--keep observation.images.X] [--out DIR]

Prints the root to train on: <root> itself when it already has one camera, else the derived directory. Without --keep
the first camera in sorted order is kept and the choice is reported on stderr. Refuses to overwrite a directory it did not
create (marker meta/_SELECT_CAMERA.json).
"""
import argparse
import json
import os
import shutil
import sys

MARKER = "_SELECT_CAMERA.json"


def image_keys(info: dict) -> list:
    return sorted(k for k, v in info.get("features", {}).items() if v.get("dtype") in ("video", "image"))


def derive(root: str, keep=None, out: str = None) -> str:
    """keep: one camera key, a list of keys (policies.<name>.cameras), or None = the first camera in sorted order."""
    root = os.path.abspath(root)
    with open(os.path.join(root, "meta", "info.json")) as f:
        info = json.load(f)
    keys = image_keys(info)
    keeps = [keep] if isinstance(keep, str) else list(keep or [])
    bad = [k for k in keeps if k not in keys]
    if bad:
        sys.exit(f"FATAL: camera(s) {bad} are not in the dataset; cameras: {keys}")
    if not keeps:
        keeps = keys[:1]
    if set(keeps) == set(keys) or len(keys) <= 1:
        return root
    keep = keeps[0]
    dropped = [k for k in keys if k not in keeps]
    out = os.path.abspath(out or f"{root}_1cam")
    if os.path.exists(out):
        if not os.path.isfile(os.path.join(out, "meta", MARKER)):
            sys.exit(f"FATAL: {out} exists and was not created by select_camera.py; refusing to overwrite")
        shutil.rmtree(out)
    os.makedirs(os.path.join(out, "videos"))
    os.symlink(os.path.join(root, "data"), os.path.join(out, "data"))
    for k in keeps:
        src_vid = os.path.join(root, "videos", k)
        if os.path.isdir(src_vid):   # dtype "image" cameras live in the parquet files and have no videos/ directory
            os.symlink(src_vid, os.path.join(out, "videos", k))
    shutil.copytree(os.path.join(root, "meta"), os.path.join(out, "meta"))
    info["features"] = {k: v for k, v in info["features"].items() if k not in dropped}
    with open(os.path.join(out, "meta", "info.json"), "w") as f:
        json.dump(info, f, indent=4)
    stats_path = os.path.join(out, "meta", "stats.json")
    if os.path.isfile(stats_path):
        with open(stats_path) as f:
            stats = json.load(f)
        with open(stats_path, "w") as f:
            json.dump({k: v for k, v in stats.items() if k not in dropped}, f, indent=4)
    with open(os.path.join(out, "meta", MARKER), "w") as f:
        json.dump({"source": root, "kept": keeps, "dropped": dropped}, f, indent=2)
    print(f"select_camera: keeping {keeps}, ignoring {dropped} (CAMERA / CAMERAS, policies.<name>.camera / cameras choose)", file=sys.stderr)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root")
    ap.add_argument("--keep", action="append", default=None, help="image feature to keep (repeatable; default: the first in sorted order)")
    ap.add_argument("--out", default=None, help="where to write the view (default: <root>_1cam)")
    a = ap.parse_args(argv)
    print(derive(a.root, a.keep, a.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
