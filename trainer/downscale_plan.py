#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Decide the per-camera target size for the in-job downscale (aspect-preserving, even width) and print it.

    python3 downscale_plan.py <dataset_root> [--height 360]     -> one line per camera: "<camera> <width> <height>"

A 720x1280 (16:9) camera at height 360 becomes 640x360, exactly what the validated ACT run used. A 480x640 (4:3)
camera becomes 480x360 and is reported as a WARNING on stderr: training works, but the numbers are not comparable
to the 16:9 reference. Cameras already at the target height are listed too (the caller decides whether to re-encode).
Exit 1 only when a camera feature has no readable shape.
"""
from __future__ import annotations

import argparse
import json
import sys


def plan(info: dict, height: int):
    out, warnings = [], []
    for key, feat in info["features"].items():
        if feat.get("dtype") not in ("video", "image"):
            continue
        shape = feat.get("shape") or []
        hw = [x for x in shape if x != 3]
        if len(hw) != 2:
            raise ValueError(f"{key}: cannot identify height/width in shape {shape}")
        h, w = hw
        width = 2 * round(height * w / h / 2)  # even width for yuv420p
        ratio = w / h
        if abs(ratio - 16 / 9) > 0.02:
            warnings.append(f"{key} is {h}x{w} (aspect {ratio:.3f}, not 16:9): downscaled to {width}x{height}; "
                            f"the width follows the aspect ratio and widths other than 640 are fine; reference numbers are per dataset")
        out.append((key, width, height, h, w))
    return out, warnings


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root")
    ap.add_argument("--height", type=int, default=360)
    a = ap.parse_args(argv)
    info = json.load(open(f"{a.root}/meta/info.json"))
    try:
        rows, warnings = plan(info, a.height)
    except ValueError as e:
        print(f"FATAL: {e}", file=sys.stderr)
        return 1
    if not rows:
        print("FATAL: no video/image features in meta/info.json", file=sys.stderr)
        return 1
    for w in warnings:
        print(f"WARN {w}", file=sys.stderr)
    for key, width, height, h, w in rows:
        print(f"{key} {width} {height}")
        print(f"{key}: {h}x{w} -> {height}x{width}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
