#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Keep only the newest K complete checkpoints under <output_dir>/train/checkpoints (lerobot writes one every save_freq
and never deletes any; SageMaker mirrors the directory to S3, so a 16 GB pi0 checkpoint every 1000 steps costs ~$0.4/month
each). Run by train_entrypoint.sh every two minutes during training and once at the end (compute.keep_checkpoints, 0 = keep all).

Rules: a checkpoint is complete when pretrained_model/train_config.json and training_state/ exist; incomplete (in-progress)
directories are never touched; the directory `last` points to is never deleted; the newest K complete ones stay.
"""
import argparse
import os
import shutil
import sys


def complete(d: str) -> bool:
    return os.path.isfile(os.path.join(d, "pretrained_model", "train_config.json")) and os.path.isdir(os.path.join(d, "training_state"))


def prune(root: str, keep: int) -> list:
    if keep <= 0 or not os.path.isdir(root):
        return []
    numbered = sorted((d for d in os.listdir(root) if d.isdigit() and os.path.isdir(os.path.join(root, d))), key=int)
    done = [d for d in numbered if complete(os.path.join(root, d))]
    protected = set(done[-keep:])
    last = os.path.join(root, "last")
    if os.path.islink(last):
        protected.add(os.path.basename(os.path.normpath(os.readlink(last))))
    removed = []
    for d in done:
        if d not in protected:
            shutil.rmtree(os.path.join(root, d), ignore_errors=True)
            removed.append(d)
    return removed


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="<output_dir>/train/checkpoints")
    ap.add_argument("--keep", type=int, required=True, help="newest complete checkpoints to keep; 0 keeps everything")
    a = ap.parse_args(argv)
    removed = prune(a.root, a.keep)
    if removed:
        print(f"prune_checkpoints: removed {len(removed)} older checkpoint(s) {removed} (keep {a.keep})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
