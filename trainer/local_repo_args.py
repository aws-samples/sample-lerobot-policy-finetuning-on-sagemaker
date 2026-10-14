#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Turn a registry entry's `train.local_repo_args` into --policy.<field>=<staged snapshot directory> overrides.

Some loaders cannot resolve a Hub repo id offline even from a complete cache: transformers' AutoProcessor with
trust_remote_code probes files that do not exist in the repo (pi0_fast: bpe_tokenizer/config.json), which is a tolerated
404 online and a hard LocalEntryNotFoundError under HF_HUB_OFFLINE=1. Passing the snapshot directory instead of the repo id
side-steps the probe. train_policy.sh appends the printed tokens to OVERRIDES when the hub channel is mounted.

    python3 local_repo_args.py --policy pi0_fast --hf-home /opt/ml/input/data/hub
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render_flags  # noqa: E402


def snapshot_dir(hf_home: str, repo: str) -> str:
    repo_dir = os.path.join(hf_home, "hub", "models--" + repo.replace("/", "--"))
    ref = os.path.join(repo_dir, "refs", "main")
    if os.path.isfile(ref):
        with open(ref) as f:
            sha = f.read().strip()
        d = os.path.join(repo_dir, "snapshots", sha)
        if os.path.isdir(d):
            return d
    snaps = sorted(glob.glob(os.path.join(repo_dir, "snapshots", "*")))
    if len(snaps) == 1:
        return snaps[0]
    sys.exit(f"FATAL: {repo} is not staged under {hf_home} (no snapshot for refs/main); check hub_artifacts / _STAGED.json")


def tokens(policy: str, hf_home: str) -> list:
    spec = render_flags.load_policy(policy)
    return [f"{e['flag']}={snapshot_dir(hf_home, e['repo'])}" for e in (spec.get("train") or {}).get("local_repo_args", [])]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--hf-home", required=True)
    a = ap.parse_args(argv)
    print("\n".join(tokens(a.policy, a.hf_home)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
