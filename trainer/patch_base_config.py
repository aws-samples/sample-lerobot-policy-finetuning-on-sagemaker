#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Apply the registry's `base_config_patch` / `base_patches` to the staged base checkpoint files (every snapshot of the
base_checkpoint repo under $HF_HOME/hub). draccus merges dict overrides given on the command line, so a base that ships
another robot's input_features / output_features cannot be emptied with --policy.input_features={}; an empty dict in the
file makes lerobot's make_policy rebuild the features from the dataset.

    python3 patch_base_config.py --policy lingbot_va --hf-home /opt/ml/input/data/hub
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render_flags  # noqa: E402


def _hub(hf_home: str, text: str) -> str:
    """Resolve ${HUB:<repo>} placeholders to the staged snapshot directory of that repo."""
    import re
    from local_repo_args import snapshot_dir
    return re.sub(r"\$\{HUB:([^}]+)\}", lambda m: snapshot_dir(hf_home, m.group(1)), text)


def patch(policy: str, hf_home: str) -> list:
    """Apply `base_config_patch` (dict merged into config.json) and `base_patches` (list of {file, merge|replace_text})
    to every snapshot of the registry entry's base_checkpoint under $HF_HOME/hub. Idempotent; returns the files written."""
    spec = render_flags.load_policy(policy)
    base = spec.get("base_checkpoint")
    patches = list(spec.get("base_patches") or [])
    if spec.get("base_config_patch"):
        patches.insert(0, {"file": "config.json", "merge": spec["base_config_patch"]})
    if not patches or not base:
        return []
    repo_dir = os.path.join(hf_home, "hub", "models--" + base.replace("/", "--"))
    snaps = glob.glob(os.path.join(repo_dir, "snapshots", "*"))
    if not snaps:
        sys.exit(f"FATAL: no staged snapshot for {base} under {repo_dir}")
    done = []
    for snap in snaps:
        for entry in patches:
            path = os.path.join(snap, entry["file"])
            if not os.path.isfile(path):
                continue
            real = os.path.realpath(path)
            with open(real) as f:
                text = f.read()
            new = text
            if entry.get("merge"):
                cfg = json.loads(text)
                if any(cfg.get(k) != v for k, v in entry["merge"].items()):
                    cfg.update(entry["merge"]); new = json.dumps(cfg, indent=2)
            for old, rep in (entry.get("replace_text") or {}).items():
                new = new.replace(old, _hub(hf_home, rep))
            if new != text:
                with open(real, "w") as f:
                    f.write(new)
                done.append(real)
    return done

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--hf-home", required=True)
    a = ap.parse_args(argv)
    for pth in patch(a.policy, a.hf_home):
        print(f"patched base config: {pth}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
