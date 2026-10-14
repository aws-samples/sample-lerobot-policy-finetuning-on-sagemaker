#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Print shell exports a registry argv may reference, derived from the dataset (eval'd by train_policy.sh):
  CAMERA_KEYS         image features in meta/info.json order, quoted and comma-joined for a draccus list:
                      --policy.obs_cam_keys=[${CAMERA_KEYS}]  ->  ["observation.images.top","observation.images.wrist"]
  ACTION_CHANNEL_IDS  "[0,...,D-1]" for the dataset's action width: --policy.used_action_channel_ids=${ACTION_CHANNEL_IDS}
  ACTION_DIM / STATE_DIM  widths of action / observation.state (FastWAM: --policy.action_dim=${ACTION_DIM} --policy.proprio_dim=${STATE_DIM})
"""
import json
import shlex
import sys


def exports(root: str) -> str:
    with open(f"{root}/meta/info.json") as f:
        feats = json.load(f)["features"]
    cams = [k for k, v in feats.items() if v.get("dtype") in ("video", "image")]
    camera_keys = ",".join('"' + c + '"' for c in cams)
    action_dim = int(feats["action"]["shape"][0]); state_dim = int((feats.get("observation.state") or {}).get("shape", [action_dim])[0])
    ids = "[" + ",".join(str(i) for i in range(action_dim)) + "]"
    return (f"export CAMERA_KEYS={shlex.quote(camera_keys)}\nexport ACTION_CHANNEL_IDS={shlex.quote(ids)}\n"
            f"export ACTION_DIM={action_dim}\nexport STATE_DIM={state_dim}")


if __name__ == "__main__":
    print(exports(sys.argv[1]))
