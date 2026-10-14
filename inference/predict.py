#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Minimal "deploy" example: load a checkpoint this pipeline produced and predict one action chunk, offline.

    python3 inference/predict.py --model-dir runs/<run>/model --dataset-root datasets/<name> --episode 0 [--frame 0]

Needs a machine with `pip install "lerobot[training]==0.6.1"` (a robot PC, an EC2 GPU box or the training image itself).
No network is touched: the copy of config.json is patched to skip the torchvision ImageNet download that ACT would
otherwise perform at construction, and HF_HUB_OFFLINE is set. Adapt `observation` to your robot's camera and state
readers; the keys must be the dataset's feature names (see datasets/<name>/meta/info.json -> features).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", required=True, help="folder holding pretrained_model/ (runs/<run>/model or an extracted model.tar.gz)")
    ap.add_argument("--dataset-root", required=True, help="a LeRobot v3.0 dataset to take one observation from")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--policy", default=None, help="lerobot policy type (default: from SUMMARY.json, else act)")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    summary_p = os.path.join(a.model_dir, "SUMMARY.json")
    policy_type = a.policy or (json.load(open(summary_p)).get("policy") if os.path.exists(summary_p) else None) or "act"

    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    tmp = tempfile.mkdtemp(prefix="predict_")
    try:
        ckpt = os.path.join(tmp, "pretrained_model")
        src = os.path.join(a.model_dir, "pretrained_model")
        os.makedirs(ckpt)
        for name in os.listdir(src):   # symlink the weights (a merged FastWAM copy is 24 GB), copy only the config we patch
            if name == "config.json":
                shutil.copy2(os.path.join(src, name), os.path.join(ckpt, name))
            else:
                os.symlink(os.path.abspath(os.path.join(src, name)), os.path.join(ckpt, name))
        cfg_p = os.path.join(ckpt, "config.json")
        cfg = json.load(open(cfg_p))
        if cfg.get("pretrained_backbone_weights"):
            cfg["pretrained_backbone_weights"] = None     # weights are in the safetensors; skip the torch.hub fetch
            json.dump(cfg, open(cfg_p, "w"), indent=2)
        os.environ["TORCH_HOME"] = os.path.join(tmp, "torch")
        policy = get_policy_class(policy_type).from_pretrained(ckpt).to(a.device).eval()
        pre, post = make_pre_post_processors(policy_cfg=policy.config, pretrained_path=ckpt,
                                             preprocessor_overrides={"device_processor": {"device": a.device}})
        ds = LeRobotDataset("local/dataset", root=a.dataset_root, episodes=[a.episode], video_backend="pyav")
        item = ds[min(a.frame, len(ds) - 1)]
        observation = {"observation.state": item["observation.state"]}
        for k in ds.meta.camera_keys:
            observation[k] = item[k]                       # (3, H, W) float tensors in [0, 1], as the dataset stores them
        if "task" in item:
            observation["task"] = item["task"]
        with torch.no_grad():
            chunk = policy.predict_action_chunk(pre(observation))
            actions = post(chunk.float() if isinstance(chunk, torch.Tensor) else chunk)
        actions = actions.detach().cpu().numpy()[0] if hasattr(actions, "detach") else actions[0]
        names = ds.meta.features["action"].get("names")
        names = names.get("axes") if isinstance(names, dict) else names
        print(f"policy {policy_type}: {actions.shape[0]} future actions x {actions.shape[1]} dims for episode {a.episode} frame {a.frame}")
        for t, row in enumerate(actions[:5]):
            print(f"  t+{t}: " + " ".join(f"{v:8.4f}" for v in row))
        if names:
            print("  dims: " + ", ".join(names))
        print("  (absolute joint targets in the dataset's units; play them back at the dataset's fps)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
