#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Prove the published checkpoint loads with stock lerobot, offline, and predicts what the evaluation scored.

    python3 reload_check.py --model-dir /opt/ml/model --dataset-root <ds> --episodes 0 13 [--policy act] [--out reload_check.json]

Runs in-job after publish (no Docker on the laptop needed). It copies pretrained_model/ to a temp dir, patches THE COPY's
config.json to pretrained_backbone_weights=null (ACT constructs its ResNet18 from torchvision *before* reading the
safetensors, so a cold load reaches download.pytorch.org unless the file is pre-staged), loads with empty TORCH_HOME /
HF_HOME, predicts the first held-out chunk and compares it with eval/eval_last_preds.npz (max |diff| must be < 1e-4).
The published artefact itself is never modified. Exit 0 always; the JSON carries the verdict.

When publish.py merged a LoRA adapter into full weights (pretrained_model_merged/ with model.safetensors), that directory is
reloaded instead and --peft-base is ignored: stock lerobot loads the full weights, the stronger proof. Otherwise the adapter
in pretrained_model/ is attached to --peft-base and merged in memory (eval_openloop.load_policy).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile

MERGED_DIRNAME = "pretrained_model_merged"   # written by publish.py


def link_checkpoint_view(src: str, dst: str) -> None:
    """Populate `dst` with symlinks to every file of the checkpoint `src`, except config.json, which is copied so the caller can
    patch it. Copying the whole directory (the old behaviour) moved 24 GB for a merged FastWAM checkpoint into the container's
    /tmp and failed with ENOSPC on the 2026-09-22 p5 smoke; SageMaker's container filesystem is small too."""
    os.makedirs(dst)
    for name in os.listdir(src):
        s, d = os.path.join(src, name), os.path.join(dst, name)
        if name == "config.json":
            shutil.copy2(s, d)
        elif os.path.isdir(s):
            os.symlink(os.path.abspath(s), d, target_is_directory=True)
        else:
            os.symlink(os.path.abspath(s), d)


def reload_source(model_dir: str, peft_base: str = None):
    """(checkpoint dir to reload, peft base to pass, label for the JSON): the merged full weights when present, else pretrained_model/."""
    merged = os.path.join(model_dir, MERGED_DIRNAME)
    if os.path.isfile(os.path.join(merged, "model.safetensors")):
        return merged, None, MERGED_DIRNAME + "/"
    return os.path.join(model_dir, "pretrained_model"), peft_base, "pretrained_model/"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--episodes", type=int, nargs="+", required=True)
    ap.add_argument("--policy", default="act")
    ap.add_argument("--repo-id", default="local/dataset")
    ap.add_argument("--chunk", type=int, default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--history-via-queues", action="store_true", help="registry eval.history_via_queues (VQ-BeT): see eval_openloop.predict_chunk")
    ap.add_argument("--queue-image-key", default=None, help="registry eval.queue_image_key (TD-MPC)")
    ap.add_argument("--chunk-first", action="store_true", help="registry eval.chunk_first (TD-MPC)")
    ap.add_argument("--no-history", action="store_true", help="registry eval.no_history (TD-MPC)")
    ap.add_argument("--tolerance-s", type=float, default=1e-4, help="registry eval.tolerance_s (dataset timestamp tolerance)")
    ap.add_argument("--reset-per-call", action="store_true", help="registry eval.reset_per_call")
    ap.add_argument("--peft-base", default=None, help="base repo id for a PEFT adapter checkpoint")
    ap.add_argument("--stochastic", action="store_true", help="sampling policy (n_samples > 1 in the registry): compare shapes, report the diff only")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    result = {"policy": a.policy, "ok": False}
    tmp = tempfile.mkdtemp(prefix="reload_")
    try:
        import numpy as np
        import torch
        src, peft_base, result["checkpoint"] = reload_source(a.model_dir, a.peft_base)
        dst = os.path.join(tmp, "pretrained_model")
        link_checkpoint_view(src, dst)   # symlinks to the weights, a private copy of config.json (a merged FastWAM copy is 24 GB)
        cfg_p = os.path.join(dst, "config.json")
        cfg = json.load(open(cfg_p))
        patched = False
        if cfg.get("pretrained_backbone_weights"):
            cfg["pretrained_backbone_weights"] = None  # offline: no torch.hub fetch of the ImageNet init at construction
            patched = True
            json.dump(cfg, open(cfg_p, "w"), indent=2)
        result["config_patched_in_copy"] = patched
        os.environ["TORCH_HOME"] = os.path.join(tmp, "torch")   # ACT's ResNet18 init must not be fetched: proves the patch
        # policies that reference their backbone by name (SmolVLA -> SmolVLM2, X-VLA -> BART tokenizer) need the staged
        # Hub cache the job trained with; keep it when the entrypoint mounted one, otherwise use an empty cache
        if not (os.environ.get("HF_HUB_OFFLINE") == "1" and os.environ.get("HF_HOME")):
            os.environ["HF_HOME"] = os.path.join(tmp, "hf")
        os.environ["HF_HUB_OFFLINE"] = "1"
        result["hf_home"] = os.environ["HF_HOME"]
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors
        cls = get_policy_class(a.policy)
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from eval_openloop import load_policy
        policy = load_policy(cls, dst, a.device, peft_base)
        pre, post = make_pre_post_processors(policy_cfg=policy.config, pretrained_path=dst,
                                             preprocessor_overrides={"device_processor": {"device": a.device}})
        H = a.chunk or getattr(policy.config, "chunk_size", None) or getattr(policy.config, "n_action_steps", 1)
        fps = json.load(open(os.path.join(a.dataset_root, "meta", "info.json")))["fps"]
        delta = {"action": [i / fps for i in range(H)]}
        obs_idx = None if a.no_history else getattr(policy.config, "observation_delta_indices", None)
        probe = LeRobotDataset(a.repo_id, root=a.dataset_root, episodes=a.episodes[:1], video_backend="pyav", tolerance_s=a.tolerance_s)
        if obs_idx:  # observation history (diffusion, vqbet, ...): stack past frames like the eval harness does
            for k in list(probe.meta.camera_keys) + ["observation.state"]:
                delta[k] = [i / fps for i in obs_idx]
        ds = LeRobotDataset(a.repo_id, root=a.dataset_root, episodes=a.episodes, delta_timestamps=delta, video_backend="pyav", tolerance_s=a.tolerance_s)
        item = ds[0]
        obs = {"observation.state": item["observation.state"]}
        for k in ds.meta.camera_keys:
            obs[k] = item[k]
        if "task" in item:
            obs["task"] = item["task"]
        if obs_idx:  # history tensors are already 2-D/4-D: add the batch axis ourselves (see eval_openloop.py)
            obs = {k: (v.unsqueeze(0) if hasattr(v, "unsqueeze") else v) for k, v in obs.items()}
        with torch.no_grad():
            torch.manual_seed(1000 + int(item.get("index", 0)))
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from eval_openloop import predict_chunk   # same history protocol as the evaluation that produced the reference
            chunk = predict_chunk(policy, pre(obs), a.history_via_queues, a.queue_image_key, a.chunk_first, a.reset_per_call)
            pred = post(chunk.float() if isinstance(chunk, torch.Tensor) else chunk)
            pred = pred.detach().cpu().float().numpy()[0] if hasattr(pred, "detach") else np.asarray(pred)[0]
        result["pred_shape"] = list(pred.shape)
        npz = os.path.join(a.model_dir, "eval", "eval_last_preds.npz")
        if os.path.isfile(npz):
            ref = np.load(npz)
            key = f"ep{a.episodes[0]}_pred"
            if key in ref:
                ref0 = ref[key][0]
                n = min(len(ref0), len(pred))
                diff = float(np.max(np.abs(ref0[:n] - pred[:n])))
                if a.stochastic:  # fresh noise every call: the eval averaged n_samples draws, this is one draw
                    result.update({"max_abs_diff_vs_eval": diff, "ok": ref0.shape[1:] == pred.shape[1:], "note": "stochastic policy: shape compared, diff informational"})
                else:
                    result.update({"max_abs_diff_vs_eval": diff, "ok": diff < a.tol})
            else:
                result["note"] = f"{key} not in eval_last_preds.npz (keys {list(ref.keys())[:4]}...)"; result["ok"] = True
        else:
            result["note"] = "no eval_last_preds.npz to compare against; load + predict succeeded"; result["ok"] = True
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    text = json.dumps(result, indent=2)
    if a.out:
        with open(a.out, "w") as f:
            f.write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
