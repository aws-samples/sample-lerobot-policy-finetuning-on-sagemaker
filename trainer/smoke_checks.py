#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Smoke-only checks that a short run really trained (hyperparameter smoke=1; inert in real training jobs).

    python3 smoke_checks.py --ckpt <checkpoint>/pretrained_model --policy act [--out smoke_checks.json]

  weights_moved     fraction of trainable tensors that differ from a fresh policy built from the same train_config.json
                    (a run whose weights did not move loaded nothing or trained nothing)
  bf16_trainable    number of trainable parameters stored in bfloat16 -- must be 0: a bf16 weight cast discards
                    AdamW updates below the bf16 ULP (documented for X-VLA / pi0 / WALL-OSS), fp32 masters are mandatory
  loaded_keys       missing / unexpected keys when the checkpoint's state dict is loaded into a fresh policy
Needs lerobot + torch (runs inside the image only). Exit 0 always; the gate decides on the JSON (thresholds live there).
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="<checkpoint>/pretrained_model directory")
    ap.add_argument("--policy", default="act")
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args(argv)
    import torch
    from lerobot.policies.factory import get_policy_class
    from lerobot.configs.policies import PreTrainedConfig

    result = {"policy": a.policy, "ckpt": a.ckpt}
    try:
        cls = get_policy_class(a.policy)
        adapter = os.path.join(a.ckpt, "adapter_model.safetensors")
        if os.path.isfile(adapter) and not os.path.isfile(os.path.join(a.ckpt, "model.safetensors")):
            # PEFT (LoRA) checkpoint: the trainable tensors are the adapter's; lora_B starts at zero, so a moved tensor is a
            # non-zero lora_B (or a lora_A that differs from its init would need the seed; B is the honest signal)
            from safetensors import safe_open
            saved_dtypes, moved, total, bf16 = {}, 0, 0, 0
            with safe_open(adapter, framework="pt") as sf:
                for k in sf.keys():
                    t = sf.get_tensor(k); saved_dtypes[k] = str(t.dtype).replace("torch.", "").upper()
                    if "lora_B" in k:
                        total += 1; moved += int(bool((t != 0).any()))
                    if t.dtype == torch.bfloat16:
                        bf16 += t.numel()
            result.update({"trainable_tensors": total, "weights_moved_fraction": (moved / total) if total else None,
                           "bf16_trainable_params": bf16, "saved_dtypes": sorted(set(saved_dtypes.values())), "param_dtypes": [],
                           "loaded_keys": {"missing": [], "unexpected": []}, "peft": True})
            json.dump(result, open(a.out, "w"), indent=2); print(json.dumps(result, indent=2)); return 0
        trained = cls.from_pretrained(a.ckpt).to(a.device).eval()
        cfg = PreTrainedConfig.from_pretrained(a.ckpt)
        torch.manual_seed(0)
        fresh = cls(cfg).to(a.device).eval()
        # dtypes must come from the SAVED tensors: from_pretrained re-creates the modules (bf16 for several VLAs) and
        # load_state_dict casts into them, so in-memory dtypes say nothing about what the optimizer trained on
        import glob as _glob
        from safetensors import safe_open
        saved_dtypes = {}
        for f in _glob.glob(os.path.join(a.ckpt, "*.safetensors")):
            with safe_open(f, framework="pt") as sf:
                for k in sf.keys():
                    saved_dtypes[k] = sf.get_slice(k).get_dtype()
        moved, total, bf16 = 0, 0, 0
        fresh_sd = fresh.state_dict()
        for name, p in trained.named_parameters():
            if not p.requires_grad:
                continue
            total += 1
            if saved_dtypes.get(name, str(p.dtype)).upper().replace("TORCH.", "") in ("BF16", "BFLOAT16", "F16", "FLOAT16"):
                bf16 += p.numel()
            f = fresh_sd.get(name)
            if f is None or f.shape != p.shape or not torch.equal(f.to(p.dtype), p):
                moved += 1
        result.update({"trainable_tensors": total, "weights_moved_fraction": (moved / total) if total else None,
                       "bf16_trainable_params": bf16, "saved_dtypes": sorted(set(saved_dtypes.values())),
                       "param_dtypes": sorted({str(p.dtype) for p in trained.parameters()})})
        incompat = fresh.load_state_dict(trained.state_dict(), strict=False)
        result["loaded_keys"] = {"missing": list(incompat.missing_keys), "unexpected": list(incompat.unexpected_keys)}
    except Exception as e:  # report, never crash the job
        result["error"] = f"{type(e).__name__}: {e}"
    text = json.dumps(result, indent=2)
    if a.out:
        with open(a.out, "w") as f:
            f.write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
