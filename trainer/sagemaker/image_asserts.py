# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Build-time assertions for the training image (run by the Dockerfile after the pip layer)."""
import importlib
import sys

for m in ("lerobot", "torch", "torchvision", "av", "pyarrow", "yaml", "transformers", "diffusers", "peft", "timm", "tree", "num2words"):
    importlib.import_module(m)  # nosemgrep -- non-literal-import: m iterates the literal tuple above; build-time check only
for m in ("torchcodec", "decord"):
    try:
        importlib.import_module(m)  # nosemgrep -- non-literal-import: m iterates the literal tuple above
        sys.exit(f"FATAL: {m} must not be importable in this image")
    except ImportError:
        pass
import torch  # noqa: E402

flags = torch._C._cuda_getArchFlags()
assert "sm_70" in flags and "sm_120" in flags, flags
from lerobot.policies.factory import get_policy_class  # noqa: E402

# Every non-stub registry entry's lerobot_type must import here. The registry (policies/*.yaml) is COPYed after this
# step, so the tuple is kept by hand; tests/test_frozen_set.py::test_image_asserts_cover_the_registry compares it with
# the registry. groot is a hard check like the others: its policy class imports without decord (the `groot` extra is
# deliberately not installed, see the Dockerfile; the must-not-import check above keeps decord out).
POLICY_TYPES = ("act", "diffusion", "vqbet", "smolvla", "pi0", "pi05", "pi0_fast", "xvla", "tdmpc", "multi_task_dit", "evo1",
                "molmoact2", "eo1", "fastwam", "lingbot_va", "groot")
for t in POLICY_TYPES:
    get_policy_class(t)
import diffusers  # noqa: E402
import lerobot  # noqa: E402
import transformers  # noqa: E402

print("image ok:", "lerobot", lerobot.__version__, "torch", torch.__version__, "transformers", transformers.__version__,
      "diffusers", diffusers.__version__, "archs", flags)
