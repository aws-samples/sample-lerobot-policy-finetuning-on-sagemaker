# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Drive lerobot's train() with a hand-built Accelerator so we get
fp32 master weights (policy.dtype=float32 -> params + AdamW moments fp32)
PLUS bf16 autocast on the forward/backward (accelerate mixed_precision='bf16').

lerobot-train itself cannot do this: lerobot/scripts/lerobot_train.py:248 maps
policy.dtype -> mixed_precision via {"bfloat16":"bf16","float16":"fp16","float32":"no"},
so float32 weights always imply mixed_precision="no" (no autocast).
Accelerate applies the autocast in accelerator.py:1818-1827 (prepare_model).
This driver does not modify any installed file; it only passes its own Accelerator
into train(cfg, accelerator=...) (allowed by lerobot/configs/parser.py:436).
"""

import logging

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

from lerobot.scripts.lerobot_train import train

def _fp32_masters(policy):
    """Cast trainable parameters to float32 so AdamW keeps fp32 masters; bf16 autocast still runs the math in bf16.

    lerobot builds several VLAs (SmolVLA's action expert, X-VLA, pi0) in bfloat16 when their backbone is loaded in bf16.
    A bf16-stored weight silently discards every update smaller than half its ULP (measured on X-VLA: 20-60 % of the
    elements never change), so the smoke gate refuses any trainable bf16 parameter. Frozen parameters keep their dtype.
    """
    n = 0
    for prm in policy.parameters():
        if prm.requires_grad and prm.dtype in (torch.bfloat16, torch.float16):
            prm.data = prm.data.float()
            n += prm.numel()
    if n:
        print(f"[amp-driver] cast {n} trainable parameters to float32 (fp32 masters)", flush=True)
    return policy


def _patch_make_policy() -> None:
    import lerobot.policies.factory as factory
    import lerobot.scripts.lerobot_train as lt
    original = factory.make_policy

    def make_policy_fp32(*args, **kwargs):
        return _fp32_masters(original(*args, **kwargs))

    factory.make_policy = make_policy_fp32
    if hasattr(lt, "make_policy"):
        lt.make_policy = make_policy_fp32


_PROCESSOR_KWARGS_WARNING = "Kwargs passed to `processor.__call__` have to be in `processor_kwargs` dict"


def _silence_processor_kwargs_warning() -> None:
    """Drop transformers' per-step "Kwargs passed to `processor.__call__` have to be in `processor_kwargs` dict".

    lerobot 0.6.1 processor_eo1.py hands EO-1's image budget (image_min_pixels / image_max_pixels) to
    Processor.apply_chat_template as plain kwargs; transformers 5.5.4 logs this line on EVERY step. Its advice must NOT
    be followed in this image: measured on the release image, `processor_kwargs={"min_pixels": ..., "max_pixels": ...}`
    is ignored by apply_chat_template (two 360x480 frames -> 469 visual tokens, the default budget) while the plain
    kwargs are honoured (261 tokens). tests/test_eo1_pixel_budget_docker.py pins both facts; when transformers fixes the
    processor_kwargs path that test XPASSes and this filter plus lerobot's call site are to be revisited (docs/TRAPS.md).
    """
    class _Drop(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            return _PROCESSOR_KWARGS_WARNING not in record.getMessage()

    for name in ("transformers.processing_utils", "transformers"):
        logging.getLogger(name).addFilter(_Drop())


def main() -> None:
    _patch_make_policy()
    _silence_processor_kwargs_warning()
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    acc = Accelerator(
        step_scheduler_with_optimizer=False,
        mixed_precision="bf16",
        kwargs_handlers=[ddp_kwargs],
    )
    print(
        f"[amp-driver] mixed_precision={acc.mixed_precision} native_amp={acc.native_amp} "
        f"device={acc.device} torch={torch.__version__}",
        flush=True,
    )
    train(accelerator=acc)


if __name__ == "__main__":
    # lerobot's dataloader workers use the 'spawn' start method, so the module
    # body must be import-safe.
    main()
