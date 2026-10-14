# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""EO-1's image budget under the pinned transformers, measured in the training image (docker lane, `pytest -m docker`).

lerobot 0.6.1 processor_eo1.py passes `min_pixels` / `max_pixels` to `Processor.apply_chat_template` as plain kwargs.
transformers 5.5.4 logs "Kwargs passed to `processor.__call__` have to be in `processor_kwargs` dict" on every training step
but still applies them; the recommended `processor_kwargs={...}` form, however, does NOT apply them (the images keep the
default budget: more visual tokens, more memory). trainer/amp_driver.py therefore silences the warning instead of
following it. These tests pin that behaviour so a transformers bump that changes either half is noticed:
  - the flat form must keep shrinking the images (otherwise eo1's memory and token counts change silently);
  - the processor_kwargs form is recorded as ignored (strict xfail: when transformers fixes it, this XPASSes and the
    lerobot call site + docs/TRAPS.md must be revisited).
The Qwen2.5-VL-3B-Instruct processor files (~12 MB, ungated) are fetched on the host once; the container runs offline.
"""
import json
import pathlib
import subprocess

import pytest

from tests.conftest import TRAIN_IMAGE, docker_lane_unavailable

pytestmark = pytest.mark.docker

REPO = pathlib.Path(__file__).resolve().parent.parent
QWEN = "Qwen/Qwen2.5-VL-3B-Instruct"
PROBE = r'''
import json, logging, sys, torch
from transformers import Qwen2_5_VLProcessor
logging.getLogger("transformers").setLevel(logging.ERROR)
p = Qwen2_5_VLProcessor.from_pretrained(sys.argv[1], use_fast=True)
torch.manual_seed(0)
imgs = [torch.rand(3, 360, 480), torch.rand(3, 360, 480)]
msgs = [[{"role": "user", "content": [{"type": "image", "image": imgs[0]}, {"type": "image", "image": imgs[1]},
                                      {"type": "text", "text": "pick up the cube"}]}]]
common = dict(tokenize=True, padding=True, padding_side="right", add_generation_prompt=False, return_dict=True, return_tensors="pt")
lim = {"min_pixels": 64 * 28 * 28, "max_pixels": 128 * 28 * 28}
out = {}
for name, kw in (("none", {}), ("flat", dict(lim)), ("processor_kwargs", {"processor_kwargs": dict(lim)})):
    o = p.apply_chat_template(msgs, **kw, **common)
    out[name] = {"tokens": int(o["input_ids"].shape[1]), "grid": o["image_grid_thw"][0].tolist(), "patches": int(o["pixel_values"].shape[0])}
print("RESULT " + json.dumps(out))
'''


@pytest.fixture(scope="module")
def budgets(tmp_path_factory):
    if docker_lane_unavailable():
        pytest.skip("docker lane unavailable")
    try:
        from huggingface_hub import snapshot_download
        # local_dir: real files, no cache symlinks (a mounted snapshot dir would point at blobs outside the mount)
        local = snapshot_download(QWEN, allow_patterns=["*.json", "merges.txt", "vocab.json", "tokenizer*", "chat_template*"],
                                  ignore_patterns=["*.safetensors*"], local_dir=str(tmp_path_factory.mktemp("qwen")))  # nosec B615: processor/tokenizer JSON only, fetched to measure token counts; no weights are loaded
    except Exception as e:  # no network: the docker lane stays offline-only
        pytest.skip(f"cannot fetch the {QWEN} processor files: {type(e).__name__}")
    probe = tmp_path_factory.mktemp("probe") / "probe.py"
    probe.write_text(PROBE)
    cmd = ["docker", "run", "--rm", "--network", "none", "-e", "HF_HUB_OFFLINE=1", "-e", "TRANSFORMERS_OFFLINE=1",
           "-v", f"{local}:/qwen:ro", "-v", f"{probe}:/probe.py:ro", "--entrypoint", "python", TRAIN_IMAGE, "/probe.py", "/qwen"]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    line = [l for l in res.stdout.splitlines() if l.startswith("RESULT ")]
    assert res.returncode == 0 and line, res.stdout[-2000:] + res.stderr[-2000:]
    return json.loads(line[-1][len("RESULT "):])


def test_default_budget_is_larger_than_eo1s(budgets):
    # 360x480 frames stay near native resolution under the default budget: the eo1 limits must bite
    assert budgets["none"]["tokens"] > budgets["flat"]["tokens"], budgets


def test_flat_kwargs_apply_the_eo1_budget(budgets):
    # 128 * 28 * 28 max pixels -> at most 128 merged patches (2x2 of 14 px) per image; the grid is t*h*w 14-px patches
    t, h, w = budgets["flat"]["grid"]
    assert t * h * w // 4 <= 128, budgets["flat"]
    assert budgets["flat"]["patches"] < budgets["none"]["patches"], budgets


@pytest.mark.xfail(strict=True, reason="transformers 5.5.4 ignores min/max_pixels given via processor_kwargs in apply_chat_template; "
                                      "when this XPASSes, switch lerobot's eo1 call to processor_kwargs and drop the amp_driver filter")
def test_processor_kwargs_form_applies_the_budget(budgets):
    assert budgets["processor_kwargs"]["tokens"] == budgets["flat"]["tokens"], budgets
