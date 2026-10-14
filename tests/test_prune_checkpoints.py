# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "trainer"))
import prune_checkpoints  # noqa: E402


def _ckpt(root, step, complete=True):
    d = root / f"{step:06d}"; (d / "pretrained_model").mkdir(parents=True)
    (d / "pretrained_model" / "model.safetensors").write_bytes(b"w")
    if complete:
        (d / "pretrained_model" / "train_config.json").write_text("{}"); (d / "training_state").mkdir()
    return d


def test_prune_keeps_newest_complete_and_protects_last_and_in_progress(tmp_path):
    root = tmp_path / "checkpoints"
    for s in (1000, 2000, 3000, 4000): _ckpt(root, s)
    _ckpt(root, 5000, complete=False)                      # save in progress: never touched
    os.symlink("003000", root / "last")                    # whatever `last` points to survives
    assert prune_checkpoints.prune(str(root), 2) == ["001000", "002000"]
    assert sorted(p.name for p in root.iterdir()) == ["003000", "004000", "005000", "last"]
    assert prune_checkpoints.prune(str(root), 2) == []     # idempotent
    assert prune_checkpoints.prune(str(root), 0) == []     # 0 = keep everything
    assert prune_checkpoints.prune(str(tmp_path / "missing"), 3) == []
