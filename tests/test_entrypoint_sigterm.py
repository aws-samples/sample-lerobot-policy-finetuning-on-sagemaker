# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""The entrypoint must survive SIGTERM: stop the trainer's process group, publish the newest checkpoint, exit 143.
Runs on the host with a fake trainer (no lerobot): ML=<tmp> CODE=<tmp>/code."""
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from tests.fixtures.make_mini_v3 import make_mini_v3  # noqa: E402

FAKE_TRAINER = """#!/usr/bin/env bash
# fake train_policy.sh: writes one checkpoint, then sleeps like a long run (children in the same process group)
set -euo pipefail
mkdir -p "$OUT_DIR/train/checkpoints/000010/pretrained_model" "$OUT_DIR/eval"
echo weights > "$OUT_DIR/train/checkpoints/000010/pretrained_model/model.safetensors"
echo "{}" > "$OUT_DIR/train/checkpoints/000010/pretrained_model/train_config.json"
echo '{"train":[1],"held":[0]}' > "$OUT_DIR/split.json"
echo "INFO 2026-09-10 13:13:59 ot_train.py:641 step:10 smpl:20 ep:0 epch:0.00 loss:1 grdn:1 lr:1.0e-05 updt_s:0.9 data_s:0.7 smp/s:1" > "$OUT_DIR/train.log"
sleep 120 &
echo $! > "$OUT_DIR/sleep.pid"
wait
"""


def _env(tmp_path):
    ml = tmp_path / "ml"; code = ml / "code"
    (ml / "input" / "config").mkdir(parents=True)
    make_mini_v3(ml / "input" / "data" / "train", height=360, width=640)  # already at TARGET_H: no downscale step
    code.mkdir()
    for f in ("publish.py", "summary.py", "licence.py", "render_flags.py", "smoke_checks.py"):
        shutil.copy(REPO / "trainer" / f, code / f)
    shutil.copytree(REPO / "trainer" / "policies", code / "policies")
    (code / "train_policy.sh").write_text(FAKE_TRAINER); (code / "train_policy.sh").chmod(0o755)
    (ml / "input" / "config" / "hyperparameters.json").write_text(json.dumps({"steps": "20", "device": "cpu"}))
    return ml, code


@pytest.mark.skipif(shutil.which("setsid") is None, reason="needs setsid")
def test_sigterm_publishes_newest_checkpoint_and_exits_143(tmp_path):
    ml, code = _env(tmp_path)
    env = {**os.environ, "ML": str(ml), "CODE": str(code), "PYTHON": sys.executable}
    p = subprocess.Popen(["bash", str(REPO / "trainer" / "sagemaker" / "train_entrypoint.sh"), "train"], env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    deadline = time.time() + 30
    while time.time() < deadline and not (ml / "checkpoints" / "act" / "train.log").exists():
        time.sleep(0.2)
    assert (ml / "checkpoints" / "act" / "train.log").exists(), "fake trainer never started"
    time.sleep(0.5)
    p.send_signal(signal.SIGTERM)
    out, _ = p.communicate(timeout=90)
    assert p.returncode == 143, out
    assert (ml / "model" / "pretrained_model" / "model.safetensors").read_text().strip() == "weights"
    s = json.loads((ml / "model" / "SUMMARY.json").read_text())
    assert s["status"] == "interrupted" and s["checkpoint_step"] == 10
    assert "interrupted" in (ml / "output" / "failure").read_text()
    # no orphan: the sleep child of the trainer's process group is gone
    time.sleep(1)
    pid = int((ml / "checkpoints" / "act" / "sleep.pid").read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
