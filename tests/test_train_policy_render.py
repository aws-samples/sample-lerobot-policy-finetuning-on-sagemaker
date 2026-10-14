# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""train_policy.sh with POLICY=act must hand lerobot-train exactly the validated argv (RENDER_ONLY mode, no lerobot needed)."""
import os
import pathlib
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from tests.fixtures.make_mini_v3 import make_mini_v3  # noqa: E402


def _render(tmp_path, **env):
    import uuid
    tmp_path = tmp_path / uuid.uuid4().hex[:8]
    tmp_path.mkdir()
    ds = make_mini_v3(tmp_path / "ds", height=360, width=640)  # train_policy.sh expects a prepare_dataset_360.sh output
    e = {**os.environ, "RENDER_ONLY": "1", "LEROBOT_TRAIN": "/bin/true", "DS_ROOT": str(ds), "OUT_DIR": str(tmp_path / "out"),
         "HELDOUT": "0 6", "NUM_WORKERS": "12", "VIDEO_BACKEND": "pyav", **env}
    r = subprocess.run(["bash", str(REPO / "trainer" / "train_policy.sh")], env=e, capture_output=True, text=True)
    return r


def test_act_render_equals_validated_argv(tmp_path):
    r = _render(tmp_path)
    assert r.returncode == 0, r.stderr + r.stdout
    flags = [l for l in r.stdout.splitlines() if l.startswith("--")]
    root = flags[1].split("=", 1)[1]
    assert flags == [
        f"--dataset.repo_id=local/ds", f"--dataset.root={root}", "--dataset.episodes=[1,2,3,4,5,7,8,9,10,11]",
        "--dataset.video_backend=pyav", "--policy.type=act", "--policy.chunk_size=24", "--policy.n_action_steps=24",
        "--policy.device=cuda", "--policy.push_to_hub=false", "--batch_size=8", "--steps=100000", "--save_freq=10000",
        "--log_freq=200", "--num_workers=12", "--seed=1000", f"--output_dir={root[:-2]}out/train", "--job_name=out",
        "--wandb.enable=false"]
    assert (pathlib.Path(root[:-2]) / "out" / "split.json").is_file()


def test_overrides_are_appended_and_forbidden_ones_refused(tmp_path):
    r = _render(tmp_path, OVERRIDES="--policy.kl_weight=20")
    assert r.returncode == 0 and r.stdout.splitlines()[-1] == "--policy.kl_weight=20"
    r = _render(tmp_path, OVERRIDES="--policy.dtype=bfloat16")
    assert r.returncode != 0 and "forbidden" in (r.stderr + r.stdout)


def test_unknown_policy_and_licence_gate(tmp_path):
    r = _render(tmp_path, POLICY="nope")
    assert r.returncode != 0 and "no registry entry" in (r.stderr + r.stdout)
