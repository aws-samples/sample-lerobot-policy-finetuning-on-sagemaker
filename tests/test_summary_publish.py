# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "trainer"))
import summary  # noqa: E402

LOG = """INFO 2026-09-10 13:13:41 ot_train.py:597 Start offline training
Training:  25%|██▌       | 50/200 [02:38<00:24,  6.13step/s]INFO 2026-09-10 13:13:49 ot_train.py:641 step:50 smpl:400 ep:0 epch:0.00 loss:1.0 grdn:4.5 lr:5.1e-06 updt_s:0.423 data_s:2.745 smp/s:5
INFO 2026-09-10 13:13:59 ot_train.py:641 step:200 smpl:1.6K ep:0 epch:0.00 loss:46.3 grdn:887.9 lr:1.0e-05 updt_s:0.098 data_s:0.780 smp/s:81 l1_loss:0.998 kld_loss:4.5
INFO 2026-09-10 13:14:19 ot_train.py:641 step:400 smpl:3.2K ep:0 epch:0.01 loss:30.0 grdn:500.0 lr:1.0e-05 updt_s:0.097 data_s:0.001 smp/s:81 l1_loss:0.9 kld_loss:4.0
INFO 2026-09-10 13:14:39 ot_train.py:641 step:600 smpl:4.8K ep:0 epch:0.01 loss:20.0 grdn:400.0 lr:1.0e-05 updt_s:0.097 data_s:0.001 smp/s:82 l1_loss:0.8 kld_loss:3.0
INFO 2026-09-10 13:14:59 ot_train.py:641 step:800 smpl:6.4K ep:0 epch:0.02 loss:15.0 grdn:300.0 lr:1.0e-05 updt_s:0.097 data_s:0.001 smp/s:82 l1_loss:0.7 kld_loss:2.0
INFO 2026-09-10 13:15:19 ot_train.py:641 step:1K smpl:8.0K ep:0 epch:0.02 loss:14.0 grdn:290.0 lr:1.0e-05 updt_s:0.097 data_s:0.001 smp/s:82 l1_loss:0.7 kld_loss:2.0
"""
EVAL = {"episodes": [0, 13], "chunk": 24, "stride": 1, "layout": "bimanual_16", "n_samples": 1,  # fixture label; eval_openloop.py writes openarm_bimanual_16 or generic_<D> and summary.py passes the value through
        "mae": {"all": {"all_joints": {"model": 0.04271, "hold": 0.06367}, "all_grippers": {"model": 0.0016, "hold": 0.00393}}},
        "per_dim": {"all": {f"d{i}": {"model": 0.03 if i < 14 else 0.09, "hold": 0.06} for i in range(16)}},
        "gripper_timing": {"hits": 16, "total": 16}}


def test_step_rate_uses_timestamps_and_the_8k_format(tmp_path):
    (tmp_path / "train.log").write_text(LOG)
    rate, last = summary.step_rate_from_log(str(tmp_path / "train.log"))
    assert abs(rate - 10.0) < 0.01 and last["smp_per_s"] == 82.0          # 200 steps per 20 s; 'step:1K' parsed as 1000
    assert summary.step_rate_from_log(str(tmp_path / "missing.log")) == (None, None)


def test_eval_summary_and_beats_hold(tmp_path):
    p = tmp_path / "eval_last.json"; p.write_text(json.dumps(EVAL))
    e = summary.eval_summary(str(p))
    assert e["joints_mae"] == 0.04271 and e["gripper_timing"] == "16/16" and e["beats_hold"] == "14/16" and e["chunk"] == 24


def _run_tree(tmp_path):
    out = tmp_path / "out"; model = tmp_path / "model"; data = tmp_path / "data"
    for step in ("000010", "000020"):
        (out / "train" / "checkpoints" / step / "pretrained_model").mkdir(parents=True)
        (out / "train" / "checkpoints" / step / "pretrained_model" / "model.safetensors").write_bytes(b"w" * 10 + step.encode())
        (out / "train" / "checkpoints" / step / "pretrained_model" / "train_config.json").write_text("{}")
        # SageMaker's checkpoint sync agent leaves a zero-byte marker next to every mirrored file
        (out / "train" / "checkpoints" / step / "pretrained_model" / "model.safetensors.sagemaker-uploaded").write_bytes(b"")
    # an incomplete (still being written) newer checkpoint must be ignored
    (out / "train" / "checkpoints" / "000030" / "pretrained_model").mkdir(parents=True)
    (out / "train" / "checkpoints" / "000030" / "pretrained_model" / "model.safetensors").write_bytes(b"partial")
    (out / "train" / "checkpoints" / "last").mkdir()      # a broken `last` (real dir after an S3 round trip) must be ignored
    (out / "eval").mkdir(); (out / "eval" / "eval_last.json").write_text(json.dumps(EVAL))
    (out / "eval" / "eval_last.json.sagemaker-uploaded").write_bytes(b"")
    (out / "train.log").write_text(LOG); (out / "split.json").write_text(json.dumps({"train": [1], "held": [0]}))
    (out / "exitcode").write_text("0\n")
    return out, model, data


def test_publish_copies_newest_checkpoint_and_writes_summary(tmp_path):
    out, model, data = _run_tree(tmp_path)
    r = subprocess.run([sys.executable, str(REPO / "trainer" / "publish.py"), "--out-dir", str(out), "--model-dir", str(model),
                        "--output-data", str(data), "--policy", "act"], capture_output=True, text=True,
                       env={**os.environ, "IMAGE_DIGEST": "sha256:abc", "DATASET_FINGERPRINT": "ff2de01a"})
    assert r.returncode == 0, r.stderr
    assert (model / "pretrained_model" / "model.safetensors").read_bytes().endswith(b"000020")
    s = json.loads((model / "SUMMARY.json").read_text())
    assert s["checkpoint_step"] == 20 and s["status"] == "completed" and s["policy"] == "act"
    assert s["evals"]["eval_last.json"]["beats_hold"] == "14/16" and abs(s["step_rate"] - 10.0) < 0.01
    assert s["image_digest"] == "sha256:abc" and s["licence_status"] == "accepted" and s["train_exitcode"] == 0
    assert s["dataset_fingerprint"] == "ff2de01a" and s["deliverable"] is True          # act.yaml: deliverable true, stamped without --deliverable
    assert (out / "SUMMARY.json").is_file() and (data / "train.log").is_file() and (data / "split.json").is_file()
    # the sync agent's markers never reach model.tar.gz
    assert not list(model.rglob("*.sagemaker-uploaded")) and (model / "eval" / "eval_last.json").is_file()
    notice = (model / "LICENCE_NOTICE.md").read_text()
    assert "contains no model weights" in notice
    assert "Dataset fingerprint: ff2de01a  |  Image: sha256:abc" in notice and "n/a" not in notice
    assert "Run: status completed  |  licence gate: accepted" in notice and "registry status" in notice


def test_publish_without_checkpoint_exits_2(tmp_path):
    (tmp_path / "out").mkdir()
    r = subprocess.run([sys.executable, str(REPO / "trainer" / "publish.py"), "--out-dir", str(tmp_path / "out"), "--model-dir",
                        str(tmp_path / "m"), "--output-data", str(tmp_path / "d"), "--status", "interrupted"], capture_output=True, text=True)
    assert r.returncode == 2 and json.loads((tmp_path / "m" / "SUMMARY.json").read_text())["status"] == "interrupted"


def test_rerun_keeps_publish_fields(tmp_path):
    out, model, data = _run_tree(tmp_path)
    subprocess.run([sys.executable, str(REPO / "trainer" / "publish.py"), "--out-dir", str(out), "--model-dir", str(model),
                    "--output-data", str(data), "--policy", "act"], check=True, capture_output=True, env={**os.environ, "IMAGE_DIGEST": "sha256:abc"})
    (out / "reload_check.json").write_text(json.dumps({"ok": True}))
    subprocess.run([sys.executable, str(REPO / "trainer" / "summary.py"), "--out-dir", str(out), "--model-dir", str(model),
                    "--policy", "act", "--status", "completed", "--write"], check=True, capture_output=True, env={k: v for k, v in os.environ.items() if k != "IMAGE_DIGEST"})
    s = json.loads((model / "SUMMARY.json").read_text())
    assert s["licence_status"] == "accepted" and s["image_digest"] == "sha256:abc" and s["reload_check"] == {"ok": True}
    assert s["deliverable"] is True


def test_publish_rerun_without_env_keeps_provenance_in_the_notice(tmp_path):
    """The SIGTERM path may publish again after the normal one: dataset and image come from the previous SUMMARY.json."""
    out, model, data = _run_tree(tmp_path)
    env = {k: v for k, v in os.environ.items() if k not in ("IMAGE_DIGEST", "DATASET_FINGERPRINT")}
    subprocess.run([sys.executable, str(REPO / "trainer" / "publish.py"), "--out-dir", str(out), "--model-dir", str(model),
                    "--output-data", str(data), "--policy", "act", "--dataset-fingerprint", "ff2de01a", "--image-digest", "sha256:abc"],
                   check=True, capture_output=True, env=env)
    subprocess.run([sys.executable, str(REPO / "trainer" / "publish.py"), "--out-dir", str(out), "--model-dir", str(model),
                    "--output-data", str(data), "--policy", "act", "--status", "interrupted"], check=True, capture_output=True, env=env)
    notice = (model / "LICENCE_NOTICE.md").read_text()
    assert "Dataset fingerprint: ff2de01a  |  Image: sha256:abc" in notice and "Run: status interrupted" in notice
    s = json.loads((model / "SUMMARY.json").read_text())
    assert s["status"] == "interrupted" and s["dataset_fingerprint"] == "ff2de01a" and s["image_digest"] == "sha256:abc"


def test_deliverable_comes_from_the_registry():
    assert summary.registry_deliverable("act") is True
    assert summary.registry_deliverable("smolvla") is False       # class clean but the base VLM is used frozen; registry says no
    assert summary.registry_deliverable("no_such_policy") is None


def test_notice_carries_the_run_provenance():
    sys.path.insert(0, str(REPO / "trainer"))
    import licence
    from render_flags import load_policy
    spec = load_policy("act")
    text = licence.notice_text(spec, dataset_name="ff2de01a", image_digest="sha256:abc", run_status="completed", licence_status="accepted")
    assert "Dataset fingerprint: ff2de01a  |  Image: sha256:abc" in text and "Run: status completed  |  licence gate: accepted" in text
    plain = licence.notice_text(spec)
    assert "Run:" not in plain and "Dataset fingerprint: n/a  |  Image: n/a" in plain     # the bare --notice CLI stays honest
    r = subprocess.run([sys.executable, str(REPO / "trainer" / "licence.py"), "act", "--notice", "--dataset", "ff2de01a", "--image", "sha256:abc",
                        "--run-status", "interrupted", "--licence-status", "accepted"], capture_output=True, text=True)
    assert r.returncode == 0 and "Run: status interrupted  |  licence gate: accepted" in r.stdout and "n/a" not in r.stdout
