# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""publish.py's optional adapter merge: the decision (registry flag, MERGE_ADAPTER=1, adapter vs full checkpoint), the SUMMARY.json
record (merged_checkpoint / merge_error), a failing merge never changing publish's exit code, and reload_check's preference
for the merged weights. Host-only: the merge itself (lerobot + peft) is replaced by a fake."""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "trainer"))
import publish  # noqa: E402
import reload_check  # noqa: E402

SPEC_ON = {"name": "fake_lora", "lerobot_type": "fastwam", "base_checkpoint": "lerobot/fastwam_base", "publish": {"merge_adapter": True}}
SPEC_OFF = {"name": "fake_lora", "lerobot_type": "fastwam", "base_checkpoint": "lerobot/fastwam_base"}


def _tree(tmp_path, adapter=True):
    out = tmp_path / "out"; model = tmp_path / "model"; data = tmp_path / "data"
    pm = out / "train" / "checkpoints" / "000200" / "pretrained_model"
    pm.mkdir(parents=True)
    if adapter:
        (pm / "adapter_model.safetensors").write_bytes(b"lora" * 8)
        (pm / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA", "r": 16}))
    else:
        (pm / "model.safetensors").write_bytes(b"full" * 8)
    (pm / "config.json").write_text(json.dumps({"type": "fastwam", "device": "cuda", "use_peft": adapter}))
    (pm / "train_config.json").write_text("{}")
    (pm / "policy_preprocessor.json").write_text("{}")
    (pm / "policy_preprocessor_step_0_normalizer_processor.safetensors").write_bytes(b"stats")
    (out / "exitcode").write_text("0\n")
    return out, model, data


def _fake_merge(calls):
    def fake(adapter_dir, merged_dir, lerobot_type, base_checkpoint):
        calls.append((adapter_dir, merged_dir, lerobot_type, base_checkpoint))
        os.makedirs(merged_dir)
        Path(merged_dir, "model.safetensors").write_bytes(b"merged-weights")
        Path(merged_dir, "config.json").write_text(json.dumps({"device": "cuda", "use_peft": False}))
        return {"path": publish.MERGED_DIRNAME + "/", "model_sha256": publish.sha256_file(os.path.join(merged_dir, "model.safetensors")),
                "base_checkpoint": base_checkpoint, "sizes_bytes": {"model.safetensors": 14}}
    return fake


def _publish(out, model, data, status="completed"):
    return publish.main(["--out-dir", str(out), "--model-dir", str(model), "--output-data", str(data), "--policy", "act", "--status", status])


def test_merge_requested_reads_env_then_registry(monkeypatch):
    assert publish.merge_requested("act", {}) is False                      # act.yaml has no publish block
    assert publish.merge_requested("act", {"MERGE_ADAPTER": "1"}) is True
    assert publish.merge_requested("act", {"MERGE_ADAPTER": "0"}) is False
    assert publish.registry_spec("no_such_policy") == {} and publish.registry_spec("act")["lerobot_type"] == "act"
    monkeypatch.setattr(publish, "registry_spec", lambda name: SPEC_ON)
    assert publish.merge_requested("fake_lora", {}) is True
    monkeypatch.setattr(publish, "registry_spec", lambda name: SPEC_OFF)
    assert publish.merge_requested("fake_lora", {}) is False


def test_is_adapter_checkpoint(tmp_path):
    out, _, _ = _tree(tmp_path, adapter=True)
    assert publish.is_adapter_checkpoint(str(out / "train" / "checkpoints" / "000200" / "pretrained_model"))
    out2, _, _ = _tree(tmp_path / "b", adapter=False)
    assert not publish.is_adapter_checkpoint(str(out2 / "train" / "checkpoints" / "000200" / "pretrained_model"))


def test_registry_flag_merges_and_summary_records_the_merged_checkpoint(tmp_path, monkeypatch):
    out, model, data = _tree(tmp_path)
    calls = []
    monkeypatch.delenv("MERGE_ADAPTER", raising=False)
    monkeypatch.setattr(publish, "registry_spec", lambda name: SPEC_ON)
    monkeypatch.setattr(publish, "merge_adapter", _fake_merge(calls))
    assert _publish(out, model, data) == 0
    assert calls == [(str(model / "pretrained_model"), str(model / publish.MERGED_DIRNAME), "fastwam", "lerobot/fastwam_base")]
    s = json.loads((model / "SUMMARY.json").read_text())
    assert s["checkpoint"] == "pretrained_model/" and s["model_sha256"] == hashlib.sha256(b"lora" * 8).hexdigest()   # the adapter, as before
    assert s["merged_checkpoint"]["path"] == "pretrained_model_merged/"
    assert s["merged_checkpoint"]["model_sha256"] == hashlib.sha256(b"merged-weights").hexdigest()
    assert s["merged_checkpoint"]["base_checkpoint"] == "lerobot/fastwam_base" and "merge_error" not in s
    assert (out / "merge.json").is_file() and (out / "SUMMARY.json").is_file()
    # the adapter checkpoint itself is untouched next to the merged weights
    assert (model / "pretrained_model" / "adapter_model.safetensors").is_file() and not (model / "pretrained_model" / "model.safetensors").exists()
    # reload_check prefers the merged full weights and drops the PEFT base
    assert reload_check.reload_source(str(model), "lerobot/fastwam_base") == (str(model / publish.MERGED_DIRNAME), None, "pretrained_model_merged/")


def test_env_var_merges_without_registry_flag(tmp_path, monkeypatch):
    out, model, data = _tree(tmp_path)
    calls = []
    monkeypatch.setenv("MERGE_ADAPTER", "1")
    monkeypatch.setattr(publish, "registry_spec", lambda name: SPEC_OFF)
    monkeypatch.setattr(publish, "merge_adapter", _fake_merge(calls))
    assert _publish(out, model, data) == 0 and len(calls) == 1
    assert "merged_checkpoint" in json.loads((model / "SUMMARY.json").read_text())


def test_no_flag_means_no_merge(tmp_path, monkeypatch):
    out, model, data = _tree(tmp_path)
    calls = []
    monkeypatch.delenv("MERGE_ADAPTER", raising=False)
    monkeypatch.setattr(publish, "registry_spec", lambda name: SPEC_OFF)
    monkeypatch.setattr(publish, "merge_adapter", _fake_merge(calls))
    assert _publish(out, model, data) == 0 and calls == []
    s = json.loads((model / "SUMMARY.json").read_text())
    assert "merged_checkpoint" not in s and "merge_error" not in s and not (model / publish.MERGED_DIRNAME).exists()
    assert reload_check.reload_source(str(model), "lerobot/fastwam_base") == (str(model / "pretrained_model"), "lerobot/fastwam_base", "pretrained_model/")


def test_full_checkpoint_is_never_merged(tmp_path, monkeypatch):
    out, model, data = _tree(tmp_path, adapter=False)
    calls = []
    monkeypatch.setenv("MERGE_ADAPTER", "1")
    monkeypatch.setattr(publish, "registry_spec", lambda name: SPEC_ON)
    monkeypatch.setattr(publish, "merge_adapter", _fake_merge(calls))
    assert _publish(out, model, data) == 0 and calls == []
    assert "merged_checkpoint" not in json.loads((model / "SUMMARY.json").read_text())


def test_interrupted_publish_skips_the_merge(tmp_path, monkeypatch):
    out, model, data = _tree(tmp_path)
    calls = []
    monkeypatch.setenv("MERGE_ADAPTER", "1")
    monkeypatch.setattr(publish, "merge_adapter", _fake_merge(calls))
    assert _publish(out, model, data, status="interrupted") == 0 and calls == []


def test_merge_failure_is_recorded_and_does_not_change_the_exit_code(tmp_path, monkeypatch, capsys):
    out, model, data = _tree(tmp_path)
    monkeypatch.setenv("MERGE_ADAPTER", "1")
    monkeypatch.setattr(publish, "registry_spec", lambda name: SPEC_ON)

    def boom(adapter_dir, merged_dir, lerobot_type, base_checkpoint):
        os.makedirs(merged_dir + ".tmp"); Path(merged_dir + ".tmp", "model.safetensors").write_bytes(b"half")
        raise RuntimeError("Expected all tensors to be on the same device")
    monkeypatch.setattr(publish, "merge_adapter", boom)
    assert _publish(out, model, data) == 0
    assert "publish: adapter merge failed: RuntimeError: Expected all tensors" in capsys.readouterr().err
    s = json.loads((model / "SUMMARY.json").read_text())
    assert s["merge_error"].startswith("RuntimeError: Expected all tensors") and "merged_checkpoint" not in s
    assert not (model / publish.MERGED_DIRNAME).exists() and not (model / (publish.MERGED_DIRNAME + ".tmp")).exists()   # nothing half-written ships
    assert reload_check.reload_source(str(model), "lerobot/fastwam_base")[1] == "lerobot/fastwam_base"   # falls back to the adapter path
    # a missing base_checkpoint (load_policy calls SystemExit) is a recorded error too
    monkeypatch.setattr(publish, "merge_adapter", lambda *a: (_ for _ in ()).throw(SystemExit("pass --peft-base")))
    assert _publish(out, model, data) == 0
    assert json.loads((model / "SUMMARY.json").read_text())["merge_error"] == "SystemExit: pass --peft-base"
    assert not (model / (publish.MERGED_DIRNAME + ".tmp")).exists()


def test_summary_rerun_keeps_the_merge_record_and_a_new_publish_replaces_it(tmp_path, monkeypatch):
    out, model, data = _tree(tmp_path)
    monkeypatch.setenv("MERGE_ADAPTER", "1")
    monkeypatch.setattr(publish, "merge_adapter", _fake_merge([]))
    assert _publish(out, model, data) == 0
    # the entrypoint re-runs summary.py after reload_check: the merge record must survive
    (out / "reload_check.json").write_text(json.dumps({"ok": True, "checkpoint": "pretrained_model_merged/"}))
    subprocess.run([sys.executable, str(REPO / "trainer" / "summary.py"), "--out-dir", str(out), "--model-dir", str(model),
                    "--policy", "act", "--status", "completed", "--write"], check=True, capture_output=True)
    s = json.loads((model / "SUMMARY.json").read_text())
    assert s["merged_checkpoint"]["path"] == "pretrained_model_merged/" and s["reload_check"]["checkpoint"] == "pretrained_model_merged/"
    # a later publish without the flag removes the stale merged weights and the record
    monkeypatch.setenv("MERGE_ADAPTER", "0")
    assert _publish(out, model, data) == 0
    assert not (model / publish.MERGED_DIRNAME).exists() and not (out / "merge.json").exists()
    assert "merged_checkpoint" not in json.loads((model / "SUMMARY.json").read_text())


def test_merge_adapter_refuses_without_a_base(tmp_path):
    with pytest.raises(RuntimeError, match="base_checkpoint"):
        publish.merge_adapter(str(tmp_path), str(tmp_path / "merged"), "fastwam", None)


def test_merge_scratch_dir_is_on_the_model_directory_filesystem(tmp_path):
    """/opt/ml/model is its own mount on SageMaker (and a bind mount on an EC2 host): a scratch directory outside it made
    the final os.replace fail with EXDEV on the 2026-09-21 p5 smoke. The scratch must be a hidden sibling inside it."""
    import os
    model_dir = tmp_path / "model"
    merged_dir = model_dir / publish.MERGED_DIRNAME
    tmp = publish._scratch_dir(str(merged_dir))
    assert os.path.dirname(tmp) == str(model_dir) and os.path.basename(tmp).startswith("." + publish.MERGED_DIRNAME + ".")
    assert os.path.isdir(tmp) and not os.listdir(tmp)
    os.replace(tmp, str(merged_dir))   # same filesystem by construction
    assert merged_dir.is_dir()


def test_stale_scratch_dirs_are_removed_before_a_merge(tmp_path, monkeypatch):
    model_dir = tmp_path / "model"; adapter = model_dir / "pretrained_model"; adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text("{}"); (adapter / "adapter_model.safetensors").write_bytes(b"x")
    stale = model_dir / ("." + publish.MERGED_DIRNAME + ".leftover"); stale.mkdir(); (stale / "model.safetensors").write_bytes(b"half")
    monkeypatch.setenv("MERGE_ADAPTER", "1")
    monkeypatch.setattr(publish, "registry_spec", lambda name: {"lerobot_type": "fastwam", "base_checkpoint": "b", "publish": {"merge_adapter": True}})
    monkeypatch.setattr(publish, "merge_adapter", lambda *a, **k: {"path": publish.MERGED_DIRNAME + "/", "model_sha256": "0" * 64, "base_checkpoint": "b", "sizes_bytes": {}})
    out = publish.maybe_merge("fastwam_lora", str(model_dir), str(tmp_path / "out"), "completed")
    assert "merged_checkpoint" in out and not stale.exists()
