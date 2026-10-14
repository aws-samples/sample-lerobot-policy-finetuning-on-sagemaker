# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "trainer"))
import render_flags  # noqa: E402
from pipeline import compat, run  # noqa: E402
from tests.fixtures.make_mini_v3 import make_mini_v3  # noqa: E402

ALL = {p.stem: render_flags.load_policy(p.stem) for p in sorted(render_flags.POLICIES_DIR.glob("*.yaml"))}


def test_facts_and_rules_on_the_fixture(tmp_path):
    root = tmp_path / "ds"; make_mini_v3(root)                   # 16-dim bimanual, 3 cameras, 1 task, quantiles, no reward
    f = compat.dataset_facts(root)
    assert f["action_dim"] == 16 and len(f["cameras"]) == 3 and f["n_tasks"] == 1 and f["has_quantiles"] is True and not f["has_reward"]
    assert compat.evaluate(f, ALL["act"])["ok"]
    assert compat.evaluate(f, ALL["vqbet"]) == {"ok": True, "blockers": [], "notes": ["trains on one camera (observation.images.ego by default; policies.<name>.camera picks another)"]}
    r = compat.evaluate(f, ALL["smolvla"]); assert r["ok"] and "one task string" in r["notes"][0]
    r = compat.evaluate(f, ALL["tdmpc"]); assert not r["ok"] and any("reward" in b for b in r["blockers"]) and any("square" in b for b in r["blockers"])
    assert not compat.evaluate(f, ALL["gaussian_actor"])["ok"]
    wide = {**f, "action_dim": 40}
    assert any("pads to at most 32" in b for b in compat.evaluate(wide, ALL["pi0"])["blockers"])


def test_names_dict_tasks_and_quantile_rules(tmp_path):
    root = tmp_path / "d"; make_mini_v3(root, "names_dict", "no_q99")
    f = compat.dataset_facts(root)
    assert f["names_flat"] is False and f["has_quantiles"] is False
    assert not compat.evaluate(f, ALL["groot"])["ok"]                     # requires_flat_action_names
    r = compat.evaluate(f, ALL["pi05"]); assert any("q01/q99" in n for n in r["notes"])
    no_tasks = {**f, "n_tasks": 0}
    assert not compat.evaluate(no_tasks, ALL["pi0"])["ok"] and compat.evaluate(no_tasks, ALL["diffusion"])["ok"]
    lines = compat.report(f, ALL, ["act", "tdmpc"])
    assert lines[0].startswith("compatibility of dataset d (16-dim actions, 3 camera(s), 1 task string(s)):")
    assert any(l.startswith("  runnable now (") and "act" in l for l in lines)
    assert any("not compatible: tdmpc:" in l for l in lines)
    assert compat.blocked_enabled(f, ALL, ["act", "tdmpc"]).keys() == {"tdmpc"}


def test_compat_command_prints_the_table_without_aws(tmp_path, monkeypatch, capsys):
    (tmp_path / "datasets").mkdir(); make_mini_v3(tmp_path / "datasets" / "mini")
    monkeypatch.setattr(run, "REPO", tmp_path)
    cfg = json.loads((REPO / "project_config.json").read_text())
    cfg["dataset"] = {"name": "mini", "source": "local", "hf_repo_id": None, "hf_revision": None, "s3_uri": None, "downscale_height": 360}
    cfg["policies"] = {"act": {"enabled": True}, "tdmpc": {"enabled": True}}
    (tmp_path / "cfg.json").write_text(json.dumps(cfg))
    assert run.main(["--compat", "--config", str(tmp_path / "cfg.json"), "--allow-stubs"]) == 3
    out = capsys.readouterr().out
    assert "of 17 registry policies can fine-tune on it" in out and "out of scope (3, not offered by this repository): gaussian_actor, vla_jepa, wall_x" in out and "ERROR: enabled policy tdmpc" in out
    cfg["policies"] = {"act": {"enabled": True}}
    (tmp_path / "cfg.json").write_text(json.dumps(cfg))
    assert run.main(["--compat", "--config", str(tmp_path / "cfg.json")]) == 0


def test_fastwam_camera_count_must_divide_448(tmp_path):
    root = tmp_path / "ds"; make_mini_v3(root)                        # 3 cameras
    f = compat.dataset_facts(root)
    r = compat.evaluate(f, ALL["fastwam"]); assert not r["ok"] and "divide" in r["blockers"][0] and "[1, 2]" in r["blockers"][0]
    assert compat.evaluate(f, ALL["fastwam"], ["observation.images.ego", "observation.images.right_wrist"])["ok"]
    assert compat.evaluate({**f, "cameras": {"a": (1, 1), "b": (1, 1)}}, ALL["fastwam"])["ok"]
    assert compat.blocked_enabled(f, ALL, ["fastwam"], {"fastwam": {"cameras": ["observation.images.ego", "observation.images.right_wrist"]}}) == {}
    assert "fastwam" in compat.blocked_enabled(f, ALL, ["fastwam"], {"fastwam": {}})


def test_capped_policies_block_wider_datasets_before_submission():
    """xvla pads to 20 action dims (eo1/molmoact2 to 32, evo1 to 24, lingbot_va to 30, groot to 132); a wider dataset is a
    plan-time blocker, not a failure inside the paid job."""
    from pipeline import compat
    from trainer import render_flags
    specs = render_flags.load_all() if hasattr(render_flags, "load_all") else {n: render_flags.load_policy(n) for n in ("xvla", "eo1", "evo1", "molmoact2", "lingbot_va", "groot")}
    for name, cap in (("xvla", 20), ("eo1", 32), ("evo1", 24), ("molmoact2", 32), ("lingbot_va", 30), ("groot", 132)):
        spec = specs[name] if isinstance(specs, dict) else render_flags.load_policy(name)
        assert spec.get("max_action_dim") == cap, name
