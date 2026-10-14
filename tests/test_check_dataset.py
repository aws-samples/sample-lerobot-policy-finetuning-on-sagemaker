# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from tests.fixtures.make_mini_v3 import make_mini_v3  # noqa: E402

SCRIPT = REPO / "trainer" / "check_dataset.py"
# The one FAIL line only the codebase_version gate prints (the v2.1 fixture also fails on missing v3.0 files, and that
# message contains "v3.0" too, so a bare "v3.0" needle would not prove the gate exists).
V21_FAIL = "codebase_version is 'v2.1'; this pipeline needs v3.0 (convert with the LeRobot v3.0 tools first)"


def run(root, *args):
    r = subprocess.run([sys.executable, str(SCRIPT), str(root), *args], capture_output=True, text=True)
    return r.returncode, r.stderr, r.stdout


def test_default_fixture_passes_and_emits_split(tmp_path):
    root = make_mini_v3(tmp_path / "ds")
    rc, err, out = run(root, "--heldout", "0 6", "--emit-split")
    assert rc == 0, err
    split = json.loads(out)
    assert split["held"] == [0, 6] and len(split["train"]) == 10 and split["fps"] == 25 and split["chunk"] == 25
    assert "bimanual layout: 6 left-arm and 6 right-arm episodes" in err   # arm report for the 16-dim layout
    assert "EXACT copy" not in err                           # a real command signal must not trigger the lag warning
    assert "max is far above q99" not in err                 # the healthy baseline's gripper stats are sane
    # the gripper QUANTILES warning is its own variant (stats.json only; the parquet data is untouched)
    rc, err, _ = run(make_mini_v3(tmp_path / "outlier", variant="gripper_q99_outlier"))
    assert rc == 0 and "max is far above q99" in err         # gripper QUANTILES warning


def test_auto_split_and_small_dataset(tmp_path):
    root = make_mini_v3(tmp_path / "ds")
    rc, err, out = run(root, "--emit-split")
    assert rc == 0, err
    split = json.loads(out)
    assert split["held"] == [0] and len(split["train"]) == 11          # 10% of 12 -> 1 held-out (evenly spaced): episode 0
    few = make_mini_v3(tmp_path / "few", variant="few_episodes")
    rc, err, _ = run(few)
    assert rc == 1 and "force-small" in err
    rc, _, out = run(few, "--force-small", "--emit-split")
    assert rc == 0 and len(json.loads(out)["held"]) == 1


def test_action_equals_state_in_second_file_fails_unless_forced(tmp_path):
    root = make_mini_v3(tmp_path / "ds", variant="action_eq_state")
    # the copy sits ONLY in file-001: the old first-file/5000-row check would have passed it; now that file FAILs
    rc, err, _ = run(root)
    assert rc == 1 and "file-001.parquet: action == observation.state bit-exactly in this whole file" in err
    root2 = make_mini_v3(tmp_path / "eq", variant="action_eq_state", frames=40)
    # make it bit-exact everywhere to assert the FAIL path
    import pyarrow.parquet as pq
    for f in (root2 / "data/chunk-000").glob("*.parquet"):
        t = pq.read_table(f); t = t.set_column(t.schema.get_field_index("action"), "action", t["observation.state"]); pq.write_table(t, f)
    rc, err, _ = run(root2)
    assert rc == 1 and "bit-exactly" in err
    rc, err, _ = run(root2, "--force-action-eq-state")
    assert rc == 0 and "proceeding because" in err


def test_exact_lag_copy_warns(tmp_path):
    root = make_mini_v3(tmp_path / "ds", variant="exact_lag_copy")
    rc, err, _ = run(root)
    assert rc == 0 and "EXACT copy of state shifted by 3" in err


def test_names_dict_warn_or_fail(tmp_path):
    root = make_mini_v3(tmp_path / "ds", variant="names_dict")
    rc, err, _ = run(root)
    assert rc == 0 and "dict {'axes'" in err
    rc, err, _ = run(root, "--require-flat-names")
    assert rc == 1


def test_hard_failures(tmp_path):
    for variant, needle in [("v21_version", V21_FAIL), ("missing_tasks", "tasks.parquet"), ("missing_stats", "stats.json"),
                            ("image_dtype", "dtype 'image'"), ("names_short", "entries but action has")]:
        rc, err, _ = run(make_mini_v3(tmp_path / variant, variant=variant))
        assert rc == 1 and needle in err, (variant, err)


def test_codebase_version_alone_is_refused(tmp_path):
    """A complete, otherwise valid v3.0 layout whose info.json says codebase_version v2.1 is refused for that reason alone."""
    root = make_mini_v3(tmp_path / "ds")
    info_p = root / "meta" / "info.json"
    info = json.loads(info_p.read_text())
    info["codebase_version"] = "v2.1"
    info_p.write_text(json.dumps(info))
    rc, err, _ = run(root)
    assert rc == 1, err
    fails = [l for l in err.splitlines() if l.startswith("FAIL ")]  # the "FAIL: N fail(s)" summary line is not a finding
    assert fails == ["FAIL " + V21_FAIL], err                    # exactly one FAIL, and it is the version gate


def test_non_16_9_is_a_warning_and_q99_missing_is_a_warning(tmp_path):
    rc, err, _ = run(make_mini_v3(tmp_path / "a", variant="non_16_9"))
    assert rc == 0 and "is 4:3" in err
    rc, err, _ = run(make_mini_v3(tmp_path / "b", variant="no_q99"))
    assert rc == 0 and "lacks q01/q99" in err


def test_six_dim_dataset_is_scored_generically(tmp_path):
    rc, err, out = run(make_mini_v3(tmp_path / "ds", variant="six_dims"), "--heldout", "0", "--emit-split")
    assert rc == 0 and "left-arm" not in err and json.loads(out)["action_dim"] == 6


def test_bad_episode_ids(tmp_path):
    root = make_mini_v3(tmp_path / "ds")
    rc, err, _ = run(root, "--heldout", "0 99")
    assert rc == 1 and "out of range" in err


@pytest.mark.skipif(not (REPO / "datasets" / "svla_so101_pickplace" / "meta" / "info.json").exists(), reason="public dataset not downloaded")
def test_public_dataset_when_present():
    rc, err, out = run(REPO / "datasets" / "svla_so101_pickplace", "--heldout", "0 25", "--emit-split")
    assert rc == 0 and json.loads(out)["chunk"] == 30 and "is 4:3" in err


def test_missing_camera_stats_is_a_warning_naming_the_camera(tmp_path):
    root = tmp_path / "ds"; make_mini_v3(root)
    stats_path = root / "meta" / "stats.json"; stats = json.loads(stats_path.read_text()); stats.pop("observation.images.left_wrist"); stats_path.write_text(json.dumps(stats))
    r = subprocess.run([sys.executable, str(REPO / "trainer" / "check_dataset.py"), str(root)], capture_output=True, text=True)
    assert r.returncode == 0 and "WARN stats.json has no entry for camera(s) ['observation.images.left_wrist']" in r.stderr
