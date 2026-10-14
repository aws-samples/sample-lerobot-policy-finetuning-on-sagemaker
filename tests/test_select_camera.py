# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "trainer"))
import select_camera  # noqa: E402
from tests.fixtures.make_mini_v3 import make_mini_v3  # noqa: E402

SCRIPT = REPO / "trainer" / "select_camera.py"


def _cams(root):
    return select_camera.image_keys(json.loads((Path(root) / "meta" / "info.json").read_text()))


def test_default_keeps_first_sorted_camera_and_leaves_source_untouched(tmp_path):
    root = tmp_path / "ds"; make_mini_v3(root)
    before = sorted(p.relative_to(root).as_posix() for p in root.rglob("*"))
    out = select_camera.derive(str(root), out=str(tmp_path / "view"))
    assert out == str(tmp_path / "view")
    assert _cams(out) == ["observation.images.ego"]
    assert _cams(root) == ["observation.images.ego", "observation.images.left_wrist", "observation.images.right_wrist"]
    assert sorted(p.relative_to(root).as_posix() for p in root.rglob("*")) == before
    view = Path(out)
    assert (view / "data").is_symlink() and (view / "data" / "chunk-000" / "file-000.parquet").is_file()
    assert (view / "videos" / "observation.images.ego").is_symlink()
    assert not (view / "videos" / "observation.images.left_wrist").exists()
    stats = json.loads((view / "meta" / "stats.json").read_text())
    assert "observation.images.ego" in stats and "observation.images.left_wrist" not in stats and "action" in stats
    marker = json.loads((view / "meta" / select_camera.MARKER).read_text())
    assert marker["kept"] == ["observation.images.ego"] and len(marker["dropped"]) == 2


def test_keep_picks_the_named_camera_and_rerun_overwrites_only_its_own_output(tmp_path):
    root = tmp_path / "ds"; make_mini_v3(root)
    out = tmp_path / "view"
    r = subprocess.run([sys.executable, str(SCRIPT), str(root), "--keep", "observation.images.right_wrist", "--out", str(out)],
                       capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.strip() == str(out) and "ignoring" in r.stderr
    two = tmp_path / "two"                                              # a two-camera subset (FastWAM on a 3-camera robot)
    assert select_camera.derive(str(root), keep=["observation.images.ego", "observation.images.right_wrist"], out=str(two)) == str(two)
    assert _cams(two) == ["observation.images.ego", "observation.images.right_wrist"] and (two / "videos" / "observation.images.right_wrist").is_symlink()
    assert select_camera.derive(str(root), keep=list(_cams(root)), out=str(tmp_path / "all")) == str(root)   # every camera kept: no view
    assert _cams(out) == ["observation.images.right_wrist"]
    assert select_camera.derive(str(root), keep="observation.images.ego", out=str(out)) == str(out)   # marker present: replaced
    assert _cams(out) == ["observation.images.ego"]
    foreign = tmp_path / "mine"; (foreign / "meta").mkdir(parents=True)
    with pytest.raises(SystemExit):
        select_camera.derive(str(root), out=str(foreign))


def test_single_camera_dataset_is_returned_as_is_and_unknown_camera_is_refused(tmp_path):
    root = tmp_path / "one"; make_mini_v3(root, "single_camera")
    assert select_camera.derive(str(root), out=str(tmp_path / "unused")) == str(root)
    assert not (tmp_path / "unused").exists()
    with pytest.raises(SystemExit):
        select_camera.derive(str(root), keep="observation.images.nope", out=str(tmp_path / "x"))


def test_dataset_env_exports_quoted_camera_keys_and_action_ids(tmp_path):
    import subprocess
    root = tmp_path / "ds"; make_mini_v3(root)
    out = subprocess.run(["bash", "-c", f'eval "$(python3 {REPO}/trainer/dataset_env.py {root})"; echo "[$CAMERA_KEYS]"; echo "$ACTION_CHANNEL_IDS"'],
                         capture_output=True, text=True, check=True).stdout.split()
    assert out[0] == '["observation.images.ego","observation.images.left_wrist","observation.images.right_wrist"]'
    assert out[1] == "[" + ",".join(map(str, range(16))) + "]"
    out2 = subprocess.run(["bash", "-c", f'eval "$(python3 {REPO}/trainer/dataset_env.py {root})"; echo "$ACTION_DIM $STATE_DIM"'], capture_output=True, text=True, check=True).stdout.split()
    assert out2 == ["16", "16"]


def test_patch_base_config_empties_features_in_the_staged_snapshot(tmp_path):
    import patch_base_config
    snap = tmp_path / "hub" / "models--lerobot--lingbot_va_base" / "snapshots" / "abc"; snap.mkdir(parents=True)
    (snap / "config.json").write_text(json.dumps({"type": "lingbot_va", "input_features": {"observation.images.cam_high": {"shape": [3, 1, 1]}}, "output_features": {"action": {"shape": [14]}}, "height": 256}))
    done = patch_base_config.patch("lingbot_va", str(tmp_path))
    assert done and json.loads((snap / "config.json").read_text()) == {"type": "lingbot_va", "input_features": {}, "output_features": {}, "height": 256}
    assert patch_base_config.patch("lingbot_va", str(tmp_path)) == []      # idempotent
    assert patch_base_config.patch("act", str(tmp_path)) == []             # no patch in the registry


def test_local_repo_args_resolve_staged_snapshot_dirs(tmp_path):
    import local_repo_args
    for repo, sha in (("lerobot/fast-action-tokenizer", "aaa"), ("google/paligemma-3b-pt-224", "bbb")):
        d = tmp_path / "hub" / ("models--" + repo.replace("/", "--")); (d / "snapshots" / sha).mkdir(parents=True); (d / "refs").mkdir()
        (d / "refs" / "main").write_text(sha)
    toks = local_repo_args.tokens("pi0_fast", str(tmp_path))
    assert toks == [f"--policy.action_tokenizer_name={tmp_path}/hub/models--lerobot--fast-action-tokenizer/snapshots/aaa",
                    f"--policy.text_tokenizer_name={tmp_path}/hub/models--google--paligemma-3b-pt-224/snapshots/bbb"]
    assert local_repo_args.tokens("act", str(tmp_path)) == []
    with pytest.raises(SystemExit):
        local_repo_args.tokens("pi0_fast", str(tmp_path / "empty"))


def test_base_patches_replace_text_with_staged_snapshot_paths(tmp_path):
    import patch_base_config
    base = tmp_path / "hub" / "models--lerobot--pi0fast-base" / "snapshots" / "b1"; base.mkdir(parents=True)
    (base / "policy_preprocessor.json").write_text(json.dumps({"steps": [{"config": {"action_tokenizer_name": "physical-intelligence/fast"}}]}))
    (base / "policy_postprocessor.json").write_text(json.dumps({"steps": []}))
    tok = tmp_path / "hub" / "models--lerobot--fast-action-tokenizer"; (tok / "snapshots" / "t1").mkdir(parents=True); (tok / "refs").mkdir(); (tok / "refs" / "main").write_text("t1")
    done = patch_base_config.patch("pi0_fast", str(tmp_path))
    assert [os.path.basename(d) for d in done] == ["policy_preprocessor.json"]           # the postprocessor had nothing to replace
    assert json.loads((base / "policy_preprocessor.json").read_text())["steps"][0]["config"]["action_tokenizer_name"] == f"{tmp_path}/hub/models--lerobot--fast-action-tokenizer/snapshots/t1"
    assert patch_base_config.patch("pi0_fast", str(tmp_path)) == []                     # idempotent


def test_train_policy_runs_the_patch_hook_for_base_patches_only_entries():
    """pi0_fast has `base_patches` but no `base_config_patch`; the job must still run patch_base_config.py for it.
    (Regression: the hook once keyed on base_config_patch alone, so the FAST tokenizer path rewrite never happened in-job.)"""
    import subprocess, sys
    sh = (REPO / "trainer" / "train_policy.sh").read_text()
    assert "--print base_patches" in sh and "--print base_config_patch" in sh
    out = subprocess.run([sys.executable, str(REPO / "trainer" / "render_flags.py"), "--policy", "pi0_fast", "--print", "base_patches"],
                         capture_output=True, text=True)
    assert out.returncode == 0 and out.stdout.strip() not in ("", "None", "[]")
    # the shell condition: HF_HOME set and either key present -> the hook runs
    probe = 'BASE_PATCH="None"; BASE_PATCHES="[{x}]"; HF_HOME=/x; ' \
            'if [[ -n "${HF_HOME:-}" ]] && { [[ -n "$BASE_PATCH" && "$BASE_PATCH" != "None" ]] || [[ -n "$BASE_PATCHES" && "$BASE_PATCHES" != "None" && "$BASE_PATCHES" != "[]" ]]; }; then echo RUN; else echo SKIP; fi'
    assert subprocess.run(["bash", "-c", probe], capture_output=True, text=True).stdout.strip() == "RUN"
    assert probe.split("if ", 1)[1].split("; then")[0] in sh          # the probe is the script's own condition, verbatim
