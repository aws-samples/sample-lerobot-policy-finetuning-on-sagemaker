# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "trainer"))
import ensure_quantiles  # noqa: E402
from tests.fixtures.make_mini_v3 import make_mini_v3  # noqa: E402


def test_view_adds_missing_quantiles_from_the_data(tmp_path):
    root = tmp_path / "ds"; make_mini_v3(root, "no_q99")
    assert set(ensure_quantiles.missing_quantiles(str(root))) == {"action", "observation.state"}
    out = ensure_quantiles.derive(str(root), out=str(tmp_path / "view"))
    assert out == str(tmp_path / "view")
    stats = json.loads((Path(out) / "meta" / "stats.json").read_text())
    for k in ("action", "observation.state"):
        st = stats[k]
        assert all(q in st for q in ("q01", "q10", "q50", "q90", "q99")) and len(st["q01"]) == 16
        assert all(a <= b <= c for a, b, c in zip(st["q01"], st["q50"], st["q99"]))
        assert all(lo <= q for lo, q in zip(st["min"], st["q01"])) and all(q <= hi for q, hi in zip(st["q99"], st["max"]))
    assert "q01" not in json.loads((root / "meta" / "stats.json").read_text())["action"]   # source untouched
    assert (Path(out) / "data").is_symlink() and (Path(out) / "videos").is_symlink()
    assert ensure_quantiles.missing_quantiles(out) == []


def test_complete_dataset_is_returned_as_is_and_foreign_dir_is_refused(tmp_path):
    root = tmp_path / "ok"; make_mini_v3(root)
    assert ensure_quantiles.derive(str(root), out=str(tmp_path / "unused")) == str(root)
    assert not (tmp_path / "unused").exists()
    bad = tmp_path / "bad"; make_mini_v3(bad, "no_q99")
    foreign = tmp_path / "mine"; (foreign / "meta").mkdir(parents=True)
    with pytest.raises(SystemExit):
        ensure_quantiles.derive(str(bad), out=str(foreign))


def test_missing_camera_stats_get_imagenet_placeholders(tmp_path):
    root = tmp_path / "ds"; make_mini_v3(root)
    stats_path = root / "meta" / "stats.json"; stats = json.loads(stats_path.read_text())
    stats.pop("observation.images.ego"); stats_path.write_text(json.dumps(stats))
    assert ensure_quantiles.missing_image_stats(str(root)) == ["observation.images.ego"]
    out = ensure_quantiles.derive(str(root), out=str(tmp_path / "view"), quantiles=False)
    fixed = json.loads((Path(out) / "meta" / "stats.json").read_text())["observation.images.ego"]
    assert fixed["mean"] == [[[0.485]], [[0.456]], [[0.406]]] and fixed["count"] == [json.loads((root / "meta" / "info.json").read_text())["total_frames"]]
    assert ensure_quantiles.missing_image_stats(out) == []
    complete = tmp_path / "ok"; make_mini_v3(complete)
    assert ensure_quantiles.derive(str(complete), out=str(tmp_path / "unused"), quantiles=False) == str(complete)
