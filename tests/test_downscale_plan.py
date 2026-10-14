# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "trainer"))
import downscale_plan  # noqa: E402


def _info(*cams):
    return {"features": {k: {"dtype": "video", "shape": [h, w, 3]} for k, (h, w) in cams}}


def test_16_9_source_keeps_the_validated_640x360_and_no_warning():
    rows, warns = downscale_plan.plan(_info(("observation.images.ego", (720, 1280))), 360)
    assert rows == [("observation.images.ego", 640, 360, 720, 1280)] and warns == []


def test_4_3_source_is_aspect_preserving_with_a_warning():
    rows, warns = downscale_plan.plan(_info(("observation.images.up", (480, 640))), 360)
    assert rows[0][1:3] == (480, 360) and "not 16:9" in warns[0]


def test_width_is_even_and_bad_shape_raises():
    rows, _ = downscale_plan.plan(_info(("cam", (1080, 1917))), 360)
    assert rows[0][1] % 2 == 0
    with pytest.raises(ValueError):
        downscale_plan.plan({"features": {"cam": {"dtype": "video", "shape": [720]}}}, 360)


def test_cli_output_shape(tmp_path):
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta" / "info.json").write_text(json.dumps(_info(("a", (720, 1280)), ("b", (480, 640)))))
    r = subprocess.run([sys.executable, str(REPO / "trainer" / "downscale_plan.py"), str(tmp_path)], capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.split("\n")[:2] == ["a 640 360", "b 480 360"] and "WARN" in r.stderr
