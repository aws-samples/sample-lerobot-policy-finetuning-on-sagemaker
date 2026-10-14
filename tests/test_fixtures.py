# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""The synthetic fixture must mirror the v3.0 recorder export field by field, or every downstream check is tested
against the wrong shape. These tests pin the layout (file set, arrow schemas, index arithmetic, per-episode stats,
the pandas index metadata on tasks.parquet), the failure modes each variant injects, determinism, the recorder-PC
import contract and the time budget; the fixture uses only neutral robot, joint and task names.
The docker-marked tests prove the fixture is a dataset lerobot 0.6.1 actually loads (real videos in container mode,
host-mode parquet + metadata in the image). tests/test_check_dataset.py, tests/test_train_policy_render.py and
tests/test_entrypoint_sigterm.py consume the same module through
`from tests.fixtures.make_mini_v3 import make_mini_v3`; the keyword form, frames= and the arm alternation they rely on
are pinned here."""
import ast
import importlib.util
import json
import os
import pathlib
import subprocess
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import tests.fixtures.make_mini_v3 as generator_module
from tests.fixtures.make_mini_v3 import EXACT_LAG, MIN_FRAMES, make_mini_v3

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIXTURES_DIR = ROOT / "tests" / "fixtures"
GENERATOR = FIXTURES_DIR / "make_mini_v3.py"
FPS = 25
CAMERAS = ["observation.images.ego", "observation.images.left_wrist", "observation.images.right_wrist"]
STAT_KEYS = {"min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99"}
BASELINE_FILES = [
    "data/chunk-000/file-000.parquet",
    "data/chunk-000/file-001.parquet",
    "meta/episodes/chunk-000/file-000.parquet",
    "meta/info.json",
    "meta/stats.json",
    "meta/tasks.parquet",
    "videos/observation.images.ego/chunk-000/file-000.mp4",
    "videos/observation.images.left_wrist/chunk-000/file-000.mp4",
    "videos/observation.images.right_wrist/chunk-000/file-000.mp4",
]


def _files(root):
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())


def _data_files(root):
    return sorted(root.glob("data/**/*.parquet"))


def _arrays(path):
    t = pq.read_table(path, columns=["action", "observation.state", "episode_index", "frame_index"]).to_pydict()
    return (np.array(t["action"], dtype=np.float32), np.array(t["observation.state"], dtype=np.float32),
            np.array(t["episode_index"]), np.array(t["frame_index"]))


def _info(root):
    return json.loads((root / "meta" / "info.json").read_text())


# every variant is built and inspected by a dedicated test below; this set pins that no variant is added without one
EXERCISED = {"names_flat", "names_dict", "action_eq_state", "exact_lag_copy", "non_16_9", "few_episodes", "v21_version",
             "missing_tasks", "missing_stats", "image_dtype", "single_camera", "six_dims", "names_short", "no_q99",
             "gripper_q99_outlier"}


def _arm_per_episode(root):
    """Per-episode L/R label computed EXACTLY as trainer/check_dataset.py's _arm_report does it (state std of
    dims 0:7 vs 8:15, joints only)."""
    arms = {}
    for f in _data_files(root):
        t = pq.read_table(f, columns=["observation.state", "episode_index"]).to_pydict()
        s, ep = np.asarray(t["observation.state"], dtype=np.float32), np.asarray(t["episode_index"])
        for e in np.unique(ep):
            se = s[ep == e]
            arms[int(e)] = "L" if se[:, 0:7].std(axis=0).mean() > se[:, 8:15].std(axis=0).mean() else "R"
    return arms


# ---------------------------------------------------------------------------------------------- baseline layout
def test_every_variant_has_a_dedicated_test(make_mini_v3_module):
    assert set(make_mini_v3_module.VARIANTS) == EXERCISED
    assert all(v for v in make_mini_v3_module.VARIANTS.values())  # one-line description each


def test_conftest_and_consumers_share_one_module_object(make_mini_v3_module):
    """conftest imports the package module, not a by-path copy: the consumers' `make_mini_v3` IS the fixture's."""
    assert make_mini_v3_module is generator_module
    assert make_mini_v3_module.make_mini_v3 is make_mini_v3
    assert generator_module.__name__ == "tests.fixtures.make_mini_v3"


def test_baseline_file_set(mini_v3):
    assert _files(mini_v3) == BASELINE_FILES
    for cam in CAMERAS:
        assert (mini_v3 / "videos" / cam / "chunk-000" / "file-000.mp4").stat().st_size == 0


def test_info_json_fields(mini_v3):
    info = _info(mini_v3)
    rows = sum(pq.read_metadata(f).num_rows for f in _data_files(mini_v3))
    assert info["codebase_version"] == "v3.0"
    assert info["fps"] == FPS
    assert info["robot_type"] == "synthetic_bimanual"
    assert info["total_episodes"] == 12
    assert info["total_frames"] == rows
    assert info["total_tasks"] == 1
    assert info["chunks_size"] == 1000
    assert info["data_files_size_in_mb"] == 100 and info["video_files_size_in_mb"] == 200
    assert info["data_path"] == "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
    assert info["video_path"] == "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
    assert info["splits"] == {"train": "0:12"}
    feats = info["features"]
    assert list(feats) == ["action", "observation.state"] + CAMERAS + ["timestamp", "frame_index", "episode_index",
                                                                       "index", "task_index"]
    assert feats["action"] == {"dtype": "float32", "shape": [16], "names": feats["action"]["names"]}
    assert isinstance(feats["action"]["names"], list) and len(feats["action"]["names"]) == 16
    assert feats["action"]["names"][7] == "left_gripper" and feats["action"]["names"][15] == "right_gripper"
    assert feats["observation.state"] == feats["action"]
    for cam in CAMERAS:
        f = feats[cam]
        assert f["dtype"] == "video" and f["shape"] == [720, 1280, 3] and f["names"] == ["height", "width", "channels"]
        assert f["info"]["video.height"] == 720 and f["info"]["video.width"] == 1280
        assert f["info"]["video.codec"] == "h264" and f["info"]["video.pix_fmt"] == "yuv420p"
        assert f["info"]["video.fps"] == FPS and f["info"]["video.channels"] == 3
        assert f["info"]["has_audio"] is False and f["info"]["is_depth_map"] is False
        assert f["info"]["video.g"] == 2 and f["info"]["video.crf"] == 30 and f["info"]["video.preset"] is None
        assert f["info"]["video.fast_decode"] == 0 and f["info"]["video.video_backend"] == "pyav"
        assert f["info"]["video.extra_options"] == {}
    assert feats["timestamp"] == {"dtype": "float32", "shape": [1], "names": None}
    for k in ("frame_index", "episode_index", "index", "task_index"):
        assert feats[k] == {"dtype": "int64", "shape": [1], "names": None}


def test_data_parquet_schema(mini_v3):
    expected = pa.schema([
        ("action", pa.list_(pa.float32(), 16)),
        ("observation.state", pa.list_(pa.float32(), 16)),
        ("timestamp", pa.float32()),
        ("frame_index", pa.int64()),
        ("episode_index", pa.int64()),
        ("index", pa.int64()),
        ("task_index", pa.int64()),
    ])
    files = _data_files(mini_v3)
    assert [f.name for f in files] == ["file-000.parquet", "file-001.parquet"]
    for f in files:
        schema = pq.read_table(f).schema
        assert schema.equals(expected, check_metadata=False), schema
        assert not (schema.metadata or {}).get(b"huggingface")  # no hand-rolled HF metadata
        assert pq.ParquetFile(f).num_row_groups == 6  # one row group per episode, like the recorder and lerobot


def test_data_episode_split_and_index(mini_v3):
    tables = [pq.read_table(f).to_pydict() for f in _data_files(mini_v3)]
    assert sorted(set(tables[0]["episode_index"])) == list(range(0, 6))
    assert sorted(set(tables[1]["episode_index"])) == list(range(6, 12))
    index = np.concatenate([np.array(t["index"]) for t in tables])
    assert index.tolist() == list(range(len(index)))
    assert len(index) == _info(mini_v3)["total_frames"]
    for t in tables:
        assert set(t["task_index"]) == {0}
        ts = np.array(t["timestamp"], dtype=np.float32)
        expect = (np.array(t["frame_index"]) / FPS).astype(np.float32)
        assert np.array_equal(ts, expect)
        assert ts[1] == np.float32(0.04)  # 0.039999999... as float32, not the float64 0.04


def test_episodes_parquet(mini_v3):
    t = pq.read_table(mini_v3 / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    info = _info(mini_v3)
    rows = t.to_pylist()
    assert [r["episode_index"] for r in rows] == list(range(12))
    lengths = [r["length"] for r in rows]
    assert sum(lengths) == info["total_frames"]
    assert min(lengths) >= 24 and max(lengths) <= 40 and len(set(lengths)) > 1
    assert rows[0]["dataset_from_index"] == 0
    for a, b in zip(rows, rows[1:]):
        assert a["dataset_to_index"] == b["dataset_from_index"]
        assert b["dataset_to_index"] - b["dataset_from_index"] == b["length"]
    assert [r["data/file_index"] for r in rows] == [0] * 6 + [1] * 6
    assert all(r["data/chunk_index"] == 0 for r in rows)
    assert all(r["tasks"] == ["Stack the block on the marker"] for r in rows)
    for cam in CAMERAS:
        start = 0.0
        for r in rows:
            assert r["videos/%s/chunk_index" % cam] == 0 and r["videos/%s/file_index" % cam] == 0
            assert r["videos/%s/from_timestamp" % cam] == pytest.approx(start)
            assert r["videos/%s/to_timestamp" % cam] == pytest.approx(start + r["length"] / FPS)
            start += r["length"] / FPS
    schema = t.schema
    for feat in info["features"]:
        for key in STAT_KEYS:
            col = "stats/%s/%s" % (feat, key)
            assert col in t.column_names, col
            typ = schema.field(col).type
            if key == "count":
                assert typ == pa.list_(pa.int64())
            elif feat in CAMERAS:
                assert typ == pa.list_(pa.list_(pa.list_(pa.float64())))
            else:
                assert typ == pa.list_(pa.float64())
    for r in rows:
        assert r["stats/action/count"] == [r["length"]]
        assert r["stats/observation.images.ego/count"] == [r["length"]]
        assert len(r["stats/action/mean"]) == 16 and len(r["stats/timestamp/mean"]) == 1
        assert np.array(r["stats/observation.images.ego/mean"]).shape == (3, 1, 1)
        assert r["stats/frame_index/max"] == [float(r["length"] - 1)]
        assert r["stats/timestamp/max"] == [(r["length"] - 1) / FPS]  # float64 like lerobot, not the float32 column
    # pin the recorder column order too (presence + type above would let a reorder through)
    expected = ["episode_index", "tasks", "length", "data/chunk_index", "data/file_index", "dataset_from_index",
                "dataset_to_index"]
    for cam in CAMERAS:
        expected += ["videos/%s/%s" % (cam, k) for k in ("chunk_index", "file_index", "from_timestamp", "to_timestamp")]
    for feat in info["features"]:
        expected += ["stats/%s/%s" % (feat, k) for k in ("min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99")]
    expected += ["meta/episodes/chunk_index", "meta/episodes/file_index"]
    assert t.column_names == expected
    assert schema.field("episode_index").type == pa.int64() and schema.field("tasks").type == pa.list_(pa.string())
    assert schema.field("videos/observation.images.ego/from_timestamp").type == pa.float64()


def test_stats_json(mini_v3):
    stats = json.loads((mini_v3 / "meta" / "stats.json").read_text())
    info = _info(mini_v3)
    assert set(stats) == set(info["features"])
    for feat, st in stats.items():
        assert list(st) == ["min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99"], feat
        assert st["count"] == [info["total_frames"]]
        shape = (3, 1, 1) if feat in CAMERAS else ((16,) if feat in ("action", "observation.state") else (1,))
        for key in STAT_KEYS - {"count"}:
            assert np.array(st[key], dtype=np.float64).shape == shape, (feat, key)
    assert stats["observation.images.ego"]["min"] == [[[0.0]], [[0.0]], [[0.0]]]
    assert stats["observation.images.ego"]["max"] == [[[1.0]], [[1.0]], [[1.0]]]
    assert stats["index"]["max"] == [float(info["total_frames"] - 1)]
    a, s, _, _ = _arrays(_data_files(mini_v3)[0])
    a2, s2, _, _ = _arrays(_data_files(mini_v3)[1])
    allact = np.concatenate([a, a2]).astype(np.float64)
    assert stats["action"]["min"] == pytest.approx(allact.min(axis=0).tolist())
    assert stats["action"]["q50"] == pytest.approx(np.quantile(allact, 0.5, axis=0).tolist())


def test_tasks_parquet_pandas_index_metadata(mini_v3):
    t = pq.read_table(mini_v3 / "meta" / "tasks.parquet")
    assert t.schema.field("task_index").type == pa.int64()
    assert t.schema.field("task").type == pa.string()
    assert t.to_pydict() == {"task_index": [0], "task": ["Stack the block on the marker"]}
    md = json.loads(t.schema.metadata[b"pandas"])
    assert md["index_columns"] == ["task"]
    assert [c["name"] for c in md["columns"]] == ["task_index", "task"]
    assert md["creator"]["library"] == "pyarrow"


# ---------------------------------------------------------------------------------------------- signal
def test_baseline_action_differs_from_state_in_every_file(mini_v3):
    files = _data_files(mini_v3)
    assert len(files) == 2
    for f in files:
        a, s, ep, fi = _arrays(f)
        assert float(np.max(np.abs(a - s))) > 0.0, f.name
        # not an exact one-frame lag copy either: within every episode, action[t] != state[t+1] somewhere
        for e in np.unique(ep):
            m = ep == e
            assert not np.array_equal(a[m][:-1], s[m][1:]), (f.name, int(e))
    a, s, ep, fi = _arrays(files[0])
    grip = a[ep == ep[0]][:, [7, 15]]
    assert grip.min() < 0.1 and grip.max() > 0.9  # gripper dims step 0 -> 1 inside the episode
    # episode 0 moves the LEFT arm: its gripper steps, the right gripper stays closed
    assert grip[:, 0].max() > 0.9 and grip[:, 1].max() < 0.1


def test_baseline_arms_alternate_as_check_dataset_sees_them(mini_v3):
    arms = _arm_per_episode(mini_v3)
    assert [arms[e] for e in range(12)] == ["L", "R"] * 6
    assert sum(v == "L" for v in arms.values()) == 6 and sum(v == "R" for v in arms.values()) == 6
    for f in _data_files(mini_v3):
        a, s, ep, _ = _arrays(f)
        for e in np.unique(ep):
            se, ae = s[ep == e], a[ep == e]
            moving, holding = ((slice(0, 7), slice(8, 15)) if arms[int(e)] == "L" else (slice(8, 15), slice(0, 7)))
            mov_std, hold_std = se[:, moving].std(axis=0), se[:, holding].std(axis=0)
            assert mov_std.mean() > 0.1 and mov_std.min() > 0.01   # every moving joint really moves (>= 7x the jitter)
            assert hold_std.max() < 0.01                           # the holding arm is a pose plus tiny jitter
            assert mov_std.mean() > 20 * hold_std.mean()           # the validator's comparison, with a wide margin
            assert float(np.max(np.abs(ae[:, holding] - se[:, holding]))) > 0.0  # action still != state on the idle arm
            mov_grip, hold_grip = (7, 15) if arms[int(e)] == "L" else (15, 7)
            assert ae[:, mov_grip].max() > 0.9 and ae[:, mov_grip].min() < 0.1  # step on the moving arm only
            assert ae[:, hold_grip].max() < 0.1


def test_action_eq_state_only_in_second_file(mini_v3_factory):
    root = mini_v3_factory("action_eq_state")
    files = _data_files(root)
    a0, s0, _, _ = _arrays(files[0])
    a1, s1, _, _ = _arrays(files[1])
    assert float(np.max(np.abs(a0 - s0))) > 0.0
    assert np.array_equal(a1, s1)  # bit-exact float32 equality
    # a first-file-only check (files[0], first 5000 rows) would PASS this dataset; check_dataset.py must read every file:
    t = pq.read_table(files[0], columns=["action", "observation.state"]).slice(0, 5000).to_pydict()
    d = float(np.max(np.abs(np.array(t["action"], np.float32) - np.array(t["observation.state"], np.float32))))
    assert d > 0.0
    # while a check over ALL files catches it:
    worst = min(float(np.max(np.abs(_arrays(f)[0] - _arrays(f)[1]))) for f in files)
    assert worst == 0.0


def test_exact_lag_copy(mini_v3_factory):
    assert EXACT_LAG == 3  # the validator test asserts "shifted by 3"
    root = mini_v3_factory("exact_lag_copy")
    for f in _data_files(root):
        a, s, ep, _ = _arrays(f)
        assert float(np.max(np.abs(a - s))) > 0.0
        for e in np.unique(ep):
            m = ep == e
            assert np.array_equal(a[m][:-EXACT_LAG], s[m][EXACT_LAG:]), int(e)
            for k in range(1, EXACT_LAG + 1):
                assert np.array_equal(a[m][-k], s[m][-1])
            # the validator scans k = 1..10 and reports the FIRST exact shift: it must be EXACT_LAG, not a smaller k
            for k in range(1, EXACT_LAG):
                assert not np.array_equal(a[m][:-k], s[m][k:]), (int(e), k)


def test_non_16_9(mini_v3_factory):
    info = _info(mini_v3_factory("non_16_9"))
    for cam in CAMERAS:
        assert info["features"][cam]["shape"] == [480, 640, 3]
        assert info["features"][cam]["info"]["video.height"] == 480
        assert info["features"][cam]["info"]["video.width"] == 640


def test_few_episodes(mini_v3_factory):
    root = mini_v3_factory("few_episodes")
    info = _info(root)
    assert info["total_episodes"] == 4 and info["splits"] == {"train": "0:4"}
    ep = np.concatenate([_arrays(f)[2] for f in _data_files(root)])
    assert sorted(set(ep.tolist())) == [0, 1, 2, 3]
    assert pq.read_metadata(root / "meta" / "episodes" / "chunk-000" / "file-000.parquet").num_rows == 4


def test_v21_layout(mini_v3_factory):
    root = mini_v3_factory("v21_version")
    info = _info(root)
    assert info["codebase_version"] == "v2.1"
    assert info["total_chunks"] == 1 and info["total_videos"] == 12 * 3
    assert info["data_path"] == "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    assert info["video_path"] == "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
    files = _files(root)
    assert "meta/tasks.parquet" not in files and not (root / "meta" / "episodes").exists()
    assert {"meta/episodes.jsonl", "meta/tasks.jsonl", "meta/episodes_stats.jsonl"} <= set(files)
    assert "meta/stats.json" not in files  # v2.1 kept per-episode stats only in episodes_stats.jsonl
    assert [f for f in files if f.startswith("data/")] == ["data/chunk-000/episode_%06d.parquet" % i for i in range(12)]
    assert len([f for f in files if f.endswith(".mp4")]) == 36
    assert (root / "videos" / "chunk-000" / "observation.images.ego" / "episode_000000.mp4").stat().st_size == 0
    lines = [json.loads(l) for l in (root / "meta" / "episodes.jsonl").read_text().splitlines()]
    assert sum(l["length"] for l in lines) == info["total_frames"]
    es = [json.loads(l) for l in (root / "meta" / "episodes_stats.jsonl").read_text().splitlines()]
    assert set(es[0]["stats"]["action"]) == {"min", "max", "mean", "std", "count"}  # v2.1 had no quantiles
    assert "video.is_depth_map" in info["features"]["observation.images.ego"]["info"]
    assert "video.g" not in info["features"]["observation.images.ego"]["info"]  # encoder knobs arrived with v3.0
    assert json.loads((root / "meta" / "tasks.jsonl").read_text()) == {"task_index": 0,
                                                                        "task": "Stack the block on the marker"}


def test_missing_tasks_and_stats(mini_v3_factory):
    no_tasks = mini_v3_factory("missing_tasks")
    assert not (no_tasks / "meta" / "tasks.parquet").exists() and (no_tasks / "meta" / "stats.json").exists()
    no_stats = mini_v3_factory("missing_stats")
    assert not (no_stats / "meta" / "stats.json").exists() and (no_stats / "meta" / "tasks.parquet").exists()


def test_image_dtype(mini_v3_factory):
    root = mini_v3_factory("image_dtype")
    info = _info(root)
    assert info["video_path"] is None and not (root / "videos").exists()
    for cam in CAMERAS:
        assert info["features"][cam] == {"dtype": "image", "shape": [720, 1280, 3], "names": ["height", "width", "channels"]}
    cols = pq.read_table(root / "meta" / "episodes" / "chunk-000" / "file-000.parquet").column_names
    assert not any(c.startswith("videos/") for c in cols)


def test_single_camera(mini_v3_factory):
    root = mini_v3_factory("single_camera")
    info = _info(root)
    cams = [k for k, v in info["features"].items() if v["dtype"] == "video"]
    assert cams == ["observation.images.ego"]
    assert [f for f in _files(root) if f.endswith(".mp4")] == ["videos/observation.images.ego/chunk-000/file-000.mp4"]


def test_six_dims(mini_v3_factory):
    root = mini_v3_factory("six_dims")
    info = _info(root)
    assert info["robot_type"] == "synthetic_so101"
    assert info["features"]["action"]["shape"] == [6]
    assert info["features"]["action"]["names"] == ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex",
                                                   "wrist_roll", "gripper"]
    schema = pq.read_table(_data_files(root)[0]).schema
    assert schema.field("action").type == pa.list_(pa.float32(), 6)
    a, s, _, _ = _arrays(_data_files(root)[0])
    assert a.shape[1] == 6 and float(np.max(np.abs(a - s))) > 0.0


def test_names_dict_vs_flat(mini_v3_factory):
    flat = _info(mini_v3_factory("names_flat"))["features"]["action"]["names"]
    dct = _info(mini_v3_factory("names_dict"))["features"]["action"]["names"]
    assert isinstance(flat, list) and len(flat) == 16
    assert isinstance(dct, dict) and list(dct) == ["axes"] and dct["axes"] == flat


def test_combined_variants(mini_v3_factory):
    root = mini_v3_factory("few_episodes", "names_dict")
    info = _info(root)
    assert info["total_episodes"] == 4 and isinstance(info["features"]["action"]["names"], dict)


def test_names_short(mini_v3_factory):
    root = mini_v3_factory("few_episodes", "names_short")  # no assertion depends on the episode count
    feats = _info(root)["features"]
    assert feats["action"]["shape"] == [16] and len(feats["action"]["names"]) == 15  # dim - 1 entries
    assert feats["observation.state"]["names"] == feats["action"]["names"]
    assert pq.read_table(_data_files(root)[0]).schema.field("action").type == pa.list_(pa.float32(), 16)  # data untouched
    stats = json.loads((root / "meta" / "stats.json").read_text())
    assert len(stats["action"]["mean"]) == 16
    six = _info(mini_v3_factory("few_episodes", "names_short", "six_dims"))["features"]["action"]
    assert six["shape"] == [6] and len(six["names"]) == 5


def test_no_q99(mini_v3_factory, mini_v3):
    root = mini_v3_factory("no_q99")
    stats = json.loads((root / "meta" / "stats.json").read_text())
    for feat in ("action", "observation.state"):
        assert "q01" not in stats[feat] and "q99" not in stats[feat]
        assert set(stats[feat]) == {"min", "max", "mean", "std", "count", "q10", "q50", "q90"}
    assert set(stats["timestamp"]) == STAT_KEYS  # only the two vector features lose their quantiles
    healthy = json.loads((mini_v3 / "meta" / "stats.json").read_text())
    assert stats["action"]["max"] == healthy["action"]["max"]  # the values that remain are the healthy ones


def test_gripper_q99_outlier(mini_v3_factory, mini_v3):
    root = mini_v3_factory("gripper_q99_outlier")
    stats = json.loads((root / "meta" / "stats.json").read_text())["action"]
    healthy = json.loads((mini_v3 / "meta" / "stats.json").read_text())["action"]
    names = _info(root)["features"]["action"]["names"]
    for i, name in enumerate(names):
        rng = stats["max"][i] - stats["min"][i]
        ratio = (stats["max"][i] - stats["q99"][i]) / rng  # the validator's saturation inequality, verbatim
        if "gripper" in name:
            assert rng > 0 and ratio > 0.3, (name, ratio)
        else:
            assert ratio <= 0.3 and stats["max"][i] == healthy["max"][i], name
    assert stats["q99"] == healthy["q99"] and stats["min"] == healthy["min"]  # only max moved
    # the parquet data is untouched: same tables as the healthy build
    for rel in ("data/chunk-000/file-000.parquet", "data/chunk-000/file-001.parquet", "meta/episodes/chunk-000/file-000.parquet"):
        assert pq.read_table(root / rel).equals(pq.read_table(mini_v3 / rel)), rel
    six = json.loads((mini_v3_factory("few_episodes", "gripper_q99_outlier", "six_dims") / "meta" / "stats.json").read_text())["action"]
    assert (six["max"][5] - six["q99"][5]) / (six["max"][5] - six["min"][5]) > 0.3


def test_frames_pins_every_episode_length(mini_v3_factory):
    root = mini_v3_factory(frames=40)
    info = _info(root)
    rows = pq.read_table(root / "meta" / "episodes" / "chunk-000" / "file-000.parquet").to_pylist()
    assert [r["length"] for r in rows] == [40] * 12
    assert info["total_frames"] == 40 * 12 == sum(pq.read_metadata(f).num_rows for f in _data_files(root))
    ep = np.concatenate([_arrays(f)[2] for f in _data_files(root)])
    assert np.bincount(ep).tolist() == [40] * 12
    short = _info(mini_v3_factory("few_episodes", frames=MIN_FRAMES))
    assert short["total_frames"] == MIN_FRAMES * 4
    assert _arm_per_episode(root) == {e: ("L" if e % 2 == 0 else "R") for e in range(12)}  # alternation survives frames=


def test_variant_keyword_equals_positional(mini_v3_factory, mini_v3, tmp_path):
    kw = mini_v3_factory("few_episodes", variant="names_dict", frames=40)
    pos = mini_v3_factory("few_episodes", "names_dict", frames=40)
    for rel in ("data/chunk-000/file-000.parquet", "data/chunk-000/file-001.parquet",
                "meta/episodes/chunk-000/file-000.parquet", "meta/tasks.parquet"):
        assert pq.read_table(kw / rel).equals(pq.read_table(pos / rel)), rel
    assert (kw / "meta" / "info.json").read_text() == (pos / "meta" / "info.json").read_text()
    assert (kw / "meta" / "stats.json").read_text() == (pos / "meta" / "stats.json").read_text()
    # variant="default" is the baseline; the keyword combines with positional variants
    assert (mini_v3_factory(variant="default") / "meta" / "stats.json").read_text() == \
           (mini_v3 / "meta" / "stats.json").read_text()
    both = _info(mini_v3_factory("few_episodes", variant="names_dict"))
    assert both["total_episodes"] == 4 and isinstance(both["features"]["action"]["names"], dict)
    with pytest.raises(ValueError, match="variant must be"):
        make_mini_v3(tmp_path / "bad_x", variant=["names_dict"])
    with pytest.raises(ValueError, match="unknown variant"):
        make_mini_v3(tmp_path / "bad_y", variant="default_")
    assert not (tmp_path / "bad_x").exists() and not (tmp_path / "bad_y").exists()  # refused before touching disk


# ---------------------------------------------------------------------------------------------- contract
def test_determinism(mini_v3_factory):
    a = mini_v3_factory("few_episodes", seed=3)
    b = mini_v3_factory("few_episodes", seed=3)
    c = mini_v3_factory("few_episodes", seed=4)
    for rel in ("data/chunk-000/file-000.parquet", "data/chunk-000/file-001.parquet",
                "meta/episodes/chunk-000/file-000.parquet", "meta/tasks.parquet"):
        assert pq.read_table(a / rel).equals(pq.read_table(b / rel)), rel
    assert (a / "meta" / "stats.json").read_text() == (b / "meta" / "stats.json").read_text()
    # a different seed changes the action VALUES, not merely the seeded episode lengths: compare episode 0 on a common prefix
    aa, _, aep, _ = _arrays(_data_files(a)[0])
    ca, _, cep, _ = _arrays(_data_files(c)[0])
    a0, c0 = aa[aep == 0], ca[cep == 0]
    n = min(len(a0), len(c0))
    assert not np.array_equal(a0[:n], c0[:n])
    assert (a / "meta" / "stats.json").read_text() != (c / "meta" / "stats.json").read_text()


def test_unknown_variant_raises(mini_v3_factory):
    with pytest.raises(ValueError, match="names_flat"):
        mini_v3_factory("no_such_variant")


def test_refuses_non_empty_target(mini_v3, make_mini_v3_module):
    with pytest.raises(FileExistsError):
        make_mini_v3_module.make_mini_v3(mini_v3)  # the session baseline is a populated directory


def test_accepts_pre_created_empty_target(tmp_path, make_mini_v3_module):
    """Host and container mode share one contract: OUT_DIR may pre-exist as long as it is empty."""
    root = tmp_path / "pre"
    root.mkdir()
    assert make_mini_v3_module.make_mini_v3(root, "few_episodes") == root
    assert (root / "meta" / "info.json").is_file()


def test_argument_conflicts_raise(tmp_path, make_mini_v3_module):
    mk = make_mini_v3_module.make_mini_v3
    with pytest.raises(ValueError, match="few_episodes fixes episodes=4"):
        mk(tmp_path / "a", "few_episodes", episodes=12)
    with pytest.raises(ValueError, match="seed must be an int"):
        mk(tmp_path / "b", seed=None)
    with pytest.raises(ValueError, match="v21_version and image_dtype"):
        mk(tmp_path / "c", "v21_version", "image_dtype")
    with pytest.raises(ValueError, match="host-only"):
        mk(tmp_path / "d", "action_eq_state", mode="container")
    with pytest.raises(ValueError, match="EVEN height and width"):
        mk(tmp_path / "e", mode="container", height=143, width=256)
    with pytest.raises(ValueError, match="positive"):
        mk(tmp_path / "f", height=0, width=0)
    with pytest.raises(ValueError, match="episodes must be an int"):
        mk(tmp_path / "g", episodes=5.9)
    with pytest.raises(ValueError, match="frames must be >= %d" % MIN_FRAMES):
        mk(tmp_path / "h", frames=MIN_FRAMES - 1)
    with pytest.raises(ValueError, match="frames must be an int"):
        mk(tmp_path / "i", frames=40.0)
    with pytest.raises(ValueError, match="missing_stats"):
        mk(tmp_path / "j", "no_q99", "missing_stats")
    with pytest.raises(ValueError, match="v2.1"):
        mk(tmp_path / "k", "gripper_q99_outlier", "v21_version")
    with pytest.raises(ValueError, match="no_q99 removes it"):
        mk(tmp_path / "l", "gripper_q99_outlier", "no_q99")
    for host_only in ("names_short", "no_q99", "gripper_q99_outlier"):
        with pytest.raises(ValueError, match="host-only"):
            mk(tmp_path / host_only, host_only, mode="container")
    assert not any(tmp_path.iterdir())  # nothing touched disk
    pre = tmp_path / "pre"  # a caller's pre-existing empty directory must survive a container-mode refusal
    pre.mkdir()
    with pytest.raises(ValueError, match="host-only"):
        mk(pre, "action_eq_state", mode="container")
    assert pre.is_dir() and not any(pre.iterdir())
    (tmp_path / "file").write_text("x")
    with pytest.raises(FileExistsError, match="not a directory"):
        mk(tmp_path / "file")


def test_cli_errors_are_argparse_style(tmp_path, capsys, make_mini_v3_module):
    cases = [["--variant", "no_such_variant", str(tmp_path / "x")], ["--height", "144", str(tmp_path / "y")]]
    if importlib.util.find_spec("lerobot") is None:  # the recorder PC / CI box: --container must not print a traceback
        cases.append(["--container", str(tmp_path / "z")])
    for argv in cases:
        with pytest.raises(SystemExit) as e:
            make_mini_v3_module.main(argv)
        assert e.value.code == 2, argv
        err = capsys.readouterr().err
        assert "Traceback" not in err and "error:" in err, argv
    assert not any(tmp_path.iterdir())  # every refusal happened before anything touched disk


def test_cli_list_variants_and_build(tmp_path, capsys, make_mini_v3_module):
    assert make_mini_v3_module.main(["--list-variants"]) == 0
    assert capsys.readouterr().out.split() == list(make_mini_v3_module.VARIANTS)
    out = tmp_path / "cli"
    assert make_mini_v3_module.main([str(out), "--variant", "few_episodes", "--seed", "7", "--frames", "10"]) == 0
    assert capsys.readouterr().out.strip() == str(out)
    assert _info(out)["total_episodes"] == 4 and _info(out)["total_frames"] == 40
    dflt = tmp_path / "cli_default"  # the CLI spells the baseline like the keyword form does
    assert make_mini_v3_module.main([str(dflt), "--variant", "default", "--variant", "few_episodes", "--frames", "10"]) == 0
    assert capsys.readouterr().out.strip() == str(dflt) and _info(dflt)["total_episodes"] == 4


def test_module_imports_only_stdlib_pyarrow_numpy():
    """Recorder-PC contract: the generator must run where neither lerobot nor pandas nor torch is installed."""
    tree = ast.parse(GENERATOR.read_text())
    top = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            top.add(node.module.split(".")[0])
    stdlib = {"__future__", "argparse", "json", "math", "pathlib", "sys"}
    assert top <= stdlib | {"numpy", "pyarrow"}, top
    assert ast.get_docstring(tree)  # module docstring present
    assert any(isinstance(n, ast.ImportFrom) and n.module == "__future__" for n in tree.body)


def test_host_build_time_budget(mini_v3_factory):
    t0 = time.perf_counter()
    mini_v3_factory()
    assert time.perf_counter() - t0 < 0.5


# ---------------------------------------------------------------------------------------------- docker lane
def _docker_python(image, mounts, args):
    cmd = ["docker", "run", "--rm", "--network", "none", "--user", "%d:%d" % (os.getuid(), os.getgid()),
           "-e", "HOME=/tmp", "-e", "HF_HOME=/tmp/hf", "-e", "HF_LEROBOT_HOME=/tmp/lerobot", "-e", "HF_HUB_OFFLINE=1"]
    for host, guest in mounts:
        cmd += ["-v", "%s:%s" % (host, guest)]
    cmd += ["--entrypoint", "python", image] + list(args)
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout


@pytest.mark.docker
def test_container_mode_builds_real_videos(tmp_path, train_image):
    (tmp_path / "mini").mkdir()  # pre-existing EMPTY target must work, as in host mode (lerobot wants to mkdir it)
    _docker_python(train_image, [(FIXTURES_DIR, "/fixtures:ro"), (tmp_path, "/out")],
                   ["/fixtures/make_mini_v3.py", "/out/mini", "--container", "--height", "144", "--width", "256"])
    root = tmp_path / "mini"
    info = _info(root)
    assert info["codebase_version"] == "v3.0" and info["fps"] == FPS and info["robot_type"] == "synthetic_bimanual"
    assert info["total_episodes"] == 12
    assert info["total_frames"] == sum(pq.read_metadata(f).num_rows for f in _data_files(root))
    for cam in CAMERAS:
        assert info["features"][cam]["shape"] == [144, 256, 3]
        assert info["features"][cam]["info"]["video.codec"] == "h264"
        mp4s = sorted((root / "videos" / cam).glob("chunk-*/file-*.mp4"))
        assert mp4s and all(p.stat().st_size > 0 for p in mp4s)
    out = _docker_python(train_image, [(tmp_path, "/out")], ["-c", "\n".join([
        "import json",
        "from lerobot.datasets.lerobot_dataset import LeRobotDataset",
        "ds = LeRobotDataset('synthetic/mini_v3', root='/out/mini')",
        "x = ds[0]['observation.images.ego']",
        "print(json.dumps({'len': len(ds), 'shape': list(x.shape), 'action': list(ds[0]['action'].shape)}))",
    ])])
    got = json.loads(out.strip().splitlines()[-1])
    assert got == {"len": info["total_frames"], "shape": [3, 144, 256], "action": [16]}


@pytest.mark.docker
def test_host_mode_baseline_loads_in_image(mini_v3, train_image):
    info = _info(mini_v3)
    out = _docker_python(train_image, [(mini_v3, "/out/mini:ro")], ["-c", "\n".join([
        "import json",
        "from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata",
        "task = 'Stack the block on the marker'",
        "meta = LeRobotDatasetMetadata('synthetic/mini_v3', root='/out/mini')",
        "ds = LeRobotDataset('synthetic/mini_v3', root='/out/mini')",  # videos are never decoded here
        "print(json.dumps({'total_frames': meta.total_frames, 'has_task': task in meta.tasks.index,",
        "                  'task_index': meta.get_task_index(task), 'len': len(ds),",
        "                  'action_len': len(ds.hf_dataset[0]['action'])}))",
    ])])
    got = json.loads(out.strip().splitlines()[-1])
    assert got == {"total_frames": info["total_frames"], "has_task": True, "task_index": 0,
                   "len": info["total_frames"], "action_len": 16}
