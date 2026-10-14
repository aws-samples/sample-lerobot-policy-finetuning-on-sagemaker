#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Synthetic LeRobot v3.0 dataset fixtures for the host test suite and the docker lane.

WHY: the dataset checks (check_dataset.py and the trainer's own guards) must be exercised against every failure mode
we have met on real recorder exports -- action == state in a parquet file the first-file-only check never reads,
an action that is an exact EXACT_LAG-frame lag copy of the state, 4:3 cameras, too few episodes, a v2.1 layout,
missing tasks/stats metadata, image-dtype cameras, a names list shorter than the action, stats without quantiles, a
gripper whose max sits far above q99 -- without shipping a real recording. Everything here is synthetic: neutral
robot type, neutral axis names, neutral task text, sinusoidal joints and a stepped gripper. In the 16-dim bimanual
baseline the arms alternate: even episodes move the LEFT arm (dims 0-7) while the right arm holds a constant pose,
odd episodes move the RIGHT arm, so an arm report over 12 episodes reads 6 left-arm and 6 right-arm.

Two modes share one feature description:

* host mode (default): stdlib + pyarrow + numpy only, python >= 3.9. Writes the parquet files and the metadata
  exactly as the v3.0 recorder export does (two data files, meta/episodes parquet with per-episode stats,
  meta/tasks.parquet with the pandas index metadata LeRobot relies on, meta/stats.json) and ZERO-BYTE mp4
  placeholders. The whole baseline builds in well under half a second. This file is also the "recorder PC
  contract": it must keep importing only stdlib + pyarrow + numpy at module level, because the same file is run
  on the machine that produced the recording, where neither lerobot nor pandas is installed.
* container mode (--container): runs inside the training image (python 3.12, lerobot 0.6.1) and records the same
  synthetic episodes through LeRobotDataset.create / add_frame / save_episode / finalize with the h264 (libx264) encoder,
  producing REAL h264 videos that the trainer can decode. lerobot is imported lazily inside the function. lerobot
  creates OUT_DIR itself, so an existing (empty) OUT_DIR is removed first; height and width must be even (yuv420p).

    python3 tests/fixtures/make_mini_v3.py OUT_DIR [--variant NAME ...] [--episodes N] [--frames N] [--seed S]
                                           [--container] [--height H --width W] [--list-variants]

Variants (combine freely, e.g. ``make_mini_v3(root, "few_episodes", "names_dict")``) are documented in VARIANTS.
The keyword form ``make_mini_v3(root, variant="names_dict", frames=40)`` is equivalent: `variant` is appended to the
positional variants ("default" means the baseline) and `frames` makes every episode exactly that long instead of the
seeded 24-40 frames. The build is deterministic for a given (variants, episodes, frames, seed); `seed` must be an int
for that reason.
The `image_dtype` variant is a NEGATIVE fixture only: it declares dtype "image" cameras in meta/info.json but does
not embed image bytes in the data parquet (a real image-dtype dataset would); it exists so the validator can refuse
image-dtype datasets, nothing more.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

FPS = 25
DEFAULT_EPISODES = 12
FEW_EPISODES = 4
MIN_LEN, MAX_LEN = 24, 40  # frames per episode, seeded per episode (unless frames= pins them)
MIN_FRAMES = 8  # smallest frames= a caller may pin: room for the gripper step and the EXACT_LAG shift
EXACT_LAG = 3  # exact_lag_copy: action[t] == state[t + EXACT_LAG]; the last EXACT_LAG actions equal the final state
HOLD_NOISE_STD = 1e-3  # holding-arm jitter; the moving arm's std is ~0.15-0.6 at the seeded lengths, > 30x this at MIN_FRAMES
REPO_ID = "synthetic/mini_v3"
TASK = "Stack the block on the marker"
ROBOT_BIMANUAL = "synthetic_bimanual"
ROBOT_SIX = "synthetic_so101"
CAMERAS = ("observation.images.ego", "observation.images.left_wrist", "observation.images.right_wrist")
# 16-dim layout of the trainer: dims 0-6 left joints, 7 left gripper, 8-14 right joints, 15 right gripper.
BIMANUAL_NAMES = (["left_joint_%d" % i for i in range(1, 8)] + ["left_gripper"]
                  + ["right_joint_%d" % i for i in range(1, 8)] + ["right_gripper"])
LEFT_ARM_DIMS, RIGHT_ARM_DIMS = list(range(0, 8)), list(range(8, 16))  # joints + gripper of each arm
SIX_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
STAT_KEYS = ("min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99")  # recorder column order
QUANTILES = {"q01": 0.01, "q10": 0.10, "q50": 0.50, "q90": 0.90, "q99": 0.99}
IMAGE_STATS = {"min": 0.0, "max": 1.0, "mean": 0.5, "std": 0.25, "q01": 0.01, "q10": 0.1, "q50": 0.5, "q90": 0.9, "q99": 0.99}

VARIANTS = {
    "names_flat": "baseline: action/observation.state names are a flat list (canonical LeRobot)",
    "names_dict": "action/observation.state names are {'axes': [...]} (the recorder output shape; validator WARNs)",
    "action_eq_state": "action == observation.state bit-exactly in data/chunk-000/file-001.parquet ONLY",
    "exact_lag_copy": "action[t] == observation.state[t+%d] exactly within each episode (last %d actions == final state)"
                      % (EXACT_LAG, EXACT_LAG),
    "non_16_9": "cameras are 480x640 (4:3) instead of 720x1280",
    "few_episodes": "4 episodes (< 10 triggers the 'set HELDOUT or EPISODES' FATAL without FORCE)",
    "v21_version": "LeRobot v2.1 layout (codebase_version v2.1, jsonl metadata, one parquet per episode, no stats.json)",
    "missing_tasks": "v3.0 without meta/tasks.parquet",
    "missing_stats": "v3.0 without meta/stats.json",
    "image_dtype": "cameras declared with dtype 'image' (no videos/ dir, video_path null); negative fixture only",
    "single_camera": "only observation.images.ego",
    "six_dims": "D = 6 with SO-101 style flat names, robot_type synthetic_so101",
    "names_short": "action/observation.state names list has D-1 entries (validator FAILs on names vs dim); host-only",
    "no_q99": "meta/stats.json action and observation.state lack q01/q99 (QUANTILES normalisation cannot run); host-only",
    "gripper_q99_outlier": "meta/stats.json action max for the gripper dims far above q99 (saturation WARN); parquet data "
                           "untouched; host-only",
}
CONTAINER_VARIANTS = ("names_flat", "names_dict", "non_16_9", "few_episodes", "single_camera", "six_dims", "exact_lag_copy")
VIDEO_INFO_TEMPLATE = {
    "video.codec": "h264", "video.pix_fmt": "yuv420p", "video.fps": FPS, "video.channels": 3, "has_audio": False,
    "video.g": 2, "video.crf": 30, "video.preset": None, "video.fast_decode": 0, "video.video_backend": "pyav",
    "video.extra_options": {}, "is_depth_map": False,
}
V21_VIDEO_INFO_TEMPLATE = {
    "video.fps": FPS, "video.channels": 3, "video.codec": "h264", "video.pix_fmt": "yuv420p",
    "video.is_depth_map": False, "has_audio": False,
}
V21_STAT_KEYS = ("min", "max", "mean", "std", "count")  # quantiles arrived with v3.0


# --------------------------------------------------------------------------------------------------------------
# spec: which variants -> one resolved description of the dataset to write
# --------------------------------------------------------------------------------------------------------------
def resolve_spec(variants, episodes=None, frames=None, height=None, width=None):
    unknown = sorted(set(variants) - set(VARIANTS))
    if unknown:
        raise ValueError("unknown variant(s) %s; valid names: %s" % (unknown, ", ".join(VARIANTS)))
    if "names_flat" in variants and "names_dict" in variants:
        raise ValueError("names_flat and names_dict are mutually exclusive")
    v = set(variants)
    if "v21_version" in v and "image_dtype" in v:
        raise ValueError("v21_version and image_dtype cannot be combined (the v2.1 writer only knows video cameras)")
    if "missing_stats" in v and ("no_q99" in v or "gripper_q99_outlier" in v):
        raise ValueError("no_q99 and gripper_q99_outlier edit meta/stats.json; missing_stats removes it")
    if "v21_version" in v and ("no_q99" in v or "gripper_q99_outlier" in v):
        raise ValueError("no_q99 and gripper_q99_outlier edit meta/stats.json; the v2.1 layout has none")
    if "no_q99" in v and "gripper_q99_outlier" in v:
        raise ValueError("gripper_q99_outlier needs q99 in stats.json; no_q99 removes it")
    if episodes is not None:
        if isinstance(episodes, bool) or not isinstance(episodes, (int, np.integer)):
            raise ValueError("episodes must be an int, got %r" % (episodes,))
        episodes = int(episodes)
    if frames is not None:
        if isinstance(frames, bool) or not isinstance(frames, (int, np.integer)):
            raise ValueError("frames must be an int, got %r" % (frames,))
        frames = int(frames)
        if frames < MIN_FRAMES:
            raise ValueError("frames must be >= %d, got %d" % (MIN_FRAMES, frames))
    if "six_dims" in v:
        names, robot_type = list(SIX_NAMES), ROBOT_SIX
    else:
        names, robot_type = list(BIMANUAL_NAMES), ROBOT_BIMANUAL
    cam_h, cam_w = (480, 640) if "non_16_9" in v else (720, 1280)
    if (height is None) != (width is None):
        raise ValueError("height and width must be given together")
    if height is not None:
        cam_h, cam_w = int(height), int(width)
        if cam_h <= 0 or cam_w <= 0:
            raise ValueError("height and width must be positive, got %dx%d" % (cam_h, cam_w))
    if "few_episodes" in v and episodes is not None and episodes != FEW_EPISODES:
        raise ValueError("few_episodes fixes episodes=%d; do not pass episodes=%r as well" % (FEW_EPISODES, episodes))
    if episodes is None:
        episodes = FEW_EPISODES if "few_episodes" in v else DEFAULT_EPISODES
    if episodes < 2:
        raise ValueError("need at least 2 episodes (one per data file)")
    if "action_eq_state" in v and "exact_lag_copy" in v:
        raise ValueError("action_eq_state and exact_lag_copy are mutually exclusive")
    action_mode = "eq_state_second_file" if "action_eq_state" in v else ("exact_lag" if "exact_lag_copy" in v else "healthy")
    return {
        "variants": tuple(variants),
        "n_episodes": episodes,
        "frames": frames,  # None: seeded per-episode lengths in [MIN_LEN, MAX_LEN]
        "dim": len(names),
        "names_declared": names[:-1] if "names_short" in v else names,  # what meta/info.json declares
        "names_dict": "names_dict" in v,
        "robot_type": robot_type,
        "cameras": ("observation.images.ego",) if "single_camera" in v else CAMERAS,
        "cam_shape": [cam_h, cam_w, 3],
        "camera_dtype": "image" if "image_dtype" in v else "video",
        "layout": "v2.1" if "v21_version" in v else "v3.0",
        "write_tasks": "missing_tasks" not in v,
        "write_stats": "missing_stats" not in v,
        "stats_quantiles": "no_q99" not in v,
        "gripper_q99_outlier": "gripper_q99_outlier" in v,
        "action_mode": action_mode,
    }


def feature_dict(spec, with_video_info=True, v21=False):
    """The meta/info.json `features` block (also what LeRobotDataset.create needs, minus the video info)."""
    names = {"axes": list(spec["names_declared"])} if spec["names_dict"] else list(spec["names_declared"])
    feats = {
        "action": {"dtype": "float32", "shape": [spec["dim"]], "names": names},
        "observation.state": {"dtype": "float32", "shape": [spec["dim"]], "names": names},
    }
    for cam in spec["cameras"]:
        f = {"dtype": spec["camera_dtype"], "shape": list(spec["cam_shape"]), "names": ["height", "width", "channels"]}
        if spec["camera_dtype"] == "video" and with_video_info:
            info = {"video.height": spec["cam_shape"][0], "video.width": spec["cam_shape"][1]}
            info.update(V21_VIDEO_INFO_TEMPLATE if v21 else VIDEO_INFO_TEMPLATE)
            f["info"] = info
        feats[cam] = f
    feats["timestamp"] = {"dtype": "float32", "shape": [1], "names": None}
    for k in ("frame_index", "episode_index", "index", "task_index"):
        feats[k] = {"dtype": "int64", "shape": [1], "names": None}
    return feats


# --------------------------------------------------------------------------------------------------------------
# signal: deterministic per (n_episodes, dim, seed); variants only post-process the action
# --------------------------------------------------------------------------------------------------------------
def gripper_dims(dim):
    return [7, 15] if dim == 16 else [dim - 1]


def holding_arm_dims(dim, episode_index):
    """Bimanual baseline: the arm that HOLDS in this episode (even episodes move the left arm, odd the right one);
    [] for the single-arm layout, where every dim moves in every episode."""
    if dim != 16:
        return []
    return RIGHT_ARM_DIMS if episode_index % 2 == 0 else LEFT_ARM_DIMS


def first_file_episodes(n):
    """Episodes 0..first-1 go to data file-000, the rest to file-001; action_eq_state poisons file-001 ONLY, so the
    writer and the variant must agree on this split -- hence one function."""
    return n // 2


def generate_episodes(spec, seed):
    """Return a list of (state, action) float32 arrays of shape [L, D], one pair per episode.

    The healthy action is state one frame ahead plus a small constant per-dim offset plus noise, i.e. deliberately
    CLOSE to a lag copy (MAE to state[t+1] is about half the MAE to state[t]). A lag check must therefore be exact
    float32 equality of action[t] and state[t+k] within each episode (the exact_lag_copy contract, k = EXACT_LAG); a
    tolerance- or ratio-based heuristic would flag this healthy baseline too.

    16-dim layout: the arms alternate. Even episodes move the left arm (sinusoidal joints, gripper step) while the
    right arm holds one random pose plus HOLD_NOISE_STD jitter and its gripper stays closed; odd episodes swap. The
    per-episode joint std of the moving arm (about 0.15-0.6 at the seeded lengths, still > 30x HOLD_NOISE_STD at
    frames=MIN_FRAMES) is therefore always far above the holding arm's, which is exactly the comparison a bimanual
    arm report makes. The 6-dim layout moves every dim in every episode.
    """
    rng = np.random.default_rng(seed)
    dim, fps = spec["dim"], FPS
    if spec["frames"] is not None:
        lengths = np.full(spec["n_episodes"], spec["frames"], dtype=np.int64)
    else:
        lengths = rng.integers(MIN_LEN, MAX_LEN + 1, size=spec["n_episodes"])
    grippers = gripper_dims(dim)
    out = []
    for e, length in enumerate(lengths):
        t = np.arange(int(length) + 1, dtype=np.float64) / fps  # one extra frame so "one frame ahead" exists
        phase = rng.uniform(0.0, 2.0 * math.pi, size=dim)
        freq = rng.uniform(0.3, 0.9, size=dim)
        amp = rng.uniform(0.2, 0.6, size=dim)
        traj = amp * np.sin(2.0 * math.pi * freq * t[:, None] + phase)
        step_at = int(length) // 2 + int(rng.integers(-2, 3))
        for g in grippers:
            traj[:, g] = (np.arange(int(length) + 1) >= step_at).astype(np.float64)
        hold = holding_arm_dims(dim, e)
        if hold:  # bimanual: one arm holds a constant pose, its gripper stays closed (the step is on the moving arm)
            pose = rng.uniform(-0.6, 0.6, size=len(hold))
            traj[:, hold] = pose
            traj[:, [g for g in grippers if g in hold]] = 0.0
        traj += rng.normal(0.0, HOLD_NOISE_STD, size=traj.shape)
        state = traj[:-1]
        ahead = traj[1:]
        offset = 0.02 * (rng.random(dim) + 0.5)  # small tracking offset per dim, never zero
        action = ahead + offset + rng.normal(0.0, 1e-3, size=state.shape)
        state32 = state.astype(np.float32)
        action32 = action.astype(np.float32)
        if spec["action_mode"] == "exact_lag":  # exact copy EXACT_LAG frames ahead, clamped at the final state
            action32 = np.concatenate([state32[EXACT_LAG:], np.repeat(state32[-1:], EXACT_LAG, axis=0)], axis=0)
        out.append((state32, action32))
    if spec["action_mode"] == "eq_state_second_file":
        first = first_file_episodes(spec["n_episodes"])
        out = [(s, (s.copy() if i >= first else a)) for i, (s, a) in enumerate(out)]
    return out


def synthetic_frame(height, width, frame_index, cam_index):
    """uint8 HWC frame: cheap gradient plus a moving square (fast to encode, cheap to build)."""
    yy = np.linspace(0, 255, height, dtype=np.float32)[:, None]
    xx = np.linspace(0, 255, width, dtype=np.float32)[None, :]
    img = np.empty((height, width, 3), dtype=np.uint8)
    img[..., 0] = np.broadcast_to(yy, (height, width)).astype(np.uint8)
    img[..., 1] = np.broadcast_to(xx, (height, width)).astype(np.uint8)
    img[..., 2] = (40 * cam_index + 60) % 256
    side = max(4, min(height, width) // 6)
    y0 = (frame_index * 3) % max(1, height - side)
    x0 = (frame_index * 5 + cam_index * 17) % max(1, width - side)
    img[y0:y0 + side, x0:x0 + side, :] = (255, 255, 255)
    return img


# --------------------------------------------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------------------------------------------
def vector_stats(arr):
    """arr [N, D] -> {key: list of D floats} plus count [N]. Scalars are passed as [N, 1].

    Exact global quantiles (spec). lerobot's writer instead stores a length-weighted mean of per-episode quantiles,
    so real exports have e.g. index/q01 near N/2; a check that compares q99 to max must tolerate that.
    """
    a = np.asarray(arr, dtype=np.float64)
    st = {"min": a.min(axis=0), "max": a.max(axis=0), "mean": a.mean(axis=0), "std": a.std(axis=0),
          "count": np.array([a.shape[0]], dtype=np.int64)}
    for k, q in QUANTILES.items():
        st[k] = np.quantile(a, q, axis=0)
    return {k: v.tolist() for k, v in st.items()}


def image_stats(count):
    """Synthetic per-channel image stats in [0, 1], shaped [3][1][1] like lerobot's, with count = number of frames.

    Real exports have count < length for episodes longer than 100 frames -- lerobot SAMPLES at most 100 frames per
    episode for image statistics -- so a dataset check must assert count == length for NON-image features only.
    """
    return {k: ([int(count)] if k == "count" else [[[IMAGE_STATS[k]]], [[IMAGE_STATS[k]]], [[IMAGE_STATS[k]]]])
            for k in STAT_KEYS}


def all_feature_stats(spec, columns):
    """columns: dict feature -> np array ([N, D] for vectors, [N] for scalars); cameras get synthetic stats."""
    n = columns["index"].shape[0]
    stats = {}
    for feat in feature_dict(spec):
        if feat in spec["cameras"]:
            stats[feat] = image_stats(n)
        elif feat in ("action", "observation.state"):
            stats[feat] = vector_stats(columns[feat])
        elif feat == "timestamp":
            # lerobot computes these from the float64 frame_index / fps BEFORE the float32 cast of the parquet column
            # (reference: stats/timestamp/max == (length - 1) / fps exactly, parquet max is the float32 neighbour)
            stats[feat] = vector_stats((columns["frame_index"].astype(np.float64) / FPS)[:, None])
        else:
            stats[feat] = vector_stats(columns[feat][:, None])
    return stats


def dataset_stats(spec, all_frames):
    """meta/stats.json: the global stats, then the two stats-only variants (the parquet data is never touched)."""
    stats = all_feature_stats(spec, all_frames)
    if not spec["stats_quantiles"]:
        for feat in ("action", "observation.state"):
            for key in ("q01", "q99"):
                del stats[feat][key]
    if spec["gripper_q99_outlier"]:
        st = stats["action"]
        for g in gripper_dims(spec["dim"]):
            # (max - q99) / (max - min) becomes >= 0.8, well past the 0.3 a saturation check uses
            spread = max(st["q99"][g] - st["min"][g], 1e-3)
            st["max"][g] = st["q99"][g] + 4.0 * spread
    return stats


# --------------------------------------------------------------------------------------------------------------
# arrow tables
# --------------------------------------------------------------------------------------------------------------
def data_table(spec, frames):
    """frames: dict of numpy columns -> arrow table with the exact recorder schema and no custom metadata."""
    dim = spec["dim"]

    def fsl(arr):
        flat = pa.array(np.ascontiguousarray(arr, dtype=np.float32).ravel(), type=pa.float32())
        return pa.FixedSizeListArray.from_arrays(flat, dim)

    return pa.table({
        "action": fsl(frames["action"]),
        "observation.state": fsl(frames["observation.state"]),
        "timestamp": pa.array(frames["timestamp"], type=pa.float32()),
        "frame_index": pa.array(frames["frame_index"], type=pa.int64()),
        "episode_index": pa.array(frames["episode_index"], type=pa.int64()),
        "index": pa.array(frames["index"], type=pa.int64()),
        "task_index": pa.array(frames["task_index"], type=pa.int64()),
    })


def frames_of(episodes, ep_indices, first_global_index):
    """Concatenate the given episodes into the per-frame columns of one data file."""
    cols = {"action": [], "observation.state": [], "timestamp": [], "frame_index": [], "episode_index": [], "index": [],
            "task_index": []}
    g = first_global_index
    for ep in ep_indices:
        state, action = episodes[ep]
        n = state.shape[0]
        fi = np.arange(n, dtype=np.int64)
        cols["action"].append(action)
        cols["observation.state"].append(state)
        cols["timestamp"].append((fi / FPS).astype(np.float32))
        cols["frame_index"].append(fi)
        cols["episode_index"].append(np.full(n, ep, dtype=np.int64))
        cols["index"].append(np.arange(g, g + n, dtype=np.int64))
        cols["task_index"].append(np.zeros(n, dtype=np.int64))
        g += n
    return {k: np.concatenate(v, axis=0) for k, v in cols.items()}


def tasks_table():
    """meta/tasks.parquet: LeRobot 0.6.1 reads it with pandas and expects the task strings as the DataFrame index.

    `task` is written as arrow `string` (what lerobot 0.6.1 / pandas 2.x emit). Recorders on pandas >= 3 emit
    `large_string` with numpy_type "str" in the pandas metadata instead; pd.read_parquet restores both identically,
    so a v3.0 validator must accept pa.string() OR pa.large_string() here rather than copy the fixture test's pin.
    """
    pandas_md = {
        "index_columns": ["task"],
        "column_indexes": [{"name": None, "field_name": None, "pandas_type": "unicode", "numpy_type": "object",
                            "metadata": {"encoding": "UTF-8"}}],
        "columns": [
            {"name": "task_index", "field_name": "task_index", "pandas_type": "int64", "numpy_type": "int64", "metadata": None},
            {"name": "task", "field_name": "task", "pandas_type": "unicode", "numpy_type": "object", "metadata": None},
        ],
        "creator": {"library": "pyarrow", "version": pa.__version__},
        "pandas_version": "2.2.3",
    }
    schema = pa.schema([("task_index", pa.int64()), ("task", pa.string())],
                       metadata={b"pandas": json.dumps(pandas_md).encode()})
    return pa.table({"task_index": pa.array([0], pa.int64()), "task": pa.array([TASK], pa.string())}, schema=schema)


def episodes_table(spec, episodes, file_of_episode, per_episode_stats):
    """meta/episodes/chunk-000/file-000.parquet: one row per episode, recorder column order and types."""
    n = len(episodes)
    lengths = [int(s.shape[0]) for s, _ in episodes]
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64)
    cols = [
        ("episode_index", pa.array(list(range(n)), pa.int64())),
        ("tasks", pa.array([[TASK]] * n, pa.list_(pa.string()))),
        ("length", pa.array(lengths, pa.int64())),
        ("data/chunk_index", pa.array([0] * n, pa.int64())),
        ("data/file_index", pa.array([file_of_episode[e] for e in range(n)], pa.int64())),
        ("dataset_from_index", pa.array(starts.tolist(), pa.int64())),
        ("dataset_to_index", pa.array((starts + np.array(lengths)).tolist(), pa.int64())),
    ]
    if spec["camera_dtype"] == "video":
        # every camera sits in one video file (file-000); timestamps accumulate over the episodes in that file.
        # Real exports roll a camera over to file-001 once video_files_size_in_mb is exceeded (each camera
        # independently), and from_timestamp restarts at 0.0 in the new file -- a from/to cross-check must key the
        # cumulative sum on (camera, chunk_index, file_index), not on the dataset start. Not modelled here.
        from_ts = [float(s) / FPS for s in starts]
        to_ts = [float(s + length) / FPS for s, length in zip(starts, lengths)]
        for cam in spec["cameras"]:
            cols += [
                ("videos/%s/chunk_index" % cam, pa.array([0] * n, pa.int64())),
                ("videos/%s/file_index" % cam, pa.array([0] * n, pa.int64())),
                ("videos/%s/from_timestamp" % cam, pa.array(from_ts, pa.float64())),
                ("videos/%s/to_timestamp" % cam, pa.array(to_ts, pa.float64())),
            ]
    for feat in feature_dict(spec):
        for key in STAT_KEYS:
            values = [per_episode_stats[e][feat][key] for e in range(n)]
            if key == "count":
                typ = pa.list_(pa.int64())
            elif feat in spec["cameras"]:
                typ = pa.list_(pa.list_(pa.list_(pa.float64())))
            else:
                typ = pa.list_(pa.float64())
            cols.append(("stats/%s/%s" % (feat, key), pa.array(values, typ)))
    cols += [("meta/episodes/chunk_index", pa.array([0] * n, pa.int64())),
             ("meta/episodes/file_index", pa.array([0] * n, pa.int64()))]
    return pa.table(dict(cols))


# --------------------------------------------------------------------------------------------------------------
# writers
# --------------------------------------------------------------------------------------------------------------
def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def _write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=4) + "\n")


def _write_parquet(path, table):
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def _write_parquet_by_episode(path, table):
    """One row group per episode, like lerobot's write_table_one_row_group_per_episode and the recorder export."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ep = table.column("episode_index").to_numpy()
    bounds = (np.flatnonzero(np.diff(ep)) + 1).tolist()
    with pq.ParquetWriter(path, table.schema) as w:
        for s, e in zip([0] + bounds, bounds + [table.num_rows]):
            w.write_table(table.slice(s, e - s))


def write_host_v3(root, spec, episodes):
    n = spec["n_episodes"]
    first = first_file_episodes(n)
    split = [list(range(0, first)), list(range(first, n))]
    file_of_episode = {e: (0 if e < first else 1) for e in range(n)}
    all_frames = frames_of(episodes, list(range(n)), 0)
    g = 0
    for file_index, eps in enumerate(split):
        frames = frames_of(episodes, eps, g)
        g += frames["index"].shape[0]
        _write_parquet_by_episode(root / "data" / "chunk-000" / ("file-%03d.parquet" % file_index),
                                  data_table(spec, frames))
    per_episode_stats = []
    for e in range(n):
        f = frames_of(episodes, [e], int(all_frames["index"][all_frames["episode_index"] == e][0]))
        per_episode_stats.append(all_feature_stats(spec, f))
    _write_parquet(root / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
                   episodes_table(spec, episodes, file_of_episode, per_episode_stats))
    if spec["write_tasks"]:
        _write_parquet(root / "meta" / "tasks.parquet", tasks_table())
    if spec["write_stats"]:
        _write_json(root / "meta" / "stats.json", dataset_stats(spec, all_frames))
    total_frames = int(all_frames["index"].shape[0])
    info = {
        "codebase_version": "v3.0",
        "robot_type": spec["robot_type"],
        "total_episodes": n,
        "total_frames": total_frames,
        "total_tasks": 1,
        "chunks_size": 1000,
        "data_files_size_in_mb": 100,
        "video_files_size_in_mb": 200,
        "fps": FPS,
        "splits": {"train": "0:%d" % n},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": ("videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
                       if spec["camera_dtype"] == "video" else None),
        "features": feature_dict(spec),
    }
    _write_json(root / "meta" / "info.json", info)
    if spec["camera_dtype"] == "video":
        for cam in spec["cameras"]:
            _touch(root / "videos" / cam / "chunk-000" / "file-000.mp4")
    return root


def write_host_v21(root, spec, episodes):
    """LeRobot v2.1 layout: jsonl metadata, one parquet and one mp4 per episode and camera."""
    n = spec["n_episodes"]
    all_frames = frames_of(episodes, list(range(n)), 0)
    g = 0
    ep_lines, ep_stat_lines = [], []
    for e in range(n):
        frames = frames_of(episodes, [e], g)
        g += frames["index"].shape[0]
        _write_parquet(root / "data" / "chunk-000" / ("episode_%06d.parquet" % e), data_table(spec, frames))
        ep_lines.append({"episode_index": e, "tasks": [TASK], "length": int(frames["index"].shape[0])})
        ep_stat_lines.append({"episode_index": e, "stats": {
            f: {k: v for k, v in st.items() if k in V21_STAT_KEYS} for f, st in all_feature_stats(spec, frames).items()}})
        for cam in spec["cameras"]:
            _touch(root / "videos" / "chunk-000" / cam / ("episode_%06d.mp4" % e))
    meta = root / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / "episodes.jsonl").write_text("".join(json.dumps(l) + "\n" for l in ep_lines))
    (meta / "episodes_stats.jsonl").write_text("".join(json.dumps(l) + "\n" for l in ep_stat_lines))
    if spec["write_tasks"]:
        (meta / "tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": TASK}) + "\n")
    # no meta/stats.json: v2.1 kept per-episode stats only (episodes_stats.jsonl); stats.json returned with v3.0
    info = {
        "codebase_version": "v2.1",
        "robot_type": spec["robot_type"],
        "total_episodes": n,
        "total_frames": int(all_frames["index"].shape[0]),
        "total_tasks": 1,
        "total_videos": n * len(spec["cameras"]),
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": FPS,
        "splits": {"train": "0:%d" % n},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": feature_dict(spec, v21=True),
    }
    _write_json(meta / "info.json", info)
    return root


def write_container(root, spec, episodes):
    """Record the same episodes through lerobot so the videos are real h264 files. Runs inside the training image."""
    from lerobot.configs.video import RGBEncoderConfig  # lazy: only available inside the training image
    from lerobot.datasets.lerobot_dataset import LeRobotDataset  # lazy: only available inside the training image

    features = {k: v for k, v in feature_dict(spec, with_video_info=False).items()
                if k in ("action", "observation.state") or k in spec["cameras"]}
    height, width = spec["cam_shape"][0], spec["cam_shape"][1]
    # lerobot 0.6.1 validates vcodec against ['auto', 'h264', 'hevc', ...] and REJECTS 'libx264'; 'h264' is the name
    # that selects the libx264 encoder through PyAV, and info.json then records video.codec "h264" like the recorder.
    ds = LeRobotDataset.create(repo_id=REPO_ID, fps=FPS, features=features, root=root, robot_type=spec["robot_type"],
                               use_videos=True, rgb_encoder=RGBEncoderConfig(vcodec="h264"))
    for state, action in episodes:
        for t in range(state.shape[0]):
            frame = {"action": action[t], "observation.state": state[t], "task": TASK}
            for ci, cam in enumerate(spec["cameras"]):
                frame[cam] = synthetic_frame(height, width, t, ci)
            ds.add_frame(frame)
        ds.save_episode()
    ds.finalize()
    return root


# --------------------------------------------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------------------------------------------
def make_mini_v3(root, *variants, variant=None, episodes=None, frames=None, seed=0, mode="host", height=None,
                 width=None):
    """Build a synthetic dataset at `root` and return it as a pathlib.Path. See the module docstring.

    `variant="name"` is appended to the positional `*variants` ("default" adds nothing in either form); `frames=N` pins
    every episode to N frames instead of the seeded lengths.
    """
    root = pathlib.Path(root)
    variants = [v for v in variants if v != "default"]
    if variant is not None:
        if not isinstance(variant, str):
            raise ValueError("variant must be a variant name or 'default', got %r" % (variant,))
        if variant != "default":
            variants.append(variant)
    if root.exists() and not root.is_dir():
        raise FileExistsError("%s exists and is not a directory" % root)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("refusing to write into a non-empty directory: %s" % root)
    if mode not in ("host", "container"):
        raise ValueError("mode must be 'host' or 'container', got %r" % (mode,))
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an int for a reproducible fixture, got %r" % (seed,))
    spec = resolve_spec(variants, episodes=episodes, frames=frames, height=height, width=width)
    if mode == "container":  # every container-mode failure must happen here, BEFORE the empty root is removed
        cam_h, cam_w = spec["cam_shape"][0], spec["cam_shape"][1]
        if cam_h % 2 or cam_w % 2:
            raise ValueError("container mode (libx264, yuv420p) needs EVEN height and width, got %dx%d" % (cam_h, cam_w))
        bad = sorted(set(spec["variants"]) - set(CONTAINER_VARIANTS))
        if bad:
            raise ValueError("variant(s) %s are host-only (no real video to record); container mode supports: %s"
                             % (bad, ", ".join(CONTAINER_VARIANTS)))
        import lerobot.datasets.lerobot_dataset  # noqa: F401  fail with ModuleNotFoundError before touching disk
    eps = generate_episodes(spec, int(seed))
    if mode == "container":
        if root.exists():  # empty (checked above); LeRobotDataset.create insists on mkdir(exist_ok=False)
            try:
                root.rmdir()
            except OSError as e:  # e.g. EBUSY: the empty root is itself a bind-mount point
                raise ValueError("container mode needs a path that does not exist yet (lerobot creates it); "
                                 "could not remove empty %s: %s" % (root, e))
        return write_container(root, spec, eps)
    root.mkdir(parents=True, exist_ok=True)
    if spec["layout"] == "v2.1":
        return write_host_v21(root, spec, eps)
    return write_host_v3(root, spec, eps)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("out_dir", nargs="?", help="directory to create (must not exist or be empty)")
    p.add_argument("--variant", action="append", default=[], metavar="NAME",
                   help="repeatable; see --list-variants ('default' names the baseline and adds nothing)")
    p.add_argument("--episodes", type=int, default=None,
                   help="number of episodes (default %d; few_episodes fixes %d)" % (DEFAULT_EPISODES, FEW_EPISODES))
    p.add_argument("--frames", type=int, default=None,
                   help="pin every episode to N frames (>= %d); default: seeded lengths in [%d, %d]" % (MIN_FRAMES, MIN_LEN, MAX_LEN))
    p.add_argument("--seed", type=int, default=0, help="RNG seed (default 0); the same seed rebuilds identical tables")
    p.add_argument("--container", action="store_true", help="record through lerobot (inside the training image)")
    p.add_argument("--height", type=int, default=None,
                   help="override the camera height (give --width too; container mode needs even values)")
    p.add_argument("--width", type=int, default=None,
                   help="override the camera width (give --height too; container mode needs even values)")
    p.add_argument("--list-variants", action="store_true", help="print the variant names one per line and exit")
    a = p.parse_args(argv)
    if a.list_variants:
        for name in VARIANTS:
            print(name)
        return 0
    if not a.out_dir:
        p.error("OUT_DIR is required unless --list-variants is given")
    try:
        out = make_mini_v3(a.out_dir, *a.variant, episodes=a.episodes, frames=a.frames, seed=a.seed,
                           mode="container" if a.container else "host", height=a.height, width=a.width)
    except (ValueError, FileExistsError) as e:
        p.error(str(e))  # one line + usage, exit 2, instead of a traceback
    except ModuleNotFoundError as e:
        p.error("--container must run inside the training image (python with lerobot): %s" % e)
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
