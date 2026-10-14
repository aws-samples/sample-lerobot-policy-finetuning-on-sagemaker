# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Shared fixtures: the synthetic LeRobot v3.0 dataset builder and the optional docker lane.

tests/ and tests/fixtures/ are packages, so the generator is imported as `tests.fixtures.make_mini_v3` -- the SAME
module object the test modules import directly (no importlib-by-path double load). The file still runs stand-alone
on a recorder PC and inside the training image; it imports only stdlib + pyarrow + numpy. The docker lane skips --
never errors -- when `docker` is missing, the docker CLI does not answer, or the training image is not present, so
`pytest -m docker` is safe on any box; the default lane (`-m "not docker and not aws"`) never spawns `docker` at all.
"""
import functools
import itertools
import os
import pathlib
import shutil
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import tests.fixtures.make_mini_v3 as make_mini_v3_module_  # noqa: E402  (REPO on sys.path first)

TRAIN_IMAGE = os.environ.get("LEROBOT_TRAIN_IMAGE", "act-sagemaker-training:dev")


@functools.lru_cache(maxsize=None)  # one probe per session, shared by the collection hook and the train_image fixture
def docker_lane_unavailable():
    """Reason string when the docker lane cannot run, else None."""
    if shutil.which("docker") is None:
        return "docker is not on PATH"
    try:
        r = subprocess.run(["docker", "image", "inspect", TRAIN_IMAGE], capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired:
        return "docker did not answer within 30 s (daemon unreachable?)"
    if r.returncode != 0:  # docker says why: "No such image" vs "Cannot connect to the Docker daemon"
        detail = ((r.stderr or r.stdout).strip().splitlines() or ["no output"])[-1]
        return "docker image inspect %s failed: %s (start the daemon or set LEROBOT_TRAIN_IMAGE)" % (TRAIN_IMAGE, detail)
    return None


@pytest.hookimpl(trylast=True)  # after pytest's own -m deselection, so the default host lane never shells out to docker
def pytest_collection_modifyitems(config, items):
    docker_items = [it for it in items if it.get_closest_marker("docker") is not None]
    if not docker_items:
        return
    reason = docker_lane_unavailable()
    if reason:
        for it in docker_items:
            it.add_marker(pytest.mark.skip(reason=reason))


@pytest.fixture(scope="session")
def make_mini_v3_module():
    """The generator module (tests.fixtures.make_mini_v3), shared with every `from tests.fixtures.make_mini_v3 import`."""
    return make_mini_v3_module_


@pytest.fixture(scope="session")
def mini_v3(tmp_path_factory, make_mini_v3_module):
    """Baseline synthetic v3.0 dataset, built once per session (host mode, zero-byte videos)."""
    return make_mini_v3_module.make_mini_v3(tmp_path_factory.mktemp("mini_v3") / "mini")


@pytest.fixture
def mini_v3_factory(tmp_path, make_mini_v3_module):
    """Callable (*variants, **kw) -> Path; each call builds into a fresh directory under tmp_path."""
    counter = itertools.count()

    def build(*variants, **kw):
        return make_mini_v3_module.make_mini_v3(tmp_path / ("mini_%d" % next(counter)), *variants, **kw)

    return build


@pytest.fixture
def train_image():
    """Name of the training image for docker-marked tests (skips when the lane is unavailable)."""
    reason = docker_lane_unavailable()
    if reason:
        pytest.skip(reason)
    return TRAIN_IMAGE
