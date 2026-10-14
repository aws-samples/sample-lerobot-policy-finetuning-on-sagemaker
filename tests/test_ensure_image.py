# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""run.py follows a running image build instead of failing with "no image" (first deploy) or silently training on the
previous image (redeploy after a trainer/ change); dev builds that do not publish are ignored."""
import datetime as dt
import sys
from pathlib import Path

import pytest
from botocore.exceptions import ClientError

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from pipeline import build_image  # noqa: E402

STACK = {"bucket": "b", "ecr_repo_uri": "123456789012.dkr.ecr.us-east-1.amazonaws.com/p-sagemaker-training"}
OLD = STACK["ecr_repo_uri"] + "@sha256:" + "a" * 64
NEW = STACK["ecr_repo_uri"] + "@sha256:" + "b" * 64
T0 = dt.datetime(2026, 9, 17, 10, 0, tzinfo=dt.timezone(dt.timedelta(hours=9)))   # 01:00 UTC, in a non-UTC zone


class Sess:
    region_name = "us-east-1"


def _build(bid, status, publish="1"):
    return {"id": bid, "buildStatus": status, "currentPhase": "BUILD" if status == "IN_PROGRESS" else "COMPLETED", "startTime": T0,
            "environment": {"environmentVariables": [{"name": "PUBLISH_SSM", "value": publish}, {"name": "IMAGE_TAG", "value": "v1"}]}}


class FakeCodeBuild:
    """`builds` is newest first; `later` are the statuses the newest publishing build reports on later polls."""
    def __init__(self, builds, later=()):
        self.builds, self.later, self.calls, self.polled = builds, list(later), 0, []
        self.denied = False
    def list_builds_for_project(self, projectName, sortOrder):
        assert projectName == "p-image-build" and sortOrder == "DESCENDING"
        self.calls += 1
        if self.denied:
            raise ClientError({"Error": {"Code": "AccessDeniedException", "Message": "no"}}, "ListBuildsForProject")
        return {"ids": [b["id"] for b in self.builds]}
    def batch_get_builds(self, ids):
        self.calls += 1
        if len(ids) == 1 and ids[0] in self.polled:      # a poll from wait_build
            st = self.later.pop(0) if len(self.later) > 1 else self.later[0]
            return {"builds": [{**next(b for b in self.builds if b["id"] == ids[0]), "buildStatus": st, "currentPhase": "COMPLETED" if st != "IN_PROGRESS" else "BUILD"}]}
        self.polled.extend(ids)
        return {"builds": [b for b in reversed(self.builds) if b["id"] in ids]}   # deliberately not in ids order


def _patch(monkeypatch, cb, published_after):
    monkeypatch.setattr(build_image, "client", lambda sess, name: cb if name == "codebuild" else None)
    monkeypatch.setattr(build_image.time, "sleep", lambda s: None)
    return lambda sess, project, use_cache: {**STACK, **({"image_uri": published_after} if published_after else {})}


def test_first_deploy_waits_for_the_kickoff_build_then_reads_the_stack(monkeypatch):
    cb = FakeCodeBuild([_build("p-image-build:1", "IN_PROGRESS")], later=["IN_PROGRESS", "SUCCEEDED"])
    lines = []
    out = build_image.ensure_image(Sess(), dict(STACK), "p", log=lines.append, poll_s=0, resolve=_patch(monkeypatch, cb, NEW))
    assert out["image_uri"] == NEW
    assert any("deploy-time image build is still running" in l and "01:00 UTC" in l and "codesuite/codebuild" in l for l in lines), lines


def test_redeploy_with_a_running_publishing_build_waits_even_though_an_old_image_is_published(monkeypatch):
    cb = FakeCodeBuild([_build("p-image-build:2", "IN_PROGRESS"), _build("p-image-build:1", "SUCCEEDED")], later=["SUCCEEDED"])
    lines = []
    out = build_image.ensure_image(Sess(), {**STACK, "image_uri": OLD}, "p", log=lines.append, poll_s=0, resolve=_patch(monkeypatch, cb, NEW))
    assert out["image_uri"] == NEW and any("a new image build is still running" in l for l in lines)


def test_dev_builds_are_skipped_in_favour_of_the_newest_publishing_build(monkeypatch):
    cb = FakeCodeBuild([_build("p-image-build:3", "SUCCEEDED", publish="0"), _build("p-image-build:2", "FAILED", publish="0"),
                        _build("p-image-build:1", "IN_PROGRESS")], later=["SUCCEEDED"])
    out = build_image.ensure_image(Sess(), dict(STACK), "p", log=lambda m: None, poll_s=0, resolve=_patch(monkeypatch, cb, NEW))
    assert out["image_uri"] == NEW and "p-image-build:1" in cb.polled


def test_published_image_with_nothing_running_is_used_as_is(monkeypatch):
    cb = FakeCodeBuild([_build("p-image-build:1", "SUCCEEDED")])
    resolve = _patch(monkeypatch, cb, NEW)   # would hand out NEW if re-read: must not be called
    out = build_image.ensure_image(Sess(), {**STACK, "image_uri": OLD}, "p", log=lambda m: None, resolve=resolve)
    assert out["image_uri"] == OLD
    lines = []
    cb = FakeCodeBuild([_build("p-image-build:2", "FAILED"), _build("p-image-build:1", "SUCCEEDED")])
    out = build_image.ensure_image(Sess(), {**STACK, "image_uri": OLD}, "p", log=lines.append, resolve=_patch(monkeypatch, cb, NEW))
    assert out["image_uri"] == OLD and any("WARN image: the newest image build FAILED" in l for l in lines)
    cb = FakeCodeBuild([])
    assert build_image.ensure_image(Sess(), {**STACK, "image_uri": OLD}, "p", log=lambda m: None, resolve=_patch(monkeypatch, cb, NEW))["image_uri"] == OLD


def test_no_codebuild_permission_keeps_the_published_image_but_fails_without_one(monkeypatch):
    cb = FakeCodeBuild([_build("p-image-build:1", "SUCCEEDED")]); cb.denied = True
    lines = []
    out = build_image.ensure_image(Sess(), {**STACK, "image_uri": OLD}, "p", log=lines.append, resolve=_patch(monkeypatch, cb, NEW))
    assert out["image_uri"] == OLD and any("AccessDeniedException" in l for l in lines)
    with pytest.raises(ClientError):
        build_image.ensure_image(Sess(), dict(STACK), "p", log=lambda m: None, resolve=_patch(monkeypatch, cb, NEW))


def test_failed_missing_and_unpublished_builds_are_readable_errors(monkeypatch):
    with pytest.raises(RuntimeError, match="image build FAILED.*cdk deploy"):
        build_image.ensure_image(Sess(), dict(STACK), "p", log=lambda m: None, resolve=_patch(monkeypatch, FakeCodeBuild([_build("p-image-build:1", "FAILED")]), None))
    with pytest.raises(RuntimeError, match="no publishing image build"):
        build_image.ensure_image(Sess(), dict(STACK), "p", log=lambda m: None, resolve=_patch(monkeypatch, FakeCodeBuild([_build("p-image-build:1", "SUCCEEDED", publish="0")]), None))
    with pytest.raises(RuntimeError, match="still unset"):
        build_image.ensure_image(Sess(), dict(STACK), "p", log=lambda m: None, resolve=_patch(monkeypatch, FakeCodeBuild([_build("p-image-build:1", "SUCCEEDED")]), None))


def test_run_stages_check_the_dataset_before_the_stack():
    from pipeline import run
    assert run.STAGES.index("dataset") < run.STAGES.index("stack") < run.STAGES.index("preflight")
