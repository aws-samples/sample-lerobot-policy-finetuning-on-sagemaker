# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
import datetime as dt
import sys
from pathlib import Path

import botocore.exceptions
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline import wait  # noqa: E402


class FakeSM:
    def __init__(self, responses):
        self.responses = list(responses)

    def describe_training_job(self, TrainingJobName):
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def _d(status, secondary, **kw):
    d = {"TrainingJobStatus": status, "SecondaryStatus": secondary,
         "SecondaryStatusTransitions": [{"Status": secondary, "StartTime": dt.datetime(2026, 9, 10, 12, 0), "StatusMessage": "m"}]}
    d.update(kw)
    return d


def test_completes_after_transitions(monkeypatch):
    monkeypatch.setattr(wait.time, "sleep", lambda s: None)
    sm = FakeSM([_d("InProgress", "Starting"), _d("InProgress", "Training"), _d("Completed", "Completed", BillableTimeInSeconds=42)])
    out = wait.wait_for_job(lambda: sm, "job", log=lambda m: None)
    assert out["BillableTimeInSeconds"] == 42


def test_stopped_by_our_cap_is_explained(monkeypatch):
    monkeypatch.setattr(wait.time, "sleep", lambda s: None)
    sm = FakeSM([_d("Stopped", "MaxRuntimeExceeded")])
    with pytest.raises(wait.JobFailed, match="runtime_multiplier"):
        wait.wait_for_job(lambda: sm, "job", log=lambda m: None)


def test_external_stop_and_failure_reason(monkeypatch):
    monkeypatch.setattr(wait.time, "sleep", lambda s: None)
    with pytest.raises(wait.JobFailed, match="StopTrainingJob"):
        wait.wait_for_job(lambda: FakeSM([_d("Stopped", "Stopped")]), "job", log=lambda m: None)
    with pytest.raises(wait.JobFailed, match="AlgorithmError"):
        wait.wait_for_job(lambda: FakeSM([_d("Failed", "Failed", FailureReason="AlgorithmError: boom")]), "job", log=lambda m: None)


def test_transient_errors_retry_but_unknown_job_aborts(monkeypatch):
    monkeypatch.setattr(wait.time, "sleep", lambda s: None)
    expired = botocore.exceptions.ClientError({"Error": {"Code": "ExpiredTokenException", "Message": "x"}}, "DescribeTrainingJob")
    sm = FakeSM([expired, expired, _d("Completed", "Completed")])
    assert wait.wait_for_job(lambda: sm, "job", log=lambda m: None)["TrainingJobStatus"] == "Completed"
    unknown = botocore.exceptions.ClientError({"Error": {"Code": "ValidationException", "Message": "no such job"}}, "DescribeTrainingJob")
    with pytest.raises(wait.JobFailed, match="does not recognise"):
        wait.wait_for_job(lambda: FakeSM([unknown]), "job", log=lambda m: None)
