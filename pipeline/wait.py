# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Poll a training job to completion (wait loop adapted from https://github.com/aws-samples/sample-dreamzero-finetuning-on-sagemaker).

A fresh client per poll survives credential rotation; a Stopped job is explained (our own StoppingCondition vs an
external StopTrainingJob); every SecondaryStatus transition is printed once.
"""
from __future__ import annotations

import sys
import time
from typing import Callable

import botocore.exceptions

TERMINAL = ("Completed", "Failed", "Stopped")


class JobFailed(RuntimeError):
    def __init__(self, job: str, describe: dict, why: str):
        super().__init__(f"{job} ended {describe.get('TrainingJobStatus')}/{describe.get('SecondaryStatus')}: {why}")
        self.job, self.describe = job, describe


def stopped_reason(d: dict) -> str:
    sec = d.get("SecondaryStatus", "-")
    if d.get("FailureReason"):
        return d["FailureReason"]
    if sec == "MaxWaitTimeExceeded":
        return ("MaxWaitTimeExceeded: no Spot capacity for this instance type arrived within 2 x the runtime cap (nothing was "
                "billed). Try again later, another instance type, another region, or spot: false (on-demand quota needed).")
    if sec == "MaxRuntimeExceeded":
        return ("MaxRuntimeExceeded: this pipeline's own StoppingCondition fired (cap derived from the smoke's step rate x "
                "compute.runtime_multiplier). Raise runtime_multiplier in project_config.json and resume with --resume-from.")
    return ("stopped via the API, not by a training error and not by our runtime cap; check CloudTrail for "
            "StopTrainingJob (account clean-up tooling is a common cause; see extra_job_tags).")


def wait_for_job(make_client: Callable[[], object], job: str, *, poll_s: int = 30, log=print,
                 on_transition: Callable[[dict], None] = None, on_poll: Callable[[dict], None] = None) -> dict:
    """Return the final describe_training_job dict; raise JobFailed for Failed/Stopped."""
    seen, errs = 0, 0
    while True:
        try:
            d = make_client().describe_training_job(TrainingJobName=job)
            errs = 0
        except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
            code = getattr(e, "response", {}).get("Error", {}).get("Code", "")
            if code == "ValidationException":
                raise JobFailed(job, {}, f"SageMaker does not recognise job {job}: {e}") from None
            errs += 1
            if errs == 1 or errs % 10 == 0:
                log(f"  describe_training_job error ({errs}, retrying): {e} -- refresh credentials if this persists")
            time.sleep(min(poll_s, 60))
            continue
        trans = d.get("SecondaryStatusTransitions", [])
        for t in trans[seen:]:
            log(f"  [{t['StartTime']:%H:%M:%S}] {t['Status']}: {t.get('StatusMessage', '')}")
            if on_transition:
                on_transition(t)
        seen = len(trans)
        if on_poll:
            try:
                on_poll(d)
            except Exception as e:  # a log-tail hiccup must never break the wait
                log(f"  (tail: {type(e).__name__})")
        status = d["TrainingJobStatus"]
        if status in TERMINAL:
            if status != "Completed":
                raise JobFailed(job, d, stopped_reason(d))
            log(f"  {job}: Completed, billable {d.get('BillableTimeInSeconds', '?')} s")
            return d
        time.sleep(poll_s)


def main(argv=None) -> int:  # `python3 -m pipeline.wait <job> --region R [--profile P]`: reattach to a running job
    import argparse
    from pipeline.aws import client, session
    ap = argparse.ArgumentParser()
    ap.add_argument("job"); ap.add_argument("--region", required=True); ap.add_argument("--profile")
    a = ap.parse_args(argv)
    sess = session(a.region, a.profile)
    try:
        wait_for_job(lambda: client(sess, "sagemaker"), a.job)
    except JobFailed as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 6
    return 0


if __name__ == "__main__":
    sys.exit(main())
