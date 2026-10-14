# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""pipeline/smoke_aws.py: the verdict logic and the opt-in joints-MAE fingerprint flags (no AWS calls)."""
import math

import pytest

from pipeline import smoke_aws


def test_default_verdict_only_requires_a_finite_mae():
    assert smoke_aws.judge(17.3817) == (True, "finite")
    assert smoke_aws.judge(12345.6)[0] is True
    assert smoke_aws.judge(None)[0] is False
    assert smoke_aws.judge(math.nan)[0] is False
    assert smoke_aws.judge(math.inf)[0] is False
    assert smoke_aws.judge("17.3817")[0] is True   # SUMMARY.json may carry the number as a string


def test_expected_mae_passes_within_tolerance_and_fails_outside():
    ok, why = smoke_aws.judge(17.3817, expect=17.3817)
    assert ok and "matches" in why
    assert smoke_aws.judge(17.38175, expect=17.3817, tol=1e-4)[0] is True
    ok, why = smoke_aws.judge(17.39, expect=17.3817, tol=1e-4)
    assert not ok and "differs" in why and "ACT path changed" in why
    assert smoke_aws.judge(17.39, expect=17.3817, tol=0.05)[0] is True
    assert smoke_aws.judge(math.nan, expect=17.3817)[0] is False


def test_fingerprint_flags_are_opt_in_and_documented():
    ap = smoke_aws.build_parser()
    a = ap.parse_args([])
    assert a.expect_joints_mae is None and a.mae_tol == 1e-4
    a = ap.parse_args(["--expect-joints-mae", "17.3817", "--mae-tol", "0.001"])
    assert a.expect_joints_mae == 17.3817 and a.mae_tol == 0.001
    help_text = ap.format_help()
    assert "--expect-joints-mae" in help_text and "--mae-tol" in help_text and "17.3817" in help_text


def test_negative_tolerance_is_rejected():
    with pytest.raises(SystemExit):
        smoke_aws.main(["--region", "us-east-1", "--mae-tol", "-1"])
