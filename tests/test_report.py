# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from pipeline import report  # noqa: E402

REF = report.load_reference("act", "pick-and-place-v3.0")


def _summary(j, jh=0.06367, g=0.0016, gh=0.00393, timing="16/16", beats="14/16"):
    return {"evals": {"eval_last.json": {"joints_mae": j, "joints_hold": jh, "grippers_mae": g, "grippers_hold": gh,
                                         "gripper_timing": timing, "beats_hold": beats}}}


def test_reference_runs_pass_the_band():
    assert report.verdict(_summary(0.04271), REF, "pick-and-place-v3.0")[0] == "PASS"
    assert report.verdict(_summary(0.04323, g=0.00158), REF, "pick-and-place-v3.0")[0] == "PASS"


def test_repeat_and_fail_zones():
    assert report.verdict(_summary(0.0440), REF, "pick-and-place-v3.0")[0] == "REPEAT"
    assert report.verdict(_summary(0.0460), REF, "pick-and-place-v3.0")[0] == "FAIL"
    v, reasons = report.verdict(_summary(0.0430, timing="11/16", beats="8/16"), REF, "pick-and-place-v3.0")
    assert v == "FAIL" and any("timing" in r for r in reasons) and any("beat hold" in r for r in reasons)
    near = _summary(0.04315, g=0.00164, beats="13/16"); near["per_dim"] = {"all": {"d0": {"model": 0.0234, "hold": 0.0229}, "d1": {"model": 0.03, "hold": 0.06}}}
    assert report.verdict(near, REF, "pick-and-place-v3.0")[0] == "REPEAT"          # one dim at 1.02x hold -> repeat, not fail
    far = _summary(0.04315, beats="13/16"); far["per_dim"] = {"all": {"d0": {"model": 0.05, "hold": 0.02}}}
    assert report.verdict(far, REF, "pick-and-place-v3.0")[0] == "FAIL"             # a dim at 2.5x hold is a real miss
    v, reasons = report.verdict(_summary(0.030), REF, "pick-and-place-v3.0")
    assert v == "PASS" and any("leakage" in r for r in reasons)


def test_other_dataset_has_no_verdict_and_table_renders():
    v, _ = report.verdict(_summary(18.4), REF, "svla_so101_pickplace")
    assert v == "n/a"
    pub = report.load_reference("act", "svla_so101_pickplace")
    assert pub["reference_runs"][0]["beats_hold"] == "6/6" and pub.get("band") is None
    assert report.verdict(_summary(4.3), pub, "svla_so101_pickplace")[0] == "n/a"
    assert report.load_reference("act", "unknown") == {} and report.load_reference("nope", "x") == {}
    t = report.table(_summary(0.04271), REF)
    assert "this run" in t and "act_c24" in t and "0.04271" in t


def test_manifest_and_compare(tmp_path):
    run = {"name": "r1", "policy": "act", "dataset_s3": "s3://b/x/", "dataset_fingerprint": "abc", "summary": _summary(0.0427),
           "instance_type": "ml.g6e.4xlarge", "billable_seconds": 100, "verdict": "PASS"}
    d = {"TrainingJobName": "j", "TrainingJobStatus": "Completed", "ResourceConfig": {"InstanceType": "ml.g6e.4xlarge"}, "BillableTimeInSeconds": 100,
         "EnableNetworkIsolation": True, "AlgorithmSpecification": {"TrainingImage": "x@sha256:aa"}, "HyperParameters": {"steps": "100"},
         "ModelArtifacts": {"S3ModelArtifacts": "s3://b/j/output/model.tar.gz"}}
    m = report.manifest(run, d, run["summary"])
    assert m["network_isolation"] is True and m["image"].endswith("aa") and m["repo_git_sha"]
    p = tmp_path / "r1.json"; p.write_text(json.dumps(run))
    out = report.compare([str(p)])
    assert "r1" in out and "PASS" in out


def test_stride_other_than_one_gets_no_verdict():
    s = _summary(0.03989, beats="13/16")
    s["evals"]["eval_last.json"]["stride"] = 60
    v, notes = report.verdict(s, REF, "pick-and-place-v3.0")
    assert v == "n/a" and "stride 60" in notes[0]
    s["evals"]["eval_last.json"]["stride"] = 1
    assert report.verdict(s, REF, "pick-and-place-v3.0")[0] in ("PASS", "REPEAT", "FAIL")
