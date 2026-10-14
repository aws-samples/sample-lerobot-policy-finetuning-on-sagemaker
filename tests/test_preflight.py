# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "trainer"))
from pipeline import instance_table, logs, preflight  # noqa: E402
import render_flags  # noqa: E402

CFG = {"policies": {"act": {"enabled": True, "accept_licence": False}}, "compute": {"instance_type": "ml.g6e.4xlarge"}}
POLICIES = {"act": render_flags.load_policy("act")}
STACK = {"bucket": "b", "ecr_repo_uri": "123456789012.dkr.ecr.ap-northeast-1.amazonaws.com/x-sagemaker-training", "region": "ap-northeast-1"}
IMG = STACK["ecr_repo_uri"] + "@sha256:" + "a" * 64


def test_instance_table_rules():
    assert instance_table.spec("ml.g6e.4xlarge").gpus == 1 and instance_table.spec("ml.g6e.12xlarge").gpus == 4
    fails, _ = instance_table.problems("ml.g6e.12xlarge"); assert fails and "one process on one GPU" in fails[0]
    assert instance_table.problems("ml.g6e.12xlarge", allow_multigpu=True)[0] == []
    _, warns = instance_table.problems("ml.g6e.16xlarge"); assert any("same step rate" in w for w in warns)
    _, warns = instance_table.problems("ml.g6e.2xlarge"); assert any("dataloader workers" in w for w in warns)
    assert instance_table.problems("ml.c5.4xlarge") == ([], [])
    assert instance_table.step_rate(POLICIES["act"], "ml.g6e.4xlarge") == 10.0 and instance_table.step_rate(POLICIES["act"], "ml.g4dn.2xlarge") == 1.74
    assert instance_table.quota_names("ml.g6e.4xlarge", True) == ["ml.g6e.4xlarge for spot training job usage"]
    for it in POLICIES["act"]["hardware"]["instance_types"]:
        assert instance_table.spec(it) is not None, it


def test_local_checks_refuse_before_any_aws_call():
    class Boom:
        region_name = "ap-northeast-1"
    def client(name):
        raise AssertionError("AWS was called although local checks failed")
    cfg = {"policies": {"act": {"enabled": True}, "xvla": {"enabled": True, "accept_licence": False}}}
    pols = dict(POLICIES); xv = dict(POLICIES["act"]); xv["name"] = "xvla"; xv["licence"] = {**xv["licence"], "class": "terms"}; pols["xvla"] = xv
    with pytest.raises(preflight.PreflightError):
        preflight.run_preflight(cfg, STACK, Boom(), pols, candidates=["ml.g6e.12xlarge"], image_uri=IMG, log=lambda m: None, client=client)


class FakeClients:
    def __init__(self, loc="ap-northeast-1", quotas=None, ecr_ok=True, plan=None):
        self.loc, self.quotas, self.ecr_ok, self.plan = loc, quotas or {}, ecr_ok, plan
    def sagemaker(self):
        plan = self.plan
        def describe_training_plan(s, TrainingPlanName):
            if plan is None: raise RuntimeError("ResourceNotFound")
            return {"TrainingPlanName": TrainingPlanName, "TargetResources": ["training-job"], "AvailableInstanceCount": 1, **plan}
        return type("SM", (), {"describe_training_plan": describe_training_plan})()
    def __call__(self, name):
        return getattr(self, name.replace("-", "_"))()
    def sts(self):
        c = self; return type("S", (), {"get_caller_identity": lambda s: {"Arn": "arn:aws:sts::123456789012:assumed-role/x/y"}})()
    def s3(self):
        loc = self.loc; return type("S3", (), {"get_bucket_location": lambda s, Bucket: {"LocationConstraint": loc}})()
    def ecr(self):
        ok = self.ecr_ok
        def describe_images(s, **kw):
            if not ok: raise RuntimeError("ImageNotFoundException")
            return {"imageDetails": [{}]}
        return type("E", (), {"describe_images": describe_images})()
    def service_quotas(self):
        q = self.quotas
        class P:
            def paginate(s, **kw): yield {"Quotas": [{"QuotaName": k, "Value": v} for k, v in q.items()]}
        return type("Q", (), {"get_paginator": lambda s, name: P()})()


class Sess:
    region_name = "ap-northeast-1"


def _quotas(**kv):
    return {f"{k.replace('_', '.')} for training job usage": v for k, v in kv.items()}


def test_aws_checks_happy_path():
    q = {"ml.g6e.4xlarge for training job usage": 4, "ml.c5.4xlarge for training job usage": 8}
    rows = preflight.run_preflight(CFG, STACK, Sess(), POLICIES, candidates=["ml.g6e.4xlarge", "ml.c5.4xlarge"], image_uri=IMG,
                                   log=lambda m: None, client=FakeClients(quotas=q))
    assert all(lvl != "fail" for lvl, _, _ in rows) and any(c == "quota" and "= 4" in m for _, c, m in rows)


def test_aws_checks_failures():
    with pytest.raises(preflight.PreflightError):  # bucket elsewhere
        preflight.run_preflight(CFG, STACK, Sess(), POLICIES, candidates=["ml.g6e.4xlarge"], image_uri=IMG, log=lambda m: None, client=FakeClients(loc="eu-central-1"))
    with pytest.raises(preflight.PreflightError):  # quota 0
        preflight.run_preflight(CFG, STACK, Sess(), POLICIES, candidates=["ml.g6e.4xlarge"], image_uri=IMG, log=lambda m: None,
                                client=FakeClients(quotas={"ml.g6e.4xlarge for training job usage": 0}))
    with pytest.raises(preflight.PreflightError):  # spot quota checked under its own name
        preflight.run_preflight(CFG, STACK, Sess(), POLICIES, candidates=["ml.g6e.4xlarge"], spot=True, image_uri=IMG, log=lambda m: None,
                                client=FakeClients(quotas={"ml.g6e.4xlarge for training job usage": 4, "ml.g6e.4xlarge for spot training job usage": 0}))
    with pytest.raises(preflight.PreflightError):  # image missing / not digest pinned
        preflight.run_preflight(CFG, STACK, Sess(), POLICIES, candidates=["ml.g6e.4xlarge"], image_uri=IMG.split("@")[0] + ":v1", log=lambda m: None, client=FakeClients())
    with pytest.raises(preflight.PreflightError):
        preflight.run_preflight(CFG, STACK, Sess(), POLICIES, candidates=["ml.g6e.4xlarge"], image_uri=IMG, log=lambda m: None, client=FakeClients(ecr_ok=False))


def test_log_parsing():
    lines = logs.split_lines("Training:  5%|\r step:900 smpl:7.2K loss:1.5 x\nINFO step:8K smpl:64K ep:0 epch:0.3 loss:0.123 grdn:1 lr:1e-05\r\n")
    m = logs.latest_metric(lines)
    assert m["step"] == 8000 and m["loss"] == 0.123
    assert logs.parse_step("100", "K") == 100000 and logs.parse_step("42") == 42
    assert logs.latest_metric(["no metrics here"]) is None


def test_stub_entries_refused_unless_allowed():
    stub = dict(POLICIES["act"]); stub["status"] = "stub"; stub["stub_reason"] = "not yet"
    cfg = {"policies": {"act": {"enabled": True}}}
    rows = preflight.local_checks(cfg, {"act": stub}, ["ml.g6e.4xlarge"])
    assert any(lvl == "fail" and c == "registry" for lvl, c, _ in rows)
    rows = preflight.local_checks(cfg, {"act": stub}, ["ml.g6e.4xlarge"], allow_stubs=True)
    assert any(lvl == "warn" and c == "registry" for lvl, c, _ in rows) and not any(lvl == "fail" for lvl, _, _ in rows)


def test_spot_run_checks_the_spot_quota_only():
    """The smoke uses the same capacity as the training job, so a Spot run needs only the Spot quota (a Spot-only p5 quota is enough)."""
    rows = preflight.run_preflight(CFG, STACK, Sess(), POLICIES, candidates=["ml.g6e.4xlarge"], spot=True, image_uri=IMG, log=lambda m: None,
                                   client=FakeClients(quotas={"ml.g6e.4xlarge for training job usage": 0, "ml.g6e.4xlarge for spot training job usage": 1}))
    assert [c for _, c, _ in rows if c == "quota"] == ["quota"]
    with pytest.raises(preflight.PreflightError):   # Spot quota 0 while running on Spot
        preflight.run_preflight(CFG, STACK, Sess(), POLICIES, candidates=["ml.g6e.4xlarge"], spot=True, image_uri=IMG, log=lambda m: None,
                                client=FakeClients(quotas={"ml.g6e.4xlarge for training job usage": 4, "ml.g6e.4xlarge for spot training job usage": 0}))


PLAN = "arn:aws:sagemaker:ap-northeast-1:123456789012:training-plan/lerobot-fastwam-smoke"


def test_training_plan_replaces_the_gpu_quota_check():
    p5 = {"Status": "Active", "ReservedCapacitySummaries": [{"InstanceType": "ml.p5.4xlarge"}]}
    quotas = {"ml.p5.4xlarge for training job usage": 0, "ml.c5.4xlarge for training job usage": 8}   # on-demand p5 quota is 0: irrelevant with a plan
    rows = preflight.run_preflight(CFG, STACK, Sess(), POLICIES, candidates=["ml.p5.4xlarge", "ml.c5.4xlarge"], image_uri=IMG, log=lambda m: None,
                                   client=FakeClients(quotas=quotas, plan=p5), training_plan_arn=PLAN)
    assert any(c == "plan" and lvl == "ok" and "Active" in m for lvl, c, m in rows)
    assert any(c == "quota" and "reserved capacity" in m for _, c, m in rows) and any(c == "quota" and "ml.c5.4xlarge" in m and "= 8" in m for _, c, m in rows)


def test_training_plan_failures():
    p5 = {"Status": "Active", "ReservedCapacitySummaries": [{"InstanceType": "ml.p5.4xlarge"}]}
    with pytest.raises(preflight.PreflightError):   # plan not usable yet
        preflight.run_preflight(CFG, STACK, Sess(), POLICIES, candidates=["ml.p5.4xlarge"], image_uri=IMG, log=lambda m: None,
                                client=FakeClients(plan={**p5, "Status": "Pending"}), training_plan_arn=PLAN)
    with pytest.raises(preflight.PreflightError):   # plan reserves another instance type
        preflight.run_preflight(CFG, STACK, Sess(), POLICIES, candidates=["ml.g6e.4xlarge"], image_uri=IMG, log=lambda m: None,
                                client=FakeClients(plan=p5), training_plan_arn=PLAN)
    with pytest.raises(preflight.PreflightError):   # plan in another region
        preflight.run_preflight(CFG, STACK, Sess(), POLICIES, candidates=["ml.p5.4xlarge"], image_uri=IMG, log=lambda m: None,
                                client=FakeClients(plan=p5), training_plan_arn=PLAN.replace("ap-northeast-1", "us-east-1"))
    with pytest.raises(preflight.PreflightError):   # plan does not exist
        preflight.run_preflight(CFG, STACK, Sess(), POLICIES, candidates=["ml.p5.4xlarge"], image_uri=IMG, log=lambda m: None,
                                client=FakeClients(plan=None), training_plan_arn=PLAN)


# ---------------------------------------------------------------- --instance-type auto -------------------------------
def _spec(*types):
    return {**POLICIES["act"], "hardware": {**POLICIES["act"]["hardware"], "instance_types": list(types)}}


class Denied(FakeClients):
    def service_quotas(self):
        raise RuntimeError("AccessDeniedException")


def test_quota_values_reads_one_pass_and_leaves_unlisted_names_absent():
    q = FakeClients(quotas={"ml.g6e.4xlarge for training job usage": 4, "other": 1})
    assert preflight.quota_values(q, {"ml.g6e.4xlarge for training job usage", "ml.g5.4xlarge for training job usage"}) == {"ml.g6e.4xlarge for training job usage": 4}
    assert preflight.quota_values(q, set()) == {}


def test_auto_instance_skips_a_zero_quota_and_takes_the_next_type():
    quotas = _quotas(ml_g6e_4xlarge=0, ml_g6e_2xlarge=2)
    it, value, skipped = preflight.auto_instance(_spec("ml.g6e.4xlarge", "ml.g6e.2xlarge"), False, quotas)
    assert (it, value) == ("ml.g6e.2xlarge", 2) and skipped == ["ml.g6e.4xlarge: quota 0 (ml.g6e.4xlarge for training job usage)"]
    lines = []
    chosen = preflight.resolve_auto_instances({"act": False}, {"act": _spec("ml.g6e.4xlarge", "ml.g6e.2xlarge")}, FakeClients(quotas=quotas), "us-east-1", log=lines.append)
    assert chosen == {"act": "ml.g6e.2xlarge"}
    assert lines == ["  instance auto: act: ml.g6e.2xlarge (quota 2) after skipping ml.g6e.4xlarge: quota 0 (ml.g6e.4xlarge for training job usage)"]


def test_auto_instance_every_type_at_zero_names_each_quota():
    quotas = _quotas(ml_g6e_4xlarge=0, ml_g5_4xlarge=0)   # ml.g6.4xlarge is not listed at all: skipped too
    specs = {"act": _spec("ml.g6e.4xlarge", "ml.g6.4xlarge", "ml.g5.4xlarge")}
    with pytest.raises(preflight.PreflightError) as e:
        preflight.resolve_auto_instances({"act": False}, specs, FakeClients(quotas=quotas), "us-east-1", log=lambda m: None)
    msg = str(e.value)
    for text in ("act:", "us-east-1", "ml.g6e.4xlarge: quota 0 (ml.g6e.4xlarge for training job usage)", "ml.g6.4xlarge: quota not listed",
                 "ml.g5.4xlarge: quota 0 (ml.g5.4xlarge for training job usage)", "request an increase"):
        assert text in msg, text


def test_auto_instance_skips_multigpu_types_unless_allowed():
    quotas = _quotas(ml_g6e_12xlarge=1, ml_g6e_4xlarge=1)
    spec = _spec("ml.g6e.12xlarge", "ml.g6e.4xlarge")
    it, _, skipped = preflight.auto_instance(spec, False, quotas)
    assert it == "ml.g6e.4xlarge" and skipped == ["ml.g6e.12xlarge: 4 GPUs, refused without --allow-multigpu"]
    assert preflight.auto_instance(spec, False, quotas, allow_multigpu=True)[0] == "ml.g6e.12xlarge"


def test_auto_instance_skips_unknown_types_with_a_warning():
    it, _, skipped = preflight.auto_instance(_spec("ml.zz.9xlarge", "ml.g6e.4xlarge"), False, _quotas(ml_g6e_4xlarge=1))
    assert it == "ml.g6e.4xlarge" and len(skipped) == 1 and skipped[0].startswith("ml.zz.9xlarge: not in the instance table")
    lines = []   # the unknown type never reaches the quota read
    preflight.resolve_auto_instances({"act": False}, {"act": _spec("ml.zz.9xlarge", "ml.g6e.4xlarge")}, FakeClients(quotas=_quotas(ml_g6e_4xlarge=1)), "us-east-1", log=lines.append)
    assert "after skipping ml.zz.9xlarge: not in the instance table" in lines[0]


def test_auto_instance_unreadable_quotas_take_the_first_accepted_type_and_say_so():
    lines = []
    chosen = preflight.resolve_auto_instances({"act": False}, {"act": _spec("ml.g6e.12xlarge", "ml.g6e.4xlarge", "ml.g6e.2xlarge")}, Denied(), "us-east-1", log=lines.append)
    assert chosen == {"act": "ml.g6e.4xlarge"}
    assert lines[0] == "  instance auto: could not read service quotas (RuntimeError); taking the first type the instance rules accept for each policy"
    assert lines[1].startswith("  instance auto: act: ml.g6e.4xlarge (quota unread) after skipping ml.g6e.12xlarge")


def test_auto_instance_spot_run_reads_the_spot_quota():
    quotas = {"ml.g6e.4xlarge for training job usage": 4, "ml.g6e.4xlarge for spot training job usage": 0, "ml.g6e.2xlarge for spot training job usage": 1}
    assert preflight.auto_instance(_spec("ml.g6e.4xlarge", "ml.g6e.2xlarge"), True, quotas)[0] == "ml.g6e.2xlarge"
    assert preflight.auto_instance(_spec("ml.g6e.4xlarge", "ml.g6e.2xlarge"), False, quotas)[0] == "ml.g6e.4xlarge"


def test_auto_instance_registry_without_types_is_refused():
    spec = dict(POLICIES["act"]); spec["hardware"] = {k: v for k, v in spec["hardware"].items() if k != "instance_types"}
    with pytest.raises(preflight.PreflightError, match="lists no hardware.instance_types"):
        preflight.resolve_auto_instances({"act": False}, {"act": spec}, FakeClients(), "us-east-1", log=lambda m: None)


def test_every_registry_entry_resolves_auto_with_full_quotas():
    """With every quota at 1, 'auto' lands on a single-GPU type from the registry list for all 19 entries."""
    for p in sorted(render_flags.POLICIES_DIR.glob("*.yaml")):
        spec = render_flags.load_policy(p.stem)
        types = spec["hardware"]["instance_types"]
        it, _, _ = preflight.auto_instance(spec, False, {f"{t} for training job usage": 1 for t in types})
        assert it in types and instance_table.spec(it).gpus == 1, p.stem
