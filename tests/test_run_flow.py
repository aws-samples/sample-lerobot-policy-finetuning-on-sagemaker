# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""run.py past argument parsing, offline: the dry-run flow through stack, preflight, plan, stage, smoke and train
(fake boto3 clients, patched stack/image/preflight), the parallel fan-out's exit code and log handles, and a config
without the optional `smoke` block."""
import json
import sys
import types
from pathlib import Path

from botocore.exceptions import ClientError

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from pipeline import run  # noqa: E402
from tests.fixtures.make_mini_v3 import make_mini_v3  # noqa: E402

IMAGE = "123456789012.dkr.ecr.us-east-1.amazonaws.com/x@sha256:" + "a" * 64
STACK = {"project": "p", "region": "us-east-1", "bucket": "b", "sagemaker_role_arn": "arn:aws:iam::123456789012:role/r", "ecr_repo_uri": "x", "image_uri": IMAGE}


class _EmptyPaginator:
    def paginate(self, **kw): return [{}]


class FakeS3:
    """No staged marker, no stale objects; every write is a test failure (a dry run must not write)."""
    def get_object(self, Bucket, Key):
        raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
    def get_paginator(self, name): return _EmptyPaginator()
    def put_object(self, **kw): raise AssertionError("dry run wrote to S3")


class FakeSageMaker:
    def create_training_job(self, **kw): raise AssertionError("dry run submitted a job")


def _fake_client(sess, service):
    return {"s3": FakeS3(), "sagemaker": FakeSageMaker()}[service]


def _offline(monkeypatch, tmp_path):
    (tmp_path / "datasets").mkdir(); make_mini_v3(tmp_path / "datasets" / "mini")
    monkeypatch.setattr(run, "REPO", tmp_path)
    monkeypatch.setattr(run, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(run, "session", lambda region, profile=None: types.SimpleNamespace(region_name=region))
    monkeypatch.setattr(run, "client", _fake_client)
    monkeypatch.setattr(run, "resolve_stack", lambda sess, project, **kw: dict(STACK))
    monkeypatch.setattr(run, "ensure_image", lambda sess, stack, project, **kw: stack)
    monkeypatch.setattr(run, "run_preflight", lambda *a, **kw: [])


def _config(tmp_path, *, smoke="default", **policy):
    cfg = json.loads((REPO / "project_config.json").read_text())
    cfg["dataset"] = {"name": "mini", "source": "local", "hf_repo_id": None, "hf_revision": None, "s3_uri": None, "downscale_height": 360}
    cfg["policies"] = {"act": {"enabled": True, "steps": 1000, "save_freq": 500, **policy}}
    if smoke is None:
        del cfg["smoke"]
    elif smoke != "default":
        cfg["smoke"] = smoke
    path = tmp_path / "cfg.json"; path.write_text(json.dumps(cfg))
    return path


def test_dry_run_reaches_both_job_specs_with_the_per_policy_keep_checkpoints(tmp_path, monkeypatch, capsys):
    _offline(monkeypatch, tmp_path)
    cfg = _config(tmp_path, keep_checkpoints=5)
    assert run.main(["--config", str(cfg), "--region", "us-east-1", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "stack: bucket=b image=@sha256:aaaaaaaaaaaa..." in out and "\nplan:\n" in out and "stage:" in out
    assert out.count("(dry run: not submitted)") == 2 and "[act] smoke: 200 steps on ml.g6e.4xlarge" in out and "[act] train: 1000 steps" in out
    assert "'keep_checkpoints': '5'" in out            # R6: the per-policy override reaches the training hyperparameters
    specs = [json.loads("{\n" + chunk.split("\n  (dry run")[0]) for chunk in out.split("\n{\n")[1:]]
    assert [s["HyperParameters"].get("smoke") for s in specs] == ["1", None]
    assert "keep_checkpoints" not in specs[0]["HyperParameters"] and specs[1]["HyperParameters"]["keep_checkpoints"] == "5"
    assert specs[1]["AlgorithmSpecification"]["TrainingImage"] == IMAGE and specs[1]["EnableNetworkIsolation"] is True


def test_dry_run_without_the_smoke_block_uses_the_defaults(tmp_path, monkeypatch, capsys):
    _offline(monkeypatch, tmp_path)
    cfg = _config(tmp_path, smoke=None)
    assert run.main(["--config", str(cfg), "--region", "us-east-1", "--dry-run"]) == 0     # R12: no KeyError on cfg["smoke"]
    out = capsys.readouterr().out
    assert "[act] smoke: 200 steps" in out and out.count("(dry run: not submitted)") == 2
    assert "'keep_checkpoints': '3'" in out            # no per-policy value: compute.keep_checkpoints


def test_dry_run_with_smoke_disabled_or_skipped_prints_only_the_training_spec(tmp_path, monkeypatch, capsys):
    _offline(monkeypatch, tmp_path)
    assert run.main(["--config", str(_config(tmp_path, smoke={"enabled": False})), "--region", "us-east-1", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "[act] smoke:" not in out and out.count("(dry run: not submitted)") == 1
    assert run.main(["--config", str(_config(tmp_path)), "--region", "us-east-1", "--dry-run", "--skip-smoke"]) == 0
    assert capsys.readouterr().out.count("(dry run: not submitted)") == 1


class FakeSecretsManager:
    """The token secret as the dry run finds it (None = no secret). Every write is a test failure: before run.py passed
    `required=not a.dry_run`, a dry run with a gated policy prompted for the token and created the secret."""
    def __init__(self, value=None): self.value = value; self.calls = []
    def get_secret_value(self, SecretId):
        self.calls.append("get")
        if self.value is None:
            raise ClientError({"Error": {"Code": "ResourceNotFoundException"}}, "GetSecretValue")
        return {"SecretString": self.value}
    def create_secret(self, **kw): raise AssertionError("dry run wrote a Secrets Manager secret")
    def put_secret_value(self, **kw): raise AssertionError("dry run wrote a Secrets Manager secret")


def _refuse(msg):
    def f(*a, **k): raise AssertionError(msg)
    return f


def test_dry_run_with_a_gated_policy_only_reports_the_token_state(tmp_path, monkeypatch, capsys):
    """pi0 stages the gated PaliGemma tokenizer. A dry run reads the token secret once (GetSecretValue) and reports its
    state; it never prompts, never writes the secret, never calls the Hub without a stored token, and still reaches the
    staging lines and both job specs (hub channel mounted, no token in the training hyperparameters)."""
    _offline(monkeypatch, tmp_path)
    sm_fake = FakeSecretsManager(None)
    clients = lambda sess, service: {"s3": FakeS3(), "sagemaker": FakeSageMaker(), "secretsmanager": sm_fake}[service]
    monkeypatch.setattr(run, "client", clients)
    monkeypatch.setattr(run.hub, "client", clients)                                   # stage_hub builds its own S3 client
    monkeypatch.setattr(run.hf_token, "interactive", lambda: True)                    # a terminal is present: the old code prompted here
    monkeypatch.setattr(run.hf_token.getpass, "getpass", _refuse("dry run prompted for the token"))
    monkeypatch.setattr(run.hf_token, "whoami", _refuse("Hub called without a stored token"))
    cfg = json.loads(_config(tmp_path).read_text())
    cfg["policies"] = {"pi0": {"enabled": True, "steps": 1000, "save_freq": 500, "accept_licence": True}}
    (tmp_path / "cfg.json").write_text(json.dumps(cfg))
    assert run.main(["--config", str(tmp_path / "cfg.json"), "--region", "us-east-1", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert sm_fake.calls == ["get"]
    assert "NOTE hf token     lerobot-finetune/hf_token: not found" in out
    assert "[pi0] base weights:" in out and "will stage 2 artefact(s) for pi0" in out and "google/paligemma-3b-pt-224@" in out and "(text_tokenizer) [gated]" in out
    assert out.count("(dry run: not submitted)") == 2
    specs = [json.loads("{\n" + chunk.split("\n  (dry run")[0]) for chunk in out.split("\n{\n")[1:]]
    assert [[c["ChannelName"] for c in s["InputDataConfig"]] for s in specs] == [["train", "hub"], ["train", "hub"]]
    assert all("hf_token_secret" not in s["HyperParameters"] and s["EnableNetworkIsolation"] for s in specs)
    # a stored token the Hub rejects (401) is reported the same way: no re-prompt, no overwrite
    sm_fake.value, sm_fake.calls = json.dumps({"token": "hf_dead"}), []
    monkeypatch.setattr(run.hf_token, "whoami", lambda token, opener=None: {"ok": False, "user": None, "status": 401})
    assert run.main(["--config", str(tmp_path / "cfg.json"), "--region", "us-east-1", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert sm_fake.calls == ["get"] and "NOTE hf token     lerobot-finetune/hf_token: rejected by the Hub (401)" in out and out.count("(dry run: not submitted)") == 2


def test_stop_after_and_plan_exit_before_any_job_spec(tmp_path, monkeypatch, capsys):
    _offline(monkeypatch, tmp_path)
    cfg = _config(tmp_path)
    assert run.main(["--config", str(cfg), "--region", "us-east-1", "--stop-after", "preflight"]) == 0
    assert "plan:" not in capsys.readouterr().out
    assert run.main(["--config", str(cfg), "--region", "us-east-1", "--plan"]) == 0
    out = capsys.readouterr().out
    assert "total estimate" in out and "(dry run" not in out and "stage:" not in out


def test_run_parallel_signal_killed_child_is_non_zero_and_log_handles_are_closed(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(run, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(sys, "argv", ["run.py", "--policy", "act", "--policy", "diffusion", "--yes"])
    handles = {}

    class FakePopen:
        codes = {"act": -9, "diffusion": 0}
        def __init__(self, argv, stdout, stderr, cwd):
            assert argv[:3] == [sys.executable, "-m", "pipeline.run"] and stderr == run.subprocess.STDOUT
            self.name = argv[argv.index("--policy") + 1]
            handles[self.name] = stdout
            stdout.write(f"child {self.name} log line\n")
            self.pid = 4242
        def wait(self): return self.codes[self.name]

    monkeypatch.setattr(run.subprocess, "Popen", FakePopen)
    assert run._run_parallel(None, ["act", "diffusion"], "r") == 137
    assert set(handles) == {"act", "diffusion"} and all(h.closed for h in handles.values())
    out = capsys.readouterr().out
    assert "[act] finished with exit -9:\n  child act log line" in out and "[diffusion] finished with exit 0" in out
    assert (tmp_path / "runs" / "r-act.log").read_text() == "child act log line\n"
    assert run._child_exit_code(0) == 0 and run._child_exit_code(6) == 6 and run._child_exit_code(-15) == 143


def test_stop_after_rejects_the_stages_that_do_not_stop(tmp_path):
    """--stop-after train / report were accepted and ignored; argparse now refuses them (exit 2). --no-wait is the way
    to return after the submit."""
    import pytest
    with pytest.raises(SystemExit) as e:
        run.main(["--stop-after", "train"])
    assert e.value.code == 2


# ---------------------------------------------------------------- --instance-type auto -------------------------------
class FakeQuotas:
    def __init__(self, quotas): self.quotas = quotas
    def get_paginator(self, name):
        assert name == "list_service_quotas"
        q = self.quotas
        class P:
            def paginate(s, **kw): yield {"Quotas": [{"QuotaName": k, "Value": v} for k, v in q.items()]}
        return P()


def _offline_with_quotas(monkeypatch, tmp_path, quotas):
    _offline(monkeypatch, tmp_path)
    monkeypatch.setattr(run, "client", lambda sess, service: {"s3": FakeS3(), "sagemaker": FakeSageMaker(), "service-quotas": FakeQuotas(quotas)}[service])


def test_dry_run_instance_type_auto_takes_the_first_registry_type_with_quota(tmp_path, monkeypatch, capsys):
    """act lists ml.g6e.4xlarge first: with its quota at 0 the run lands on ml.g6e.2xlarge, in the plan, the smoke and the training spec."""
    _offline_with_quotas(monkeypatch, tmp_path, {"ml.g6e.4xlarge for training job usage": 0, "ml.g6e.2xlarge for training job usage": 1})
    assert run.main(["--config", str(_config(tmp_path)), "--region", "us-east-1", "--dry-run", "--instance-type", "auto"]) == 0
    out = capsys.readouterr().out
    assert "instance auto: act: ml.g6e.2xlarge (quota 1) after skipping ml.g6e.4xlarge: quota 0 (ml.g6e.4xlarge for training job usage)" in out
    assert out.index("instance auto:") < out.index("preflight:") < out.index("\nplan:\n")
    assert "[act] smoke: 200 steps on ml.g6e.2xlarge" in out and "[act] train: 1000 steps on ml.g6e.2xlarge" in out
    specs = [json.loads("{\n" + chunk.split("\n  (dry run")[0]) for chunk in out.split("\n{\n")[1:]]
    assert [s["ResourceConfig"]["InstanceType"] for s in specs] == ["ml.g6e.2xlarge", "ml.g6e.2xlarge"]
    plan_block = out.split("\nplan:\n")[1].split("\n\n")[0]
    assert "ml.g6e.2xlarge" in plan_block and "ml.g6e.4xlarge" not in plan_block


def test_dry_run_auto_in_the_config_without_the_flag(tmp_path, monkeypatch, capsys):
    _offline_with_quotas(monkeypatch, tmp_path, {"ml.g6e.4xlarge for training job usage": 2})
    cfg = json.loads(_config(tmp_path).read_text()); cfg["compute"]["instance_type"] = "auto"
    (tmp_path / "cfg.json").write_text(json.dumps(cfg))
    assert run.main(["--config", str(tmp_path / "cfg.json"), "--region", "us-east-1", "--plan"]) == 0
    out = capsys.readouterr().out
    assert "instance auto: act: ml.g6e.4xlarge (quota 2)\n" in out and "ml.g6e.4xlarge" in out.split("\nplan:\n")[1]
    # a per-policy "auto" under a named compute type resolves that policy only; --instance-type <type> still wins over both
    cfg["compute"]["instance_type"] = "ml.g5.4xlarge"; cfg["policies"]["act"]["instance_type"] = "auto"
    (tmp_path / "cfg.json").write_text(json.dumps(cfg))
    assert run.main(["--config", str(tmp_path / "cfg.json"), "--region", "us-east-1", "--plan"]) == 0
    assert "instance auto: act: ml.g6e.4xlarge (quota 2)" in capsys.readouterr().out
    assert run.main(["--config", str(tmp_path / "cfg.json"), "--region", "us-east-1", "--plan", "--instance-type", "ml.g5.4xlarge"]) == 0
    out = capsys.readouterr().out
    assert "instance auto:" not in out and "ml.g5.4xlarge" in out.split("\nplan:\n")[1]


def test_dry_run_auto_with_no_quota_anywhere_exits_4_before_the_plan(tmp_path, monkeypatch, capsys):
    _offline_with_quotas(monkeypatch, tmp_path, {})
    assert run.main(["--config", str(_config(tmp_path)), "--region", "us-east-1", "--dry-run", "--instance-type", "auto"]) == 4
    cap = capsys.readouterr()
    assert "plan:" not in cap.out and "no instance type with quota" in cap.err and "ml.g6e.4xlarge: quota not listed" in cap.err


def test_dry_run_auto_and_a_training_plan_are_exclusive(tmp_path, monkeypatch, capsys):
    _offline_with_quotas(monkeypatch, tmp_path, {"ml.g6e.4xlarge for training job usage": 1})
    rc = run.main(["--config", str(_config(tmp_path)), "--region", "us-east-1", "--plan", "--instance-type", "auto",
                   "--training-plan", "arn:aws:sagemaker:us-east-1:123456789012:training-plan/p"])
    assert rc == 2 and "'auto' and a training plan are exclusive" in capsys.readouterr().err
