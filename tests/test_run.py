# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
import json
import sys

import pytest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "trainer"))
import render_flags  # noqa: E402
from pipeline import run  # noqa: E402
from pipeline.config import load_project_config  # noqa: E402

CFG = load_project_config()
SPEC = render_flags.load_policy("act")


def test_policy_hps_chunk_from_horizon_or_explicit():
    pol = dict(CFG["policies"]["act"])
    hps = run.policy_hps(CFG, "act", pol, SPEC, {"fps": 30}, steps=100000, smoke=False, image_digest="sha256:a", fingerprint="f")
    assert hps["chunk"] == "30" and hps["steps"] == "100000" and hps["log_freq"] == "200" and hps["eval_stride"] == "1" and "smoke" not in hps
    assert hps["save_freq"] == "5000" and hps["policy"] == "act" and hps["image_digest"] == "sha256:a" and "heldout" not in hps
    pol["chunk"] = 24
    hps = run.policy_hps(CFG, "act", pol, SPEC, {"fps": 25}, steps=200, smoke=True, image_digest="", fingerprint="f")
    assert hps["chunk"] == "24" and hps["smoke"] == "1" and hps["log_freq"] == "50" and hps["eval_stride"] == "60" and hps["save_freq"] == "200"
    cfg2 = dict(CFG); cfg2["evaluation"] = {"heldout": [0, 13, 26], "stride": 1, "checkpoints": ["last", "050000"]}
    hps = run.policy_hps(cfg2, "act", pol, SPEC, {"fps": 25}, steps=100, smoke=False, image_digest="", fingerprint="f")
    assert hps["heldout"] == "0 13 26" and hps["eval_ckpts"] == "last 050000"


def test_smoke_gate_rules():
    good = {"evals": {"eval_last.json": {"joints_mae": 18.4}}, "train_exitcode": 0, "step_rate": 9.8,
            "smoke": {"weights_moved_fraction": 1.0, "bf16_trainable_params": 0}, "reload_check": {"ok": True}}
    ok, reasons, rate = run.smoke_gate(good, SPEC)
    assert ok and rate == 9.8
    bad = dict(good); bad["smoke"] = {"weights_moved_fraction": 0.1, "bf16_trainable_params": 5}
    ok, reasons, _ = run.smoke_gate(bad, SPEC)
    assert not ok and len(reasons) == 2
    worse_than_hold = dict(good); worse_than_hold["evals"] = {"eval_last.json": {"joints_mae": 99.0, "joints_hold": 1.0}}
    assert run.smoke_gate(worse_than_hold, SPEC)[0]     # beats-hold is never part of the gate
    assert not run.smoke_gate({"evals": {}}, SPEC)[0]


def test_job_spec_prefixes_and_tags():
    stack = {"bucket": "b", "sagemaker_role_arn": "arn:aws:iam::123456789012:role/r", "ecr_repo_uri": "x"}
    spec = run.job_spec(CFG, stack, "act", CFG["policies"]["act"], SPEC, {"steps": "200"}, kind="smoke", instance="ml.g6e.4xlarge", spot=False,
                        image="123456789012.dkr.ecr.ap-northeast-1.amazonaws.com/x@sha256:" + "a" * 64, dataset_s3="s3://b/sagemaker/datasets/d/",
                        max_runtime_s=3600, volume_gb=100, run_name="r")
    assert spec["OutputDataConfig"]["S3OutputPath"] == "s3://b/sagemaker/jobs/smoke/" and spec["CheckpointConfig"]["S3Uri"].startswith("s3://b/sagemaker/jobs/smoke/r-act-smoke-")
    assert {"Key": "kind", "Value": "smoke"} in spec["Tags"] and spec["EnableNetworkIsolation"] is True


def test_forbidden_flags_two_token_form():
    import pytest
    forbidden = ["--policy.dtype=bfloat16", "--policy.normalization_mapping"]
    render_flags.check_forbidden(["--steps=10", "--policy.dtype=float32"], forbidden)          # allowed value
    for toks in (["--policy.dtype=bfloat16"], ["--policy.dtype", "bfloat16"], ["--policy.normalization_mapping", "x"]):
        with pytest.raises(render_flags.RenderError):
            render_flags.check_forbidden(toks, forbidden)


def test_print_dotted_registry_key(capsys):
    assert render_flags.main(["--policy", "diffusion", "--print", "eval.n_samples"]) == 0
    assert capsys.readouterr().out.strip() == "5"


def test_extend_steps_requires_resume_from(capsys):
    assert run.main(["--extend-steps", "150000", "--dry-run"]) == 2
    assert "--resume-from" in capsys.readouterr().err


def test_resume_from_needs_exactly_one_policy(capsys):
    """The parent hands its --resume-from to every child; one checkpoint prefix cannot continue two policies."""
    assert run.main(["--resume-from", "job-1", "--policy", "act", "--policy", "diffusion", "--dry-run"]) == 2
    err = capsys.readouterr().err
    assert "--resume-from continues one job" in err and "act, diffusion" in err


def test_keep_checkpoints_per_policy_override():
    """policies.<name>.keep_checkpoints wins over compute.keep_checkpoints; null falls back; smoke runs send nothing."""
    cfg = json.loads(json.dumps(CFG)); cfg["compute"]["keep_checkpoints"] = 4
    assert run._keep_checkpoints(cfg, {}) == 4 and run._keep_checkpoints(cfg, {"keep_checkpoints": None}) == 4
    assert run._keep_checkpoints(cfg, {"keep_checkpoints": 0}) == 0 and run._keep_checkpoints(cfg, {"keep_checkpoints": 7}) == 7
    assert run._keep_checkpoints({"compute": {}}, {}) == 3
    kw = dict(steps=1000, smoke=False, image_digest="sha256:a", fingerprint="f")
    assert run.policy_hps(cfg, "act", {"enabled": True, "keep_checkpoints": 7}, SPEC, {"fps": 30}, **kw)["keep_checkpoints"] == "7"
    assert run.policy_hps(cfg, "act", {"enabled": True}, SPEC, {"fps": 30}, **kw)["keep_checkpoints"] == "4"
    assert "keep_checkpoints" not in run.policy_hps(cfg, "act", {"enabled": True, "keep_checkpoints": 7}, SPEC, {"fps": 30}, **{**kw, "smoke": True})


def test_download_outputs_skips_a_missing_output_archive_and_logs_other_errors(tmp_path, capsys):
    import io
    import tarfile
    from botocore.exceptions import ClientError
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        data = json.dumps({"evals": {}}).encode(); ti = tarfile.TarInfo("SUMMARY.json"); ti.size = len(data); tf.addfile(ti, io.BytesIO(data))
    model_tgz = buf.getvalue()
    describe = {"ModelArtifacts": {"S3ModelArtifacts": "s3://b/sagemaker/jobs/train/j/output/model.tar.gz"}}

    class S3:
        def __init__(self, output_error): self.output_error = output_error
        def get_object(self, Bucket, Key):
            if Key.endswith("model.tar.gz"):
                return {"Body": io.BytesIO(model_tgz)}
            assert Key == "sagemaker/jobs/train/j/output/output.tar.gz"
            raise ClientError({"Error": {"Code": self.output_error}}, "GetObject")

    assert run.download_outputs(S3("NoSuchKey"), "b", describe, tmp_path / "a") == {"evals": {}}
    assert capsys.readouterr().out == ""                                    # a missing output.tar.gz is the normal case
    assert run.download_outputs(S3("AccessDenied"), "b", describe, tmp_path / "b") == {"evals": {}}
    out = capsys.readouterr().out
    assert "NOTE" in out and "s3://b/sagemaker/jobs/train/j/output/output.tar.gz" in out and "AccessDenied" in out

    from botocore.exceptions import ConnectionClosedError   # a BotoCoreError that is not an OSError: must not abort the report

    class S3Transport(S3):
        def get_object(self, Bucket, Key):
            if Key.endswith("model.tar.gz"):
                return {"Body": io.BytesIO(model_tgz)}
            raise ConnectionClosedError(endpoint_url="https://s3.example")

    assert run.download_outputs(S3Transport("x"), "b", describe, tmp_path / "c") == {"evals": {}}
    out = capsys.readouterr().out
    assert "NOTE" in out and "output.tar.gz" in out and "closed" in out.lower()


def test_state_for_job_reads_saved_runs(tmp_path, monkeypatch):
    import json
    monkeypatch.setattr(run, "RUNS", tmp_path)
    (tmp_path / "a.json").write_text(json.dumps({"name": "a", "train_job": "job-a", "policy": "act"}))
    (tmp_path / "broken.json").write_text("{not json")
    assert run._state_for_job("job-a")["name"] == "a"
    assert run._state_for_job("job-zz") is None


def test_s3_dataset_facts_reads_info_json():
    import json
    import pytest
    from botocore.exceptions import ClientError

    class S3:
        def __init__(self, body=None, err=None): self.body, self.err = body, err
        def get_paginator(self, name): return _Paginator([{"Contents": [{"Key": "ds/x/meta/info.json", "Size": 10}, {"Key": "ds/x/data/f.parquet", "Size": 90}]}])
        def get_object(self, Bucket, Key):
            assert (Bucket, Key) == ("b", "ds/x/meta/info.json")
            if self.err:
                raise ClientError({"Error": {"Code": self.err}}, "GetObject")
            class Body:
                def read(_self): return self.body
            return {"Body": Body()}

    split, fp, count, nbytes = run._s3_dataset_facts(S3(json.dumps({"codebase_version": "v3.0", "fps": 30, "features": {}}).encode()), "s3://b/ds/x/")
    assert split["fps"] == 30 and fp.startswith("s3-") and (count, nbytes) == (2, 100)
    with pytest.raises(run.Abort):
        run._s3_dataset_facts(S3(err="NoSuchKey"), "s3://b/ds/x")
    with pytest.raises(run.Abort):
        run._s3_dataset_facts(S3(json.dumps({"codebase_version": "v2.1", "fps": 30}).encode()), "s3://b/ds/x")


def test_camera_hyperparameter_and_notes(capsys):
    spec = render_flags.load_policy("vqbet")
    hps = run.policy_hps(CFG, "vqbet", {"enabled": True, "camera": "observation.images.side"}, spec, {"fps": 30}, steps=200, smoke=True,
                         image_digest="sha256:x", fingerprint="f")
    assert hps["camera"] == "observation.images.side"
    assert "camera" not in run.policy_hps(CFG, "act", {"enabled": True}, SPEC, {"fps": 30}, steps=200, smoke=True, image_digest="sha256:x", fingerprint="f")
    keys = ["observation.images.side", "observation.images.up"]
    run._camera_notes({"vqbet": {"enabled": True}}, {"vqbet": spec}, keys, print)
    out = capsys.readouterr().out
    assert "trains on observation.images.side" in out and "policies.vqbet.camera" in out
    import pytest
    with pytest.raises(run.Abort):
        run._camera_notes({"vqbet": {"enabled": True, "camera": "observation.images.nope"}}, {"vqbet": spec}, keys, print)
    run._camera_notes({"act": {"enabled": True, "camera": "observation.images.up"}}, {"act": SPEC}, keys, print)
    assert "ignored" in capsys.readouterr().out


def test_passed_smoke_requires_a_passed_gate(tmp_path, monkeypatch):
    import json
    monkeypatch.setattr(run, "RUNS", tmp_path)
    hps = {"steps": "200", "policy": "act"}
    base = {"policy": "act", "smoke_hps": hps, "instance_type": "ml.g6e.4xlarge", "smoke_job": "j", "smoke_summary": {"step_rate": 9.8}}
    (tmp_path / "failed.json").write_text(json.dumps({**base, "smoke_passed": False}))
    (tmp_path / "legacy.json").write_text(json.dumps(base))            # no verdict recorded: never reused
    assert run._passed_smoke("act", hps, "ml.g6e.4xlarge") is None
    (tmp_path / "passed.json").write_text(json.dumps({**base, "smoke_passed": True, "smoke_job": "ok-job"}))
    assert run._passed_smoke("act", hps, "ml.g6e.4xlarge") == {"smoke_job": "ok-job", "step_rate": 9.8}


class _Paginator:
    def __init__(self, pages): self.pages = pages
    def paginate(self, **kw): return self.pages


def test_stale_remote_lists_only_objects_without_a_local_file(tmp_path):
    root = tmp_path / "ds"; (root / "meta").mkdir(parents=True); (root / ".cache" / "x").mkdir(parents=True)
    (root / "meta" / "info.json").write_text("{}"); (root / ".cache" / "x" / "tmp").write_text("")
    class S3:
        def get_paginator(self, name):
            assert name == "list_objects_v2"
            return _Paginator([{"Contents": [{"Key": "sagemaker/datasets/ds/meta/info.json"}, {"Key": "sagemaker/datasets/ds/data/old.parquet"}]}])
    assert run._stale_remote(S3(), "b", "sagemaker/datasets/ds/", root) == ["sagemaker/datasets/ds/data/old.parquet"]


def test_quota_flags_spot_only_for_gpu_instances():
    from pipeline import preflight, hub
    assert preflight._quota_flags("ml.g6e.4xlarge", True) == {True}       # smoke and training share the Spot capacity
    assert preflight._quota_flags("ml.g6e.4xlarge", False) == {False}
    assert preflight._quota_flags(hub.STAGE_INSTANCE, True) == {False}
    assert preflight._quota_flags("ml.c5.4xlarge", True) == {False}


def test_smoke_only_argv_is_appended_only_when_smoke_is_set():
    spec = render_flags.load_policy("vqbet")
    env = {"REPO_ID": "x", "DS_ROOT": "/d", "TRAIN_EPS": "[1]", "CHUNK": "5", "DEVICE": "cpu", "BATCH": "2", "STEPS": "200",
           "SAVE_FREQ": "100", "LOG_FREQ": "50", "NUM_WORKERS": "1", "SEED": "1", "LR_OUT": "/o", "JOB_NAME": "j"}
    assert [t for t in render_flags.render(spec, "fresh", env) if "n_vqvae_training_steps" in t] == ["--policy.n_vqvae_training_steps=20000"]
    assert [t for t in render_flags.render(spec, "fresh", {**env, "SMOKE": "1"}) if "n_vqvae_training_steps" in t] == ["--policy.n_vqvae_training_steps=100"]
    assert render_flags.render(SPEC, "fresh", {**env, "SMOKE": "1"}) == render_flags.render(SPEC, "fresh", env)   # ACT has no smoke argv


def test_per_policy_dataset_override_and_groups():
    import json
    import jsonschema
    cfg = json.loads(json.dumps(CFG))
    cfg["policies"]["vqbet"] = {"enabled": True, "dataset": {"name": "one_cam_task", "source": "local"}}
    cfg["policies"]["diffusion"] = {"enabled": True}
    schema = json.loads((REPO / "pipeline" / "project_config.schema.json").read_text())
    jsonschema.validate(cfg, schema)                      # the override is a documented key
    assert run._dataset_cfg(cfg, cfg["policies"]["act"]) is cfg["dataset"]
    ov = run._dataset_cfg(cfg, cfg["policies"]["vqbet"])
    assert ov["name"] == "one_cam_task" and ov["source"] == "local" and ov["downscale_height"] == cfg["dataset"]["downscale_height"]
    pols = {k: cfg["policies"][k] for k in ("act", "vqbet", "diffusion")}
    groups = run._dataset_groups(cfg, pols)
    assert groups == {cfg["dataset"]["name"]: ["act", "diffusion"], "one_cam_task": ["vqbet"]}
    with pytest.raises(jsonschema.ValidationError):
        cfg["policies"]["vqbet"]["dataset"] = {"name": "bad name with spaces"}
        jsonschema.validate(cfg, schema)


def test_quantile_note_only_when_stats_lack_q01(tmp_path, capsys):
    import json
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta" / "stats.json").write_text(json.dumps({"action": {"mean": [0]}, "observation.state": {"q01": [0], "q99": [1]}}))
    run._quantile_notes({"pi05": render_flags.load_policy("pi05"), "act": SPEC}, tmp_path, print)
    out = capsys.readouterr().out
    assert "pi05" in out and "['action']" in out
    run._quantile_notes({"act": SPEC}, tmp_path, print)
    assert capsys.readouterr().out == ""


def test_download_summary_only_fetches_summary_and_eval_files(tmp_path):
    import json
    ckpt = "s3://b/sagemaker/jobs/train/job-1/checkpoints/"
    objects = {"sagemaker/jobs/train/job-1/checkpoints/act/SUMMARY.json": json.dumps({"evals": {"eval_last.json": {"joints_mae": 1.0}}}).encode(),
               "sagemaker/jobs/train/job-1/checkpoints/act/eval/eval_last.json": b"{}", "sagemaker/jobs/train/job-1/checkpoints/act/eval/eval_last_preds.npz": b"npz",
               "sagemaker/jobs/train/job-1/checkpoints/act/eval/sub/ignored.json": b"x"}
    class S3:
        def get_object(self, Bucket, Key):
            class B:
                def __init__(s, d): s.d = d
                def read(s): return s.d
            return {"Body": B(objects[Key])}
        def get_paginator(self, name): return _Paginator([{"Contents": [{"Key": k} for k in objects if "/eval/" in k]}])
    describe = {"CheckpointConfig": {"S3Uri": ckpt}}
    summary = run.download_summary_only(S3(), "b", describe, "act", tmp_path)
    assert summary["evals"]["eval_last.json"]["joints_mae"] == 1.0
    assert sorted(p.name for p in (tmp_path / "model" / "eval").iterdir()) == ["eval_last.json", "eval_last_preds.npz"]
    assert run.main(["--no-download", "--extend-steps", "1", "--dry-run"]) == 2      # flag accepted; the guard still fires


def test_plan_rows_and_child_argv():
    from pipeline import plan, instance_table
    cfg = json.loads(json.dumps(CFG))
    spec = render_flags.load_policy("pi0_fast")
    r = plan.policy_row("pi0_fast", {"enabled": True, "batch_size": 4}, spec, cfg, "so101-4tasks-100eps", "ml.g6e.4xlarge", False, None, instance_table.spec("ml.g6e.4xlarge"))
    assert r["blocked"] and "TOO SMALL" in r["gpu"] and "accept_licence missing" in r["licence"]
    r = plan.policy_row("pi0_fast", {"enabled": True, "accept_licence": True}, spec, cfg, "so101-4tasks-100eps", "ml.p5.4xlarge", True, 1.61, instance_table.spec("ml.p5.4xlarge"))
    assert not r["blocked"] and r["est_h"] and r["est_usd"] and "Spot" in r["instance"]
    r = plan.policy_row("act", {"enabled": True, "steps": 100000}, SPEC, cfg, "svla", "ml.g6e.4xlarge", False, 10.0, instance_table.spec("ml.g6e.4xlarge"))
    assert not r["blocked"] and 2.5 < r["est_h"] < 4 and 15 < r["est_usd"] < 25
    txt = plan.table([r], ["svla: 9 files, 86 MB -> s3://b/x/ (skipped when already staged)"])
    assert "total estimate" in txt and "upload: svla" in txt
    argv = run._child_argv(["--config", "c.json", "--policy", "act", "--policy", "pi0", "--yes", "--name", "run1", "--spot", "--no-download"], "pi0", "run1")
    assert argv[3:] == ["--config", "c.json", "--spot", "--no-download", "--policy", "pi0", "--name", "run1", "--yes", "--sequential", "--no-plan"]
    # the parent resolved 'auto' to one type and printed it in the plan: the child gets that type, never 'auto'
    argv = run._child_argv(["--instance-type", "auto", "--yes", "--spot", "--policy", "act", "--policy", "pi0"], "pi0", "run1", "ml.g6e.2xlarge")
    assert "auto" not in argv and argv[argv.index("--instance-type") + 1] == "ml.g6e.2xlarge" and argv.count("--instance-type") == 1
    argv = run._child_argv(["--instance-type=auto", "--yes"], "act", "run1", "ml.g6e.4xlarge")
    assert "--instance-type=auto" not in argv and argv[argv.index("--instance-type") + 1] == "ml.g6e.4xlarge"


def test_shipped_config_lists_all_in_scope_policies():
    pols = [k for k in CFG["policies"] if not k.startswith("_")]
    assert len(pols) == 17 and [k for k, v in CFG["policies"].items() if not k.startswith("_") and v.get("enabled")] == ["act"]   # 16 policy types + the fastwam_lora recipe
    for k in pols:
        render_flags.load_policy(k)   # every listed policy exists in the registry


def test_cameras_hyperparameter_is_comma_joined():
    hps = run.policy_hps(CFG, "fastwam", {"enabled": True, "cameras": ["observation.images.front", "observation.images.wrist_right"], "accept_licence": True}, render_flags.load_policy("fastwam"), {"fps": 25}, steps=200, smoke=True, image_digest="sha256:x", fingerprint="f")
    assert hps["cameras"] == "observation.images.front,observation.images.wrist_right"


def test_no_local_variable_shadows_the_plan_module():
    """`plan` is the pipeline.plan module inside run.py; a local `plan = ...` (the training-plan ARN once had that name)
    crashes every run at the plan table with AttributeError, and no stage unit test reaches that line."""
    import re
    src = (REPO / "pipeline" / "run.py").read_text()
    assert not re.search(r"^\s+plan\s*=", src, re.M), "rename the local variable; `plan` is the module"
    assert callable(run.plan.policy_row)


def test_requested_instance_and_policy_spot_precedence():
    import types
    cfg = {"compute": {"instance_type": "ml.g6e.4xlarge", "spot": True}}
    a = types.SimpleNamespace(instance_type=None, spot=None)
    assert run._requested_instance(a, cfg, {}) == "ml.g6e.4xlarge" and run._policy_spot(a, cfg, {}) is True
    assert run._requested_instance(a, cfg, {"instance_type": None}) == "ml.g6e.4xlarge"       # null = no override
    assert run._requested_instance(a, cfg, {"instance_type": "auto"}) == "auto" and run._policy_spot(a, cfg, {"spot": False}) is False
    a = types.SimpleNamespace(instance_type="ml.g5.4xlarge", spot=False)
    assert run._requested_instance(a, cfg, {"instance_type": "auto"}) == "ml.g5.4xlarge" and run._policy_spot(a, cfg, {"spot": True}) is False
