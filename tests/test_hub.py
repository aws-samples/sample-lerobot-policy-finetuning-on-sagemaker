# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
import io
import json
import os
import sys

import pytest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "trainer"))
import hub_stage  # noqa: E402
from pipeline import hub  # noqa: E402

STACK = {"bucket": "b", "sagemaker_role_arn": "arn:aws:iam::123456789012:role/r", "ecr_repo_uri": "x"}
CFG = {"project": "p", "secrets": {"hf_token": None}, "extra_job_tags": {}}
ART = [{"repo": "lerobot/xvla-base", "revision": None, "role": "policy", "gated": False},
       {"repo": "facebook/bart-large", "revision": "main", "role": "tokenizer", "gated": False}]


def _no_listing(repo_id, revision, repo_type, token):
    return None


def test_stage_writes_cache_layout_manifest(tmp_path):
    def fake_download(repo_id, revision, cache_dir, token, repo_type, allow_patterns=None, ignore_patterns=None):
        repo = Path(cache_dir) / f"models--{repo_id.replace('/', '--')}"
        blob = repo / "blobs" / "deadbeef"; blob.parent.mkdir(parents=True); blob.write_text("{}")
        p = repo / "snapshots" / "abc123"; p.mkdir(parents=True)
        (p / "config.json").symlink_to(Path("..") / ".." / "blobs" / "deadbeef")   # the real HF cache layout
        return str(p)
    m = hub_stage.stage(ART, str(tmp_path), downloader=fake_download, lister=_no_listing)
    assert [x["resolved_sha"] for x in m["artifacts"]] == ["abc123", "abc123"] and m["schema"] == 3
    assert json.loads((tmp_path / "_STAGED.json").read_text())["request"] == ART
    cfg = tmp_path / "hub" / "models--lerobot--xvla-base" / "snapshots" / "abc123" / "config.json"
    assert cfg.is_file() and not cfg.is_symlink() and cfg.read_text() == "{}"          # materialised: the S3 mirror drops symlinks
    assert not (tmp_path / "hub" / "models--lerobot--xvla-base" / "blobs").exists() and m["materialized"] is True


# what the Hub lists for the two X-VLA artefacts: the tokenizer repo also ships 5.5 GB of weights
BART_FILES = [".gitattributes", "README.md", "config.json", "flax_model.msgpack", "merges.txt", "pytorch_model.bin", "rust_model.ot",
              "tf_model.h5", "tokenizer.json", "tokenizer_config.json", "vocab.json"]
SMOLVLM_FILES = ["config.json", "model.safetensors", "onnx/decoder_model_merged.onnx", "onnx/vision_encoder_fp16.onnx",
                 "preprocessor_config.json", "processor_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json"]


def test_download_patterns_per_role():
    allow, ignore = hub_stage.download_patterns({"repo": "facebook/bart-large", "role": "tokenizer"}, BART_FILES)
    assert ignore is None and "tokenizer*" in allow and "vocab*" in allow and "merges*" in allow and "config.json" in allow
    assert not any(p.endswith((".bin", ".h5", ".msgpack", ".ot")) for p in allow)
    allow, ignore = hub_stage.download_patterns({"repo": "google/paligemma-3b-pt-224", "role": "text_tokenizer"}, None)
    assert ignore is None and "tokenizer*" in allow
    # weights roles: full snapshot minus the alternative-framework files; *.bin only when safetensors exist
    allow, ignore = hub_stage.download_patterns({"repo": "HuggingFaceTB/SmolVLM2-500M-Video-Instruct", "role": "vlm_backbone_and_tokenizer"}, SMOLVLM_FILES)
    assert allow is None and "onnx/*" in ignore and "*.msgpack" in ignore and "*.h5" in ignore and "*.ot" in ignore and "*.gguf" in ignore
    assert "*.bin" in ignore
    allow, ignore = hub_stage.download_patterns({"repo": "org/torch-only", "role": "base_checkpoint"}, ["config.json", "pytorch_model.bin"])
    assert allow is None and "*.bin" not in ignore                                   # no safetensors: the .bin IS the model
    allow, ignore = hub_stage.download_patterns({"repo": "org/unknown", "role": "base_checkpoint"}, None)
    assert "*.bin" not in ignore and "onnx/*" in ignore                               # listing failed: keep the .bin, still drop onnx
    # action tokenizers are AutoProcessor + remote code: never reduced to the tokenizer files
    allow, ignore = hub_stage.download_patterns({"repo": "lerobot/fast-action-tokenizer", "role": "action_tokenizer"}, ["tokenizer.json", "processing.py"])
    assert allow is None
    # an explicit registry override wins
    assert hub_stage.download_patterns({"repo": "x", "role": "tokenizer", "allow_patterns": ["*.json"]}, None) == (["*.json"], None)


def test_stage_passes_patterns_to_the_downloader_and_records_them(tmp_path):
    calls = []

    def fake_download(repo_id, revision, cache_dir, token, repo_type, allow_patterns=None, ignore_patterns=None):
        calls.append((repo_id, allow_patterns, ignore_patterns))
        snap = Path(cache_dir) / f"models--{repo_id.replace('/', '--')}" / "snapshots" / "abc123"; snap.mkdir(parents=True)
        (snap / "config.json").write_text("{}")
        return str(snap)

    listed = []

    def fake_list(repo_id, revision, repo_type, token):
        listed.append(repo_id)
        return ["config.json", "model.safetensors", "pytorch_model.bin", "onnx/x.onnx"]

    m = hub_stage.stage(ART, str(tmp_path), downloader=fake_download, lister=fake_list)
    assert m["schema"] == 3 and m["materialized"] is True and m["cleared"] is True     # schema 3: patterns recorded, prefix cleared
    assert listed == ["lerobot/xvla-base"]                                            # tokenizer roles need no listing
    assert calls[0][0] == "lerobot/xvla-base" and calls[0][1] is None and "onnx/*" in calls[0][2] and "*.bin" in calls[0][2]
    assert calls[1][0] == "facebook/bart-large" and "tokenizer*" in calls[1][1] and calls[1][2] is None
    arts = json.loads((tmp_path / "_STAGED.json").read_text())["artifacts"]
    assert arts[1]["allow_patterns"] == calls[1][1] and arts[0]["ignore_patterns"] == calls[0][2]
    assert {a["resolved_sha"] for a in arts} == {"abc123"} and all("bytes" in a and "role" in a for a in arts)


def test_stage_clears_the_earlier_layout_before_downloading(tmp_path):
    """SageMaker pre-populates the output directory from the prefix; snapshot_download keeps files already present under
    snapshots/<sha>/, so the stale schema-1 hub/hub/ layer, the ONNX exports and the old manifest must be gone before
    the first download or a re-stage cannot shrink the prefix."""
    stale_layer = tmp_path / "hub" / "hub" / "x"; stale_layer.parent.mkdir(parents=True); stale_layer.write_text("schema-1 layer")
    onnx = tmp_path / "hub" / "models--lerobot--xvla-base" / "snapshots" / "abc123" / "onnx" / "x.onnx"
    onnx.parent.mkdir(parents=True); onnx.write_text("never read")
    (tmp_path / "_STAGED.json").write_text(json.dumps({"schema": 2, "materialized": True, "request": ART}))

    def fake_download(repo_id, revision, cache_dir, token, repo_type, allow_patterns=None, ignore_patterns=None):
        assert not stale_layer.exists() and not onnx.exists() and not (tmp_path / "_STAGED.json").exists()   # cleared first
        snap = Path(cache_dir) / f"models--{repo_id.replace('/', '--')}" / "snapshots" / "abc123"; snap.mkdir(parents=True, exist_ok=True)
        (snap / "config.json").write_text("{}")
        return str(snap)

    m = hub_stage.stage(ART, str(tmp_path), downloader=fake_download, lister=_no_listing)
    assert not stale_layer.exists() and not onnx.exists()
    written = json.loads((tmp_path / "_STAGED.json").read_text())
    assert written["schema"] == 3 and written["cleared"] is True and written["request"] == ART and m["schema"] == 3
    assert (tmp_path / "hub" / "models--lerobot--xvla-base" / "snapshots" / "abc123" / "config.json").is_file()


class FakeS3:
    """get_object / list_objects_v2 / delete_objects over one policy prefix; records what was deleted."""

    def __init__(self, manifest=None, objects=None):
        self.manifest, self.objects, self.deleted = manifest, dict(objects or {}), []

    def get_object(self, Bucket, Key):
        if self.manifest is None:
            raise Exception("NoSuchKey")
        return {"Body": io.BytesIO(json.dumps(self.manifest).encode())}

    def list_objects_v2(self, Bucket, Prefix, **kw):
        return {"Contents": [{"Key": k, "Size": v} for k, v in self.objects.items() if k.startswith(Prefix)], "IsTruncated": False}

    def delete_objects(self, Bucket, Delete):
        keys = [o["Key"] for o in Delete["Objects"]]
        self.deleted += keys
        for k in keys:
            self.objects.pop(k, None)
        return {}


STAGED_OK = {"schema": 3, "materialized": True, "cleared": True, "request": ART}
OLD_OBJECTS = {"sagemaker/hub/xvla/_STAGED.json": 1200, "sagemaker/hub/xvla/hub/hub/x": 10,
               "sagemaker/hub/xvla/hub/models--facebook--bart-large/snapshots/cb48c1/tf_model.h5": 1_600_000_000}


def test_already_staged_requires_the_current_schema_and_the_same_request():
    assert hub.already_staged(FakeS3(STAGED_OK), "b", "xvla", ART) == STAGED_OK
    assert hub.already_staged(FakeS3({**STAGED_OK, "schema": 2}), "b", "xvla", ART) is None        # schema-2 manifests (unfiltered downloads) are re-staged
    unpinned = [dict(ART[0], revision=None), dict(ART[1], revision=None)]
    assert hub.already_staged(FakeS3({**STAGED_OK, "request": unpinned}), "b", "xvla", ART) is None  # pins change the request
    resolved = [{"repo": a["repo"], "requested_revision": None, "resolved_sha": a["revision"]} for a in ART]
    kept = {**STAGED_OK, "request": unpinned, "artifacts": resolved}
    assert hub.already_staged(FakeS3(kept), "b", "xvla", ART) == kept                       # unpinned staging that resolved to the pin: same bytes
    other = {**kept, "artifacts": [dict(resolved[0], resolved_sha="f" * 40), resolved[1]]}
    assert hub.already_staged(FakeS3(other), "b", "xvla", ART) is None                       # resolved to something else: re-stage
    assert hub.already_staged(FakeS3({**STAGED_OK, "materialized": False}), "b", "xvla", ART) is None
    assert hub.already_staged(FakeS3(None), "b", "xvla", ART) is None


def _stage_hub(monkeypatch, s3, *, dry):
    monkeypatch.setattr(hub, "client", lambda sess, svc: s3 if svc == "s3" else object())
    submitted, logs = [], []
    monkeypatch.setattr(hub.sm_jobs, "submit", lambda c, js: submitted.append(js["TrainingJobName"]))
    monkeypatch.setattr(hub, "wait_for_job", lambda *a, **k: None)
    sess = type("Sess", (), {"region_name": "ap-northeast-1", "profile_name": None})()
    prefix = hub.stage_hub(sess, STACK, CFG, "xvla", {"hub_artifacts": ART}, "123456789012.dkr.ecr.ap-northeast-1.amazonaws.com/x@sha256:" + "a" * 64,
                           run_name="r", dataset_s3="s3://b/d/", yes=True, dry=dry, log=logs.append, confirm=lambda q, y: logs.append("CONFIRM " + q))
    return prefix, submitted, logs


def test_stage_hub_deletes_the_stale_prefix_after_confirm_then_submits(monkeypatch):
    s3 = FakeS3({**STAGED_OK, "schema": 2}, OLD_OBJECTS)
    prefix, submitted, logs = _stage_hub(monkeypatch, s3, dry=False)
    assert prefix == "s3://b/sagemaker/hub/xvla/" and len(submitted) == 1
    assert sorted(s3.deleted) == sorted(OLD_OBJECTS) and not s3.objects
    stale = next(l for l in logs if "stale staging" in l)
    assert "3 object(s)" in stale and "1.60 GB" in stale and "schema below" in stale
    confirm = next(l for l in logs if l.startswith("CONFIRM"))
    assert "delete 3 stale object(s) under s3://b/sagemaker/hub/xvla/ and run the base-weight staging job?" in confirm
    assert logs.index(stale) < logs.index(confirm) < logs.index(next(l for l in logs if l.startswith("  deleted 3")))


def test_stage_hub_matching_manifest_deletes_nothing_and_submits_nothing(monkeypatch):
    s3 = FakeS3(STAGED_OK, OLD_OBJECTS)
    prefix, submitted, logs = _stage_hub(monkeypatch, s3, dry=False)
    assert prefix == "s3://b/sagemaker/hub/xvla/" and not submitted and not s3.deleted and s3.objects == OLD_OBJECTS
    assert any("already staged" in l for l in logs) and not any("stale" in l for l in logs)


def test_stage_hub_dry_run_reports_the_stale_prefix_but_keeps_it(monkeypatch):
    s3 = FakeS3({**STAGED_OK, "request": [dict(a, revision=None) for a in ART]}, OLD_OBJECTS)
    prefix, submitted, logs = _stage_hub(monkeypatch, s3, dry=True)
    assert not submitted and not s3.deleted and s3.objects == OLD_OBJECTS
    stale = next(l for l in logs if "stale staging" in l)
    assert "different artefact request" in stale and "3 object(s)" in stale and "dry run: kept" in stale
    assert not any(l.startswith("CONFIRM") for l in logs)


def test_stage_hub_without_any_manifest_deletes_nothing(monkeypatch):
    s3 = FakeS3(None, {})
    _, submitted, logs = _stage_hub(monkeypatch, s3, dry=False)
    assert len(submitted) == 1 and not s3.deleted and not any("stale" in l for l in logs)


def test_list_prefix_follows_continuation_tokens():
    class Paged:
        def list_objects_v2(self, Bucket, Prefix, ContinuationToken=None):
            if ContinuationToken is None:
                return {"Contents": [{"Key": "a", "Size": 1}], "IsTruncated": True, "NextContinuationToken": "t"}
            return {"Contents": [{"Key": "b", "Size": 2}], "IsTruncated": False}
    assert hub.list_prefix(Paged(), "b", "sagemaker/hub/x/") == (["a", "b"], 3)


def test_staging_spec_runs_hub_stage_with_network():
    spec = hub.staging_spec(STACK, CFG, "xvla", ART, "123456789012.dkr.ecr.ap-northeast-1.amazonaws.com/x@sha256:" + "a" * 64, run_name="r", dataset_s3="s3://b/sagemaker/datasets/d/")
    assert spec["AlgorithmSpecification"]["ContainerEntrypoint"] == ["python3", "/opt/ml/code/hub_stage.py"]
    assert spec["EnableNetworkIsolation"] is False and len(spec["InputDataConfig"]) == 1
    assert spec["CheckpointConfig"]["S3Uri"] == "s3://b/sagemaker/hub/xvla/" and json.loads(spec["HyperParameters"]["hub_artifacts"]) == ART
    assert "hf_token_secret" not in spec["HyperParameters"]
    # the launcher hands the staging job the solution user-agent token; hub_stage.py reads it (it has no repo to import from)
    from pipeline import solution
    assert spec["Environment"]["USER_AGENT_STRING"] == solution.user_agent_string() and solution.user_agent_string().startswith("AWSSOLUTION")


def test_read_secret_labels_the_call_with_the_injected_token(monkeypatch):
    """hub_stage.read_secret builds its client from USER_AGENT_STRING (user_agent_extra, slashes intact); unset = unlabelled."""
    import boto3
    seen = {}

    class _SM:
        def get_secret_value(self, SecretId):
            return {"SecretString": json.dumps({"token": "hf_x"})}

    def fake_client(service, **kw):
        seen.update(service=service, **kw)
        return _SM()

    monkeypatch.setattr(boto3, "client", fake_client)
    monkeypatch.setenv("AWS_REGION", "ap-northeast-1")
    monkeypatch.setenv("USER_AGENT_STRING", "InjectedToken/x/y")   # any value: it must land verbatim, slashes intact
    assert hub_stage.read_secret("p/hf_token") == "hf_x"
    assert seen["service"] == "secretsmanager" and seen["region_name"] == "ap-northeast-1"
    assert seen["config"].user_agent_extra == "InjectedToken/x/y"
    monkeypatch.delenv("USER_AGENT_STRING")
    hub_stage.read_secret("p/hf_token")
    assert not seen["config"].user_agent_extra


def test_gated_without_secret_is_refused(tmp_path):
    spec = {"hub_artifacts": [{"repo": "google/paligemma-3b-pt-224", "gated": True, "role": "tokenizer"}]}
    with pytest.raises(RuntimeError, match="Hugging Face token secret"):
        hub.stage_hub(type("Sess", (), {"region_name": "ap-northeast-1", "profile_name": None})(), STACK, CFG, "pi0", spec, "x@sha256:" + "a" * 64,
                      run_name="r", dataset_s3="s3://b/d/", yes=True, dry=True, log=lambda m: None)


def test_stage_writes_refs_main_for_sha_pinned_downloads(tmp_path):
    import hub_stage
    sha = "0123456789abcdef0123456789abcdef01234567"  # pragma: allowlist secret

    def fake_download(repo_id, revision, cache_dir, token, repo_type, allow_patterns=None, ignore_patterns=None):
        snap = os.path.join(cache_dir, "models--org--repo", "snapshots", sha)
        os.makedirs(snap); open(os.path.join(snap, "config.json"), "w").write("{}")
        return snap   # a sha-pinned snapshot_download creates no refs/main

    hub_stage.stage([{"repo": "org/repo", "revision": sha, "role": "base"}], str(tmp_path), downloader=fake_download, lister=_no_listing)
    ref = tmp_path / "hub" / "models--org--repo" / "refs" / "main"
    assert ref.read_text() == sha
