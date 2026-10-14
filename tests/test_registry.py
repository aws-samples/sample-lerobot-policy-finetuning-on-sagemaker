# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""The registry renders exactly the validated ACT argv, obeys the resume/extend rules and the licence gate."""
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent
TRAINER = REPO / "trainer"
sys.path.insert(0, str(TRAINER))
import licence  # noqa: E402
import render_flags  # noqa: E402

ENV = dict(REPO_ID="local/pick", DS_ROOT="/opt/ml/input/data/scratch_360/pick_360", TRAIN_EPS="[1,2,3]", CHUNK="24", DEVICE="cuda",
           BATCH="8", STEPS="100000", SAVE_FREQ="10000", LOG_FREQ="200", NUM_WORKERS="12", SEED="1000",
           LR_OUT="/opt/ml/checkpoints/act/train", JOB_NAME="act")
# The exact argv of the validated ACT run (docs/RESULTS.md, ACT re-validation rows), in order, with VIDEO_BACKEND=pyav set.
GOLDEN = ["--dataset.repo_id=local/pick", "--dataset.root=/opt/ml/input/data/scratch_360/pick_360", "--dataset.episodes=[1,2,3]",
          "--dataset.video_backend=pyav", "--policy.type=act", "--policy.chunk_size=24", "--policy.n_action_steps=24",
          "--policy.device=cuda", "--policy.push_to_hub=false", "--batch_size=8", "--steps=100000", "--save_freq=10000",
          "--log_freq=200", "--num_workers=12", "--seed=1000", "--output_dir=/opt/ml/checkpoints/act/train", "--job_name=act",
          "--wandb.enable=false"]


def test_every_yaml_loads_and_has_the_schema_keys():
    for p in sorted((TRAINER / "policies").glob("*.yaml")):
        spec = render_flags.load_policy(p.stem)
        for key in ("name", "display_name", "tier", "status", "deliverable", "lerobot_type", "licence", "hardware", "train", "eval"):
            assert key in spec, (p.name, key)
        assert spec["status"] in ("validated", "trained", "smoke-tested", "stub")
        assert spec["licence"]["class"] in ("clean", "terms", "research_only", "non_commercial", "none", "unknown", "blocked")
        assert "verified_on" in spec["licence"] and "url" in spec["licence"]
        # docs/TRAPS.md cites this test as the proof that no policy can push weights to the Hub
        assert "--policy.push_to_hub=false" in spec["train"]["argv"], p.name


def test_act_fresh_matches_the_validated_argv_exactly():
    spec = render_flags.load_policy("act")
    assert render_flags.render(spec, "fresh", {**ENV, "VIDEO_BACKEND": "pyav"}) == GOLDEN
    without = render_flags.render(spec, "fresh", ENV)
    assert "--dataset.video_backend=pyav" not in without and len(without) == 17


# The LoRA flags of the fastwam_lora recipe (200-step smoke and 30k-step run of 2026-09-18 on an 80 GB H100, docs/RESULTS.md).
# PEFT does re.fullmatch on module names, so the target_modules regex must reach lerobot-train byte for byte.
LORA_TARGET = r".*\.blocks\.video\..*\.(q|k|v|o)"
LORA_PEFT = ["--peft.method_type=lora", "--peft.r=16", "--peft.lora_alpha=32", f"--peft.target_modules={LORA_TARGET}",
             '--peft.full_training_modules=["mixtures.action.action_encoder","mixtures.action.head","mixtures.action.time_embedding","mixtures.action.text_embedding","mixtures.action.time_projection","blocks.action","proprio_encoder"]']
LORA_ENV = {**ENV, "BATCH": "1", "STEPS": "20000", "SAVE_FREQ": "5000", "LR_OUT": "/opt/ml/checkpoints/fastwam_lora/train", "JOB_NAME": "fastwam_lora"}
LORA_GOLDEN = ["--dataset.repo_id=local/pick", "--dataset.root=/opt/ml/input/data/scratch_360/pick_360", "--dataset.episodes=[1,2,3]",
               "--policy.type=fastwam", "--policy.torch_dtype=float32", "--policy.freeze_video_expert=false",
               "--policy.use_gradient_checkpointing=true", *LORA_PEFT, '--policy.loss={"lambda_video":1.0,"lambda_action":1.0}',
               "--policy.device=cuda", "--policy.push_to_hub=false", "--batch_size=1", "--steps=20000", "--save_freq=5000",
               "--log_freq=200", "--num_workers=12", "--seed=1000", "--output_dir=/opt/ml/checkpoints/fastwam_lora/train",
               "--job_name=fastwam_lora", "--wandb.enable=false"]


def test_fastwam_lora_fresh_renders_the_smoke_tested_lora_argv():
    spec = render_flags.load_policy("fastwam_lora")
    out = render_flags.render(spec, "fresh", LORA_ENV)
    assert out == LORA_GOLDEN
    # exactly these peft flags, in this order, and nothing else under --peft.
    assert [t for t in out if t.startswith("--peft.")] == LORA_PEFT
    assert "--policy.freeze_video_expert=false" in out and "--policy.use_gradient_checkpointing=true" in out
    assert '--policy.loss={"lambda_video":1.0,"lambda_action":1.0}' in out
    # the regex survives YAML loading and rendering unchanged: it matches the video expert attention projections only
    (target,) = [t.split("=", 1)[1] for t in out if t.startswith("--peft.target_modules=")]
    assert target == LORA_TARGET
    assert re.fullmatch(target, "model.mot.layers.3.blocks.video.self_attn.q")
    assert re.fullmatch(target, "model.mot.layers.3.blocks.action.self_attn.q") is None
    assert re.fullmatch(target, "model.mot.mixtures.video.proj.q") is None       # the first attempt targeted mixtures.video and PEFT found nothing
    # the CLI path (what train_policy.sh reads) keeps the backslashes too
    r = subprocess.run([sys.executable, str(TRAINER / "render_flags.py"), "--policy", "fastwam_lora"], env={**os.environ, **LORA_ENV}, capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.strip().split("\n") == LORA_GOLDEN
    # what the recipe shares with fastwam.yaml and what it must differ in
    base = render_flags.load_policy("fastwam")
    for key in ("lerobot_type", "base_checkpoint", "hub_artifacts", "licence", "requires_language", "camera_count_divides", "smoke"):
        assert spec[key] == base[key], key
    # same evaluation protocol, including the strict reload check: both recipes reload exactly (diff 0.0) since 2026-09-22
    assert spec["eval"] == base["eval"] and not spec["eval"].get("stochastic")
    assert spec["publish"] == {"merge_adapter": True}
    assert spec["train"]["forbidden_flags"] == base["train"]["forbidden_flags"] and spec["train"]["scheduler"] == "none"
    assert "--policy.freeze_video_expert=true" in base["train"]["argv"] and not any(t.startswith("--peft.") for t in base["train"]["argv"])


def test_missing_variable_is_an_error():
    spec = render_flags.load_policy("act")
    with pytest.raises(render_flags.RenderError, match="STEPS"):
        render_flags.render(spec, "fresh", {k: v for k, v in ENV.items() if k != "STEPS"})


def test_resume_and_extend_rules():
    spec = render_flags.load_policy("act")
    assert render_flags.render(spec, "resume", ENV, config_path="/c/train_config.json") == ["--config_path=/c/train_config.json", "--resume=true"]
    ext = render_flags.render(spec, "extend", ENV, config_path="/c/train_config.json", steps=120000)
    assert ext == ["--config_path=/c/train_config.json", "--resume=true", "--steps=120000"]      # ACT: no scheduler flag
    sched = dict(spec); sched["train"] = {**spec["train"], "scheduler": "cosine_decay_with_warmup"}
    assert "--scheduler.num_decay_steps=120000" in render_flags.render(sched, "extend", ENV, config_path="/c/t.json", steps=120000)
    assert render_flags.render(spec, "resume", ENV, config_path="/c/t.json", overrides="--policy.chunk_size=10 --log_freq=5") == \
        ["--config_path=/c/t.json", "--resume=true", "--log_freq=5"]     # policy overrides dropped, others kept


def test_overrides_and_forbidden_flags():
    spec = render_flags.load_policy("act")
    out = render_flags.render(spec, "fresh", ENV, overrides="--policy.kl_weight=20 --log_freq=5 --job_name='a b'")
    assert out[-3:] == ["--policy.kl_weight=20", "--log_freq=5", "--job_name=a b"]
    for bad in ("--policy.dtype=bfloat16", "--dataset.video_backend=torchcodec"):
        with pytest.raises(render_flags.RenderError, match="forbidden"):
            render_flags.render(spec, "fresh", ENV, overrides=bad)


def test_cli_prints_one_token_per_line():
    r = subprocess.run([sys.executable, str(TRAINER / "render_flags.py"), "--policy", "act"], env={**os.environ, **ENV}, capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.strip().split("\n") == GOLDEN[:3] + GOLDEN[4:]
    r = subprocess.run([sys.executable, str(TRAINER / "render_flags.py"), "--policy", "nope"], capture_output=True, text=True)
    assert r.returncode == 2 and "no registry entry" in r.stderr


def test_licence_gate(tmp_path):
    spec = render_flags.load_policy("act")
    assert licence.gate(spec)[0]
    def with_class(cls, **extra):
        s = dict(spec); s["licence"] = {**spec["licence"], "class": cls, **extra}; return s
    ok, msg = licence.gate(with_class("terms"))
    assert not ok and "accept_licence" in msg
    assert licence.gate(with_class("terms"), accept=True)[0]
    ok, msg = licence.gate(with_class("terms", requires_approval=True), accept=True)
    assert not ok and "approval_ref" in msg
    assert licence.gate(with_class("terms", requires_approval=True), accept=True, approval_ref="TICKET-1")[0]
    assert not licence.gate(with_class("blocked"), accept=True, approval_ref="x")[0]
    text = licence.notice_text(spec, dataset_name="d", image_digest="sha256:abc")
    assert "contains no model weights" in text and "Apache-2.0" in text
    r = subprocess.run([sys.executable, str(TRAINER / "licence.py"), "act"], capture_output=True, text=True)
    assert r.returncode == 0


def test_eval_stochastic_flag_is_boolean_and_honoured_by_the_entrypoint():
    """eval.stochastic makes reload_check.py compare shapes only; no shipped recipe needs it since the 2026-09-22 fix."""
    entrypoint = (REPO / "trainer" / "sagemaker" / "train_entrypoint.sh").read_text()
    assert "get('stochastic', False)" in entrypoint and "--stochastic" in entrypoint
    for p in sorted((REPO / "trainer" / "policies").glob("*.yaml")):
        with open(p) as f:
            ev = (yaml.safe_load(f).get("eval") or {})
        if "stochastic" in ev:
            assert isinstance(ev["stochastic"], bool), p.name
    for name in ("fastwam", "fastwam_lora"):   # both reload exactly (diff 0.0) once the LoRA alias defect was fixed on 2026-09-22
        with open(REPO / "trainer" / "policies" / f"{name}.yaml") as f:
            assert not (yaml.safe_load(f)["eval"].get("stochastic")), name


def test_fastwam_lora_full_training_modules_are_leaf_modules():
    """A whole-expert entry ("action" = mot.mixtures.action) installs the trainable copy where FastWAM's forward never looks
    (it uses the unregistered alias action_expert): the action encoder and head stayed at their default init on the
    2026-09-21 20k run. Every entry must name a leaf module of the expert, a per-layer block, or the proprio encoder."""
    import json
    spec = render_flags.load_policy("fastwam_lora")
    flag = next(t for t in spec["train"]["argv"] if t.startswith("--peft.full_training_modules="))
    mods = json.loads(flag.split("=", 1)[1])
    assert "action" not in mods and "mixtures.action" not in mods
    for m in mods:
        assert m == "proprio_encoder" or m == "blocks.action" or m.startswith("mixtures.action."), m
    for leaf in ("mixtures.action.action_encoder", "mixtures.action.head"):
        assert leaf in mods


def test_eval_openloop_rebuilds_the_merged_adapter_policy():
    """After PeftModel.merge_and_unload() the adapter path must rebuild a fresh policy from the state dict, or FastWAM's
    aliases keep the untrained originals (reload differed from eval by up to 251 degrees, 2026-09-22)."""
    src = (REPO / "trainer" / "eval_openloop.py").read_text()
    assert "merge_and_unload()" in src and "_merged_state_dict(" in src
    body = src.split("state = _merged_state_dict(")[1].split("def _merged_state_dict(")[0]
    assert "fresh = cls(cfg)" in body and "load_state_dict(state" in body and "torch.cuda.empty_cache()" in body


def test_reload_check_links_the_checkpoint_instead_of_copying(tmp_path):
    """reload_check must not copy a checkpoint to patch its config: a merged FastWAM copy is 24 GB and filled the container
    filesystem on the 2026-09-22 p5 smoke (ENOSPC). Weights are symlinked, config.json is a private copy."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("reload_check", REPO / "trainer" / "reload_check.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    src = tmp_path / "pretrained_model"; src.mkdir(); (src / "model.safetensors").write_bytes(b"w" * 10); (src / "config.json").write_text("{}")
    (src / "sub").mkdir(); (src / "sub" / "x.bin").write_bytes(b"x")
    dst = tmp_path / "view" / "pretrained_model"
    mod.link_checkpoint_view(str(src), str(dst))
    assert (dst / "model.safetensors").is_symlink() and (dst / "sub").is_symlink() and not (dst / "config.json").is_symlink()
    (dst / "config.json").write_text('{"patched": true}')
    assert (src / "config.json").read_text() == "{}"
    assert "copytree(src, dst)" not in (REPO / "trainer" / "reload_check.py").read_text()
    assert 'copytree(os.path.join(a.model_dir, "pretrained_model")' not in (REPO / "inference" / "predict.py").read_text()


# (policy name, repo) pairs allowed to keep `revision: null`. Empty: every shipped hub_artifacts entry pins a commit sha
# (docs/SECURITY.md, Supply chain; docs/THREAT_MODEL.md T-05). To add one, list it here AND put the reason in a comment on
# the YAML line, so a reviewer can see why that artefact tracks the Hub default branch.
UNPINNED_ALLOWLIST: set = set()
COMMIT_SHA = re.compile(r"^[0-9a-f]{7,40}$")


def test_every_hub_artifact_pins_a_commit_sha():
    """A null revision resolves the Hub default branch at staging time, so the base weights could change between two
    stagings with nothing in the registry recording it. Every entry, stub or not, names a commit sha (7 to 40 hex digits;
    lingbot_va uses the 8-digit prefix), never a branch or tag name."""
    seen, with_artifacts = 0, 0
    for p in sorted((TRAINER / "policies").glob("*.yaml")):
        with open(p) as f:
            spec = yaml.safe_load(f)
        artifacts = spec.get("hub_artifacts") or []
        with_artifacts += bool(artifacts)
        for a in artifacts:
            seen += 1
            rev = a.get("revision")
            if (spec["name"], a["repo"]) in UNPINNED_ALLOWLIST:
                assert rev is None, (p.name, a["repo"], "listed as unpinned but carries a revision: drop it from the allowlist")
                continue
            assert isinstance(rev, str) and COMMIT_SHA.match(rev), (p.name, a["repo"], rev)
    assert seen >= 30 and with_artifacts >= 15      # 15 policies, 30 artefacts today; a shrinking registry is a separate change
    # docs/SECURITY.md states the count; keep the sentence true
    security = (REPO / "docs" / "SECURITY.md").read_text()
    assert f"All {with_artifacts} entries with `hub_artifacts` pin every artefact to a commit sha" in security
