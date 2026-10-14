# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Load and validate the repo-root project_config.json.

    from pipeline.config import load_project_config
    cfg = load_project_config()                 # <repo>/project_config.json
    python3 -m pipeline.config --check          # CLI: exit 2 with a readable message on any problem

Checks beyond the JSON Schema (pipeline/project_config.schema.json): duplicate keys, every enabled policy has a registry
YAML under trainer/policies/ (when that folder exists), smoke.steps <= each enabled policy's steps, overrides
never touch flags the trainer owns. The licence gate is not here: it runs in pipeline/preflight.py via
trainer/licence.py (the same gate the job applies).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO / "project_config.json"
SCHEMA_PATH = Path(__file__).resolve().parent / "project_config.schema.json"
REGISTRY_DIR = REPO / "trainer" / "policies"
# Flags the trainer renders itself from the config; an override of one of these would silently fork the recipe.
OWNED_FLAGS = ("--dataset.repo_id", "--dataset.root", "--dataset.episodes", "--output_dir", "--job_name", "--steps",
               "--batch_size", "--save_freq", "--seed", "--policy.type", "--policy.device", "--policy.push_to_hub",
               "--resume", "--config_path", "--wandb.enable")
FORBIDDEN_FLAGS = ("--dataset.video_backend=torchcodec", "--policy.dtype=bfloat16", "--policy.dtype=float16")


class ConfigError(ValueError):
    """A problem in project_config.json, with a message meant for the user."""


def _no_duplicates(pairs):
    seen = {}
    for k, v in pairs:
        if k in seen:
            raise ConfigError(f"duplicate key {k!r} in project_config.json")
        seen[k] = v
    return seen


def _strip_comments(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _strip_comments(v) for k, v in obj.items() if not k.startswith("_comment")}
    if isinstance(obj, list):
        return [_strip_comments(v) for v in obj]
    return obj


def load_project_config(path: Path = CONFIG_PATH, schema_path: Path = SCHEMA_PATH) -> dict:
    try:
        raw = json.loads(Path(path).read_text(), object_pairs_hook=_no_duplicates)
    except FileNotFoundError:
        raise ConfigError(f"{path} not found; copy the shipped project_config.json and edit it") from None
    except json.JSONDecodeError as e:
        raise ConfigError(f"{path} is not valid JSON: {e}") from None
    try:
        import jsonschema
    except ImportError:  # pragma: no cover - requirements.txt installs it
        raise ConfigError("pip install -r requirements.txt (jsonschema missing)") from None
    schema = json.loads(Path(schema_path).read_text())
    # validator_for() picks the newest draft this jsonschema knows; the schema only uses Draft-7-compatible keywords
    validator = jsonschema.validators.validator_for(schema, default=jsonschema.Draft7Validator)(schema)
    errors = sorted(validator.iter_errors(raw), key=lambda e: list(e.absolute_path))
    if errors:
        lines = [f"  at {'/'.join(str(p) for p in e.absolute_path) or '<root>'}: {e.message}" for e in errors[:12]]
        if "aws" in raw:
            lines.append("  the aws block is gone: pass --region/--profile (or export AWS_DEFAULT_REGION), and use"
                         " `cdk deploy -c retainData=true` instead of retain_data_on_destroy")
        raise ConfigError("project_config.json does not match pipeline/project_config.schema.json:\n" + "\n".join(lines))
    cfg = _strip_comments(raw)
    _cross_check(cfg)
    return cfg


def enabled_policies(cfg: dict) -> dict:
    return {k: v for k, v in cfg["policies"].items() if v.get("enabled")}


def _cross_check(cfg: dict) -> None:
    problems = []
    ds = cfg["dataset"]
    if ds["source"] == "huggingface" and not ds.get("hf_repo_id"):
        problems.append("dataset.source=huggingface needs dataset.hf_repo_id")
    if ds["source"] == "s3" and not ds.get("s3_uri"):
        problems.append("dataset.source=s3 needs dataset.s3_uri")
    enabled = enabled_policies(cfg)
    if not enabled:
        problems.append("no policy is enabled (set policies.<name>.enabled=true)")
    smoke_steps = cfg.get("smoke", {}).get("steps", 200)
    for name, pol in enabled.items():
        if REGISTRY_DIR.is_dir() and not (REGISTRY_DIR / f"{name}.yaml").is_file():
            known = sorted(p.stem for p in REGISTRY_DIR.glob("*.yaml"))
            problems.append(f"policies.{name}: no registry entry trainer/policies/{name}.yaml (known: {', '.join(known)})")
        steps = pol.get("steps")
        if steps is not None and cfg.get("smoke", {}).get("enabled", True) and smoke_steps > steps:
            problems.append(f"policies.{name}.steps={steps} is below smoke.steps={smoke_steps}")
        for ov in pol.get("overrides", []):
            key = ov.split("=", 1)[0]
            if key in OWNED_FLAGS:
                problems.append(f"policies.{name}.overrides: {key} is set by the pipeline from this config; change the config field instead")
            if ov in FORBIDDEN_FLAGS:
                problems.append(f"policies.{name}.overrides: {ov} is forbidden (see docs/TRAPS.md)")
    comp = cfg.get("compute", {})
    if comp.get("training_plan_arn") and comp.get("spot"):
        problems.append("compute.training_plan_arn and compute.spot=true are exclusive: a training plan is reserved capacity, not Spot")
    if problems:
        raise ConfigError("project_config.json problems:\n  " + "\n  ".join(problems))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(CONFIG_PATH))
    ap.add_argument("--check", action="store_true", help="validate and print a one-line summary")
    args = ap.parse_args(argv)
    try:
        cfg = load_project_config(Path(args.config))
    except ConfigError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    pols = ", ".join(enabled_policies(cfg)) or "-"
    print(f"ok: project={cfg['project']} dataset={cfg['dataset']['name']} ({cfg['dataset']['source']}) "
          f"policies={pols} instance={cfg['compute']['instance_type']} image_tag={cfg['image']['tag']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
