#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Render the lerobot-train argv for a policy from its registry YAML + the environment. One token per line on stdout.

    render_flags.py --policy act --mode fresh                         # ${VAR} from the environment (STEPS, BATCH, ...)
    render_flags.py --policy act --mode resume --config-path <ckpt>/pretrained_model/train_config.json
    render_flags.py --policy act --mode extend --config-path ... --steps 120000
    OVERRIDES='--policy.kl_weight=20 --log_freq=50' render_flags.py --policy act --mode fresh

Rules encoded here (each learned the hard way):
  resume = ONLY --config_path=<train_config.json> --resume=true  (a --policy.* flag together with --resume crashes lerobot 0.6.1)
  extend = resume + --steps=N; policies with a decay scheduler also get the TOP-LEVEL --scheduler.num_decay_steps=N
           (--policy.scheduler_decay_steps is silently ignored on resume)
  forbidden flags (from the YAML) are refused whether they come from the YAML or from OVERRIDES
Python 3.9+ with PyYAML; no other dependency, so it runs on the laptop and inside the image.
"""
from __future__ import annotations

import argparse
import os
import re
import shlex
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
POLICIES_DIR = HERE / "policies"
VAR = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)\}")


class RenderError(ValueError):
    pass


def load_policy(name: str, policies_dir: Path = POLICIES_DIR) -> dict:
    p = policies_dir / f"{name}.yaml"
    if not p.is_file():
        known = sorted(x.stem for x in policies_dir.glob("*.yaml"))
        raise RenderError(f"no registry entry for policy {name!r} (known: {', '.join(known)})")
    with open(p) as f:
        spec = yaml.safe_load(f)
    for key in ("name", "lerobot_type", "status", "licence", "train"):
        if key not in spec:
            raise RenderError(f"{p.name}: missing top-level key {key!r}")
    if spec["name"] != name:
        raise RenderError(f"{p.name}: name field is {spec['name']!r}")
    return spec


def substitute(token: str, env: dict) -> str:
    def repl(m):
        v = env.get(m.group(1))
        if v is None or v == "":
            raise RenderError(f"{token}: environment variable {m.group(1)} is not set")
        return v
    return VAR.sub(repl, token)


def flag_key(token: str) -> str:
    return token.split("=", 1)[0]


def _joined(tokens: list) -> list:
    """`--flag value` pairs joined as `--flag=value` (draccus accepts both spellings)."""
    out, i = [], 0
    while i < len(tokens):
        t = tokens[i]
        if t.startswith("--") and "=" not in t and i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
            out.append(f"{t}={tokens[i + 1]}"); i += 2
        else:
            out.append(t); i += 1
    return out


def check_forbidden(tokens: list, forbidden: list) -> None:
    tokens = _joined(tokens)
    bad = [t for t in tokens if t in forbidden or flag_key(t) in {flag_key(f) for f in forbidden if "=" not in f}]
    if bad:
        raise RenderError(f"forbidden flag(s) {bad}: see trainer/policies/<policy>.yaml forbidden_flags and docs/TRAPS.md")


def render(spec: dict, mode: str, env: dict, *, config_path: str = None, steps: int = None, overrides: str = "") -> list:
    train = spec["train"]
    forbidden = list(train.get("forbidden_flags", []))
    extra = shlex.split(overrides or "")
    if mode == "fresh":
        tokens = [substitute(t, env) for t in train["argv"]]
        for opt in train.get("optional_argv", []):
            if env.get(opt["when"]):
                tok = substitute(opt["arg"], env)
                anchor = opt.get("after")
                idx = next((i for i, t in enumerate(tokens) if flag_key(t) == anchor), len(tokens) - 1) + 1 if anchor else len(tokens)
                tokens.insert(idx, tok)
        if str(env.get("SMOKE", "")) == "1":   # smoke-only flags (VQ-BeT: finish the VQ-VAE phase inside the 200 steps)
            for tok in [substitute(t, env) for t in (spec.get("smoke") or {}).get("argv", [])]:
                tokens = [t for t in tokens if flag_key(t) != flag_key(tok)] + [tok]   # a smoke value replaces the training value
    elif mode in ("resume", "extend"):
        if not config_path:
            raise RenderError(f"--mode {mode} needs --config-path <checkpoint>/pretrained_model/train_config.json")
        tokens = [f"--config_path={config_path}", "--resume=true"]
        if mode == "extend":
            if not steps:
                raise RenderError("--mode extend needs --steps N")
            tokens.append(f"--steps={steps}")
            if str(train.get("scheduler", "none")).lower() != "none":
                tokens.append(f"--scheduler.num_decay_steps={steps}")
        dropped = [t for t in extra if t.startswith("--policy.")]
        if dropped:  # lerobot rebuilds the policy from train_config.json; a --policy.* flag next to --resume crashes it
            print(f"render_flags: dropping policy overrides on {mode} (already inside train_config.json): {dropped}", file=sys.stderr)
            extra = [t for t in extra if not t.startswith("--policy.")]
    else:
        raise RenderError(f"unknown mode {mode!r}")
    tokens += extra
    check_forbidden(tokens, forbidden)
    return tokens


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--mode", choices=["fresh", "resume", "extend"], default="fresh")
    ap.add_argument("--config-path")
    ap.add_argument("--steps", type=int)
    ap.add_argument("--overrides", default=os.environ.get("OVERRIDES", ""))
    ap.add_argument("--policies-dir", default=str(POLICIES_DIR))
    ap.add_argument("--print", dest="print_key", help="print one top-level registry field (e.g. lerobot_type) and exit")
    a = ap.parse_args(argv)
    try:
        spec = load_policy(a.policy, Path(a.policies_dir))
        if a.print_key:
            val = spec
            for part in a.print_key.split("."):   # dotted keys: eval.n_samples
                val = val[part]
            print(val)
            return 0
        tokens = render(spec, a.mode, dict(os.environ), config_path=a.config_path, steps=a.steps, overrides=a.overrides)
    except RenderError as e:
        print(f"render_flags: {e}", file=sys.stderr)
        return 2
    print("\n".join(tokens))
    return 0


if __name__ == "__main__":
    sys.exit(main())
