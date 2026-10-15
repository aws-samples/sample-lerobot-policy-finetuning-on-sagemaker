#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Render docs/POLICIES.md and the generated blocks of README.md from the registry (trainer/policies/*.yaml).
Run after any registry change:

    python3 -m pipeline.policies_doc            # writes docs/POLICIES.md and the marked blocks of README.md
    python3 -m pipeline.policies_doc --check    # exit 1 when either file is stale (CI)

The README blocks sit between `<!-- BEGIN GENERATED <name> ... -->` and `<!-- END GENERATED <name> -->` comments:
the licence disclaimer, the capability matrix, the scenario-fit table and the instance fallback tables. The prose
around them is hand-written. The scenario rows (SCENARIOS), the one-line descriptions (WHAT_IT_IS) and the licence
link texts (LICENCE_LABEL) are data in this file so that README.md and docs/POLICIES.md render from one source.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import textwrap
from pathlib import Path

import yaml

from pipeline import instance_table, plan

REPO = Path(__file__).resolve().parent.parent
POLICIES = REPO / "trainer" / "policies"
OUT = REPO / "docs" / "POLICIES.md"
README = REPO / "README.md"
PROJECT_CONFIG = REPO / "project_config.json"
TIER_LABEL = {0: "0 · from scratch", 1: "1 · opt-in research", 2: "2 · opt-in, large-GPU"}   # a grouping, not a status; deliverability is its own column
STATUS_ORDER = {"validated": 0, "trained": 1, "smoke-tested": 2, "stub": 3}
PRICE_NOTE = "ap-northeast-1 on-demand list prices, 2026-09"   # the region label pipeline/plan.py prints under its plan table (tests check it)

# One line per offered policy for the README matrix (docs/POLICIES.md uses the registry's display_name).
WHAT_IT_IS = {
    "act": "Action Chunking Transformer, trained from scratch",
    "diffusion": "Diffusion Policy (U-Net, DDPM), trained from scratch",
    "vqbet": "VQ-BeT behaviour transformer, trained from scratch",
    "tdmpc": "TD-MPC latent world model with CEM planning, trained from scratch",
    "smolvla": "SmolVLA: an action expert trained from scratch on a frozen SmolVLM2-500M backbone",
    "xvla": "X-VLA 0.9B",
    "multi_task_dit": "Multi-Task DiT with CLIP conditioning",
    "pi0": "pi0 (PaliGemma-3B + flow-matching expert)",
    "pi05": "pi0.5 (PaliGemma-3B + flow-matching expert, state as prompt tokens)",
    "evo1": "EVO-1 (InternVL3-1B + flow-matching head)",
    "groot": "NVIDIA GR00T N1.7 3B",
    "molmoact2": "MolmoAct2 (Molmo2-4B + action expert)",
    "lingbot_va": "LingBot-VA video-action world model (LoRA fine-tune)",
    "pi0_fast": "pi0-FAST (autoregressive action tokens)",
    "eo1": "EO-1 (Qwen2.5-VL-3B + flow-matching head)",
    "fastwam": "FastWAM (Wan2.2 world-action model)",
    "fastwam_lora": "FastWAM, second recipe: LoRA on the video expert, action expert in full training; publishes an adapter and a merged checkpoint",
}

# Link text of the licence cell for every non-clean entry (clean entries show their SPDX id). The text names the terms
# document, never a bare SPDX id, and carries the registry class word (tests/test_readme_policies.py CLASS_WORDS).
LICENCE_LABEL = {
    "xvla": "Apache-2.0 Hub tag; terms apply (pretraining data includes a CC BY-NC-SA corpus)",
    "multi_task_dit": "none declared (CLIP encoder)",
    "pi0": "Gemma Terms of Use",
    "pi05": "Gemma Terms of Use",
    "pi0_fast": "Gemma Terms of Use",
    "evo1": "unclear: MIT or Qwen (InternVL3-1B)",
    "groot": "NVIDIA Open Model License",
    "molmoact2": "research only (Ai2 Responsible Use)",
    "lingbot_va": "unresolved (Apache-2.0 Hub tag, no licence text)",
    "eo1": "Qwen Research Licence (non-commercial)",
    "fastwam": "unresolved (Apache-2.0 Hub tag, no licence text)",
    "fastwam_lora": "unresolved (Apache-2.0 Hub tag, no licence text)",
}

# "Which policy for which situation". Every policy named must be offered; `gpu_gb` rows may only name policies whose
# hardware.min_gpu_memory_gb fits; `clean_only` rows only licence class `clean`; `min_action_cap` rows only policies
# without max_action_dim or with a cap above that value (tests/test_readme_policies.py).
SCENARIOS = [
    {"situation": "Single-task imitation from your own recordings, 48 GB GPU", "recommended": "act", "alternatives": ["diffusion", "vqbet"],
     "why": "trained from scratch, no base weights to license, the only validated entry and the shipped default", "gpu_gb": 48},
    {"situation": "Language-conditioned, several tasks in one dataset", "recommended": "smolvla", "alternatives": ["pi05", "pi0", "xvla", "groot", "molmoact2"],
     "why": "SmolVLA is licence-clean and was trained end to end here; of the three chunk-50 VLAs (pi0, pi0.5, SmolVLA) pi0.5 had the lowest ratio to the hold-still baseline and X-VLA (chunk 24) the lowest absolute joints error (docs/RESULTS.md); pi0.5 inherits the Gemma terms", "gpu_gb": 48},
    {"situation": "An 80 GB GPU is available", "recommended": "pi05", "alternatives": ["pi0_fast", "eo1", "fastwam", "fastwam_lora"],
     "why": "the four alternatives need 80 GB and are smoke-tested only (on EC2 H100 hosts); pi0.5 is a trained entry that also lists `ml.p5.4xlarge`", "gpu_gb": 80},
    {"situation": "Reward-labelled data (`next.reward` in the dataset)", "recommended": "tdmpc", "alternatives": [],
     "why": "the only offered policy that learns from rewards; recorded demonstrations carry none, so it is not deliverable"},
    {"situation": "One camera only", "recommended": "act", "alternatives": ["vqbet", "diffusion"],
     "why": "every entry accepts a single camera; VQ-BeT and TD-MPC accept nothing else, so one camera is no restriction here"},
    {"situation": "Lowest-cost first run", "recommended": "act", "alternatives": ["vqbet", "diffusion"],
     "why": "the measured 100k-step ACT run cost about $10 on `ml.g6e.4xlarge`; ACT and VQ-BeT also run on the 24 GB and 16 GB classes, and the smoke job alone costs under $1"},
    {"situation": "Licence-clean bases only (nothing to accept)", "recommended": "act", "alternatives": ["diffusion", "vqbet", "smolvla", "tdmpc"],
     "why": "registry class `clean`, no `accept_licence`; act, diffusion and vqbet are also deliverable", "clean_only": True},
    {"situation": "Actions wider than 32 dimensions (bimanual, mobile base)", "recommended": "act", "alternatives": ["diffusion", "vqbet", "groot"],
     "why": "pi0, pi0.5, pi0-FAST, SmolVLA, EO-1 and MolmoAct2 pad actions to 32 dimensions and refuse wider datasets (`max_action_dim`); act, diffusion and vqbet read the width from the dataset and groot pads to 132", "min_action_cap": 32},
    {"situation": "World-model or video-conditioned policy", "recommended": "fastwam", "alternatives": ["fastwam_lora", "lingbot_va"],
     "why": "Wan2.2-based world-action models that predict video and actions together; lingbot_va fits 48 GB with LoRA but trains slowly; all three have an unresolved licence and need `approval_ref`"},
]


def disclaimer(specs: list) -> str:
    """The model-licence blockquote shared by README.md and docs/POLICIES.md, built from the offered non-clean entries:
    a warning line, one bullet per licence group (the entries whose licence position is not established come last),
    then the gate and the MIT-0 note."""
    named, unresolved = {}, []
    for s in offered(specs):
        lic = s.get("licence") or {}
        if lic.get("class") == "clean":
            continue
        if lic.get("class") == "unknown":   # no licence position established for the weights (see each entry's notice)
            unresolved.append(s["name"])
        else:
            named.setdefault(LICENCE_LABEL[s["name"]], []).append(s["name"])

    def quote(text: str, indent: str = "") -> list:
        width = 118 - len(indent)
        wrapped = textwrap.wrap(text, width=width, break_long_words=False, break_on_hyphens=False)
        return [f"> {indent if i else ''}{line}" for i, line in enumerate(wrapped)]

    lines = quote("⚠️ **Model licences.** This repository distributes no model weights and no datasets. For every policy that fine-tunes "
                  "a published checkpoint, the training job downloads the base weights from the Hugging Face Hub into your account, "
                  "under the publisher's terms. Several of those bases are not open-source or carry conditions:")
    lines.append(">")
    for label, names in named.items():
        lines += quote(f"- {label}: {_codes(names)}", indent="  ")
    lines += quote(f"- no licence position established for the base weights: {_codes(unresolved)} (an Apache-2.0 Hub tag without "
                   "licence text, no licence declared, or conflicting statements; the per-policy notice in docs/POLICIES.md records "
                   "what was found, and the publisher's terms for the weights and their training data have not been confirmed)", indent="  ")
    lines.append(">")
    lines += quote("Production or commercial use of any of these needs your own legal review. The job refuses to start until "
                   "`accept_licence: true` (and `approval_ref` where the table shows it) is set for the policy in `project_config.json`, "
                   "and it writes `LICENCE_NOTICE.md` next to every output. Read the linked terms before enabling a policy so you can make "
                   "an informed decision. This note concerns the model weights only: the code in this repository is MIT-0.")
    return "\n".join(lines)


def _codes(names) -> str:
    return ", ".join(f"`{n}`" for n in names)


HEADER_TOP = """# Policies: tiers, status and licences

Generated by `python3 -m pipeline.policies_doc` (CI runs `--check`). The tables and the per-policy notes come from
`trainer/policies/*.yaml`. The text around the tables lives in `pipeline/policies_doc.py`. Edit those files, not this one.
"""

HEADER = """
**Status** says what has been run through this pipeline:

- `validated`: a full run reproduced the reference numbers recorded for the same recipe in [docs/RESULTS.md](RESULTS.md),
  inside the acceptance band of `pipeline/reference_results.json`. The reference is an earlier run of ours, not a paper result.
- `trained`: a full fine-tune at the entry's `defaults` (steps, batch, chunk) finished on the public dataset. Its metrics are
  in [docs/RESULTS.md](RESULTS.md) and `pipeline/reference_results.json`. No claim beyond that.
- `smoke-tested`: a 200-step GPU job on a public dataset trained, evaluated, published and reloaded. Most ran on SageMaker.
  `pi0_fast`, `eo1`, `fastwam` and `fastwam_lora` ran on an EC2 H100 host with the same container image, input channels and
  environment variables as a SageMaker job ([docs/RESULTS.md](RESULTS.md)).
  The first `fastwam_lora` smoke ran on a private dataset and continued from an earlier `fastwam` fine-tune (advanced flow).
  The shipped default (`lerobot/fastwam_base` on `hjkso1406/so101-4tasks-100eps`) then passed its own 200-step smoke on an
  EC2 H100 host on 2026-09-22, with the adapter merge and an exact reload of the merged copy ([docs/RESULTS.md](RESULTS.md)).
- `stub`: the entry renders its flags, but preflight refuses it for the reason in the notes column
  (`./finetune.sh --allow-stubs` turns the refusal into a warning). The three stubs below are also marked
  `out_of_scope: true`, so they are not offered at all: the compatibility step refuses them even with `--allow-stubs`.

**Deliverable** (column): the checkpoint is entirely your own work. It was trained from scratch on your recordings and
inherits no licence, so you can ship it as your model. `tdmpc` and `gaussian_actor` are licence-clean but learn from
reward labels, which recorded demonstrations do not have, so they are not deliverable.

**Tier** groups entries by what you take on, not by how far they have been tested. Tier 0 trains from scratch under a
clean licence. Tier 1 entries are opt-in research policies that fit the default 48 GB GPU. Tier 2 entries need a large
GPU, a base with restrictive or unresolved publisher terms, or both. Any tier can hold any status: a tier-2 entry can
be `trained`.

**Licence class of the base weights**: `clean` (no inherited weights, or Apache/MIT/BSD weights), `terms` (usable after
you accept the publisher's terms), `research_only` / `non_commercial` (no commercial use), `none` (the publisher declared
no licence, so all rights are reserved by default), `unknown`, `blocked` (not offered). Every class except `clean` and
`blocked` needs `accept_licence: true` under `policies.NAME` in `project_config.json`. Accepting a licence there grants
nothing the publisher has not granted, and a fine-tune inherits the restrictions of its base.

**Commercial use** (column): `yes` means the weights add no restriction. `terms` means allowed only within the publisher's
terms, so get legal sign-off. `no` means research or non-commercial use only. `unknown` means nobody has established the
publisher's position.

**Inputs** (column): how many camera streams the entry takes (`one camera`, `several cameras`, or `several cameras
(mapping required)` when the dataset cameras must be mapped onto the base checkpoint's slots), the number of frames per
camera when the policy reads more than one, joint state (every entry), `language` when the dataset task string conditions
the policy (a single-task dataset makes that input a constant), `reward labels` when the loss needs `next.reward`.

**Action space** (column): `up to N dims` when the registry caps the action width (`max_action_dim`: the policy pads to N
and refuses wider datasets), otherwise `dataset width`; `chunk` is the default action chunk from `defaults.chunk`
(informational for `tdmpc`, whose horizon is fixed by the recipe).

**VRAM** (column): peak GPU memory at the entry's default batch size, from the registry's `hardware.vram_gb`. It is a
measured value for `act`, `xvla`, `pi0_fast`, `eo1`, `fastwam` and `fastwam_lora` and an estimate for the others; every smoke test confirmed
that the policy fits its listed instance type. **GPU and default instance**: the smallest GPU memory class the entry
accepts (`hardware.min_gpu_memory_gb`) and the instance the shipped `project_config.json` resolves to for it;
`(untested)` marks a type whose GPU facts in `pipeline/instance_table.py` are unverified. The instance types per memory class, with quota names and prices, are in the README under Quotas.

**Default budget and estimate** (columns): `defaults.steps` x `defaults.batch_size`, and the hours and dollars
`./finetune.sh --plan` would print for that budget on the default instance from the registry's step rate for that GPU,
including the 200-step smoke job and the fixed start-up and evaluation overhead (`pipeline/plan.py`). Prices are
approximate PRICE_NOTE_PLACEHOLDER; your region may differ. The estimate is for planning and errs long: the recorded
runs in [docs/RESULTS.md](RESULTS.md) finished faster than the registry step rate predicts.

**Precision** (per-policy notes): `fp32` trains entirely in 32-bit floats. `fp32_masters_bf16_autocast` keeps the weights
and optimizer state in fp32 and runs the forward pass in bfloat16 through `trainer/amp_driver.py` (the bf16 trap it avoids
is in [docs/TRAPS.md](TRAPS.md)).

This repository ships no weights. A staging job downloads the base into your account and records the exact commit in
`_STAGED.json` ([docs/SECURITY.md](SECURITY.md)).

**LoRA recipes.** A registry entry is a recipe, and one lerobot policy type can have more than one. `fastwam_lora` is a
second recipe for the `fastwam` type: it adapts the 5B video expert with LoRA (rank 16 on the attention projections) while
the action expert and the proprio encoder train in full, with the video loss on. `fastwam` itself freezes the video expert.
`lingbot_va` also trains with LoRA, because its full fine-tune does not fit a 48 GB GPU. A LoRA run publishes an adapter
(`adapter_model.safetensors` plus `adapter_config.json`), which needs the base weights at load time. Recipes that set
`publish.merge_adapter: true` (today `fastwam_lora`) also publish the adapter merged into full weights under
`pretrained_model_merged/`, which a server loads like a plain checkpoint without peft. `fastwam_lora` has its own hub prefix,
so enabling it next to `fastwam` stages the same 46 GB of base weights a second time. Advanced flow, continuing from your
own earlier checkpoint instead of the published base: copy that checkpoint's `pretrained_model/` directory into the
policy's hub prefix in your bucket (`sagemaker/hub/<policy>/<dir>/pretrained_model/`, which the job mounts at
`/opt/ml/input/data/hub/<dir>/pretrained_model`), then add `--policy.pretrained_path=/opt/ml/input/data/hub/<dir>/pretrained_model`
to `policies.<policy>.overrides` in `project_config.json`. Copy it after the staging job has run: a re-stage deletes
everything under that prefix. Evaluation and reload still merge the adapter onto the registry's `base_checkpoint`, which
is exact when your earlier checkpoint kept the video expert frozen (the `fastwam` recipe).

> **Non-clean rows below are research entries.** Read the licence text linked in the table before enabling them, and
> consult your legal team before shipping anything derived from them.

## How the gate decides and where it runs

The same `licence.gate` function (`trainer/licence.py`) runs three times. First on your laptop during `preflight`: a
refusal submits nothing. Then inside every smoke and training job, in `train_policy.sh` before `lerobot-train` starts.
Finally in `publish.py` at the end of the job, which records the outcome as `licence_status` in SUMMARY.json and
LICENCE_NOTICE.md. The hub staging job has no gate of its own. It runs only after the laptop gate has passed.

An `approval_ref` (your ticket or sign-off reference) under `policies.NAME` is required in two cases:

1. The YAML sets `licence.requires_approval`. The gate checks this on the laptop and in the job.
2. `licence.class` is `research_only`. The plan step (`pipeline/plan.py`) checks this on the laptop before anything is
   submitted, even when the YAML says `requires_approval: false` (today `eo1` and `molmoact2`).

```mermaid
flowchart TD
  subgraph where["Where the gate runs: licence.gate in trainer/licence.py, three times per run"]
    pf["preflight on the laptop: local_checks gates every enabled policy in the run, offline"]
    tp["SageMaker smoke and training jobs: train_policy.sh runs licence.py before lerobot-train; publish.py runs it again at the end and records licence_status"]
  end
  pf --> cls
  tp --> cls
  subgraph gate["licence.gate: NAME.yaml licence.class vs project_config.json policies.NAME"]
    cls{"licence.class"}
    cls -->|clean| pass["pass"]
    cls -->|blocked| refb["not offered: accept_licence is ignored"]
    cls -->|"terms, research_only, non_commercial, none, unknown"| acc{"accept_licence: true?"}
    cls -->|any other value| rej
    acc -->|no| refa["read licence.url, then set policies.NAME.accept_licence: true"]
    acc -->|yes| appr{"licence.requires_approval in the YAML?"}
    appr -->|no| pass
    appr -->|yes| aref{"policies.NAME.approval_ref set?"}
    aref -->|no| refr["approval_ref needed in addition to accept_licence"]
    aref -->|yes| pass
  end
  pass -->|laptop| plan{"plan step, pipeline/plan.py: licence.class research_only without approval_ref?"}
  plan -->|yes| ex4p["plan refuses the row: finetune.sh exits 4, no job is submitted"]
  plan -->|no| go["preflight continues; smoke and training jobs receive accept_licence and approval_ref as hyperparameters"]
  go --> stg["hub staging job (hub_artifacts policies): no gate of its own, relies on this laptop pass"]
  pass -->|job| train["lerobot-train starts; publish.py later writes LICENCE_NOTICE.md and summary.py records licence_status"]
  refb --> rej["refused"]
  refa --> rej
  refr --> rej
  rej -->|laptop| ex4["preflight FAIL row: PreflightError, finetune.sh exits 4, no job is submitted"]
  rej -->|job| ex3["licence.py exits 3: train_policy.sh exits 3 and the job fails before training"]
```

## Every registry entry
""".replace("PRICE_NOTE_PLACEHOLDER", PRICE_NOTE)


def load_all() -> list:
    specs = []
    for p in sorted(POLICIES.glob("*.yaml")):
        with open(p) as f:
            specs.append(yaml.safe_load(f))
    return sorted(specs, key=lambda s: (s.get("tier", 9), STATUS_ORDER.get(s.get("status"), 3), s["name"]))


def offered(specs: list) -> list:
    return [s for s in specs if not s.get("out_of_scope")]


def load_project_config(path: Path = PROJECT_CONFIG) -> dict:
    with open(path) as f:
        return json.load(f)


def default_instance(spec: dict, cfg: dict) -> str:
    """The instance the shipped project_config.json resolves to for this policy (pipeline/run.py: policies.NAME.instance_type,
    else compute.instance_type). An entry the shipped file does not configure, whose GPU need exceeds that default (the
    out-of-scope stubs), shows the first type its registry entry lists instead."""
    pol = (cfg.get("policies") or {}).get(spec["name"])
    pol = pol if isinstance(pol, dict) else {}
    inst = pol.get("instance_type") or cfg["compute"]["instance_type"]
    need = (spec.get("hardware") or {}).get("min_gpu_memory_gb") or 0
    have = instance_table.spec(inst)
    listed = (spec.get("hardware") or {}).get("instance_types") or []
    if not pol and listed and have and have.gpu_mem_gb < need:
        return listed[0]
    return inst


def _untested(instance: str) -> str:
    s = instance_table.spec(instance)
    return "" if (s and s.verified) else " (untested)"


def inputs_cell(s: dict) -> str:
    frames = (s.get("serving") or {}).get("frames_per_camera") or 1
    quals = []
    if s.get("single_camera_required"):
        cams = "one camera"
        if s.get("square_images"):
            quals.append("square")
        if frames > 1:
            quals.append(f"{frames} frames")
    else:
        cams = "several cameras"
        if s.get("camera_mapping_required"):
            quals.append("mapping required")
        if frames > 1:
            quals.append(f"{frames} frames each")
    if quals:
        cams += f" ({', '.join(quals)})"
    parts = [cams, "state"]
    if s.get("requires_reward"):
        parts.append("reward labels")
    parts.append("language" if s.get("requires_language") else "no language")
    return ", ".join(parts)


def action_cell(s: dict) -> str:
    dims = f"up to {s['max_action_dim']} dims" if s.get("max_action_dim") else "dataset width"
    return f"{dims}, chunk {(s.get('defaults') or {}).get('chunk', '?')}"


def gpu_cell(s: dict, instance: str) -> str:
    return f"{(s.get('hardware') or {}).get('min_gpu_memory_gb', '?')} GB · `{instance}`{_untested(instance)}"


def switch_text(s: dict) -> str:
    lic = s.get("licence") or {}
    if lic.get("class") == "clean":
        return ""
    sw = "`accept_licence`"
    if lic.get("requires_approval") or lic.get("class") == "research_only":
        sw += " + `approval_ref`"
    return sw


def licence_cell(s: dict) -> str:
    """`[terms document](url)` followed by ` · <switch>` for every entry that needs one."""
    lic = s.get("licence") or {}
    label = LICENCE_LABEL[s["name"]] if lic.get("class") != "clean" else (lic.get("spdx") or lic.get("class"))
    cell = f"[{label}]({lic['url']})" if lic.get("url") else label
    sw = switch_text(s)
    return f"{cell} · {sw}" if sw else cell


def capability_matrix(specs: list, cfg: dict) -> str:
    lines = ["| policy | what it is | inputs | action space | GPU and default instance | licence of the base weights | status |",
             "|---|---|---|---|---|---|---|"]
    for s in offered(specs):
        lines.append(f"| `{s['name']}` | {WHAT_IT_IS[s['name']]} | {inputs_cell(s)} | {action_cell(s)} | {gpu_cell(s, default_instance(s, cfg))} | "
                     f"{licence_cell(s)} | {s.get('status')} |")
    return "\n".join(lines)


def scenario_table() -> str:
    lines = ["| situation | recommended | alternatives | why |", "|---|---|---|---|"]
    for r in SCENARIOS:
        alts = ", ".join(f"`{a}`" for a in r["alternatives"]) or "none"
        lines.append(f"| {r['situation']} | `{r['recommended']}` | {alts} | {r['why']} |")
    return "\n".join(lines)


def fallback_order(specs: list) -> dict:
    """{gpu memory GB -> [instance types]}: the union of the offered entries' hardware.instance_types, in the order the
    registry lists them (first appearance wins), grouped by the GPU memory of the type."""
    seen, by_class = [], {}
    for s in offered(specs):
        for it in (s.get("hardware") or {}).get("instance_types") or []:
            if it not in seen:
                seen.append(it)
    for it in seen:
        sp = instance_table.spec(it)
        by_class.setdefault(sp.gpu_mem_gb if sp else 0, []).append(it)
    return dict(sorted(by_class.items()))


def fallback_tables(specs: list) -> str:
    out = []
    for mem, types in fallback_order(specs).items():
        out.append(f"**{mem} GB GPUs**" if mem else "**GPU memory unknown**")
        out.append("")
        out.append("| # | instance type | GPU | approx. USD/h | quota name (on-demand) | notes |")
        out.append("|---|---|---|---|---|---|")
        for i, it in enumerate(types, 1):
            sp = instance_table.spec(it)
            price = plan.price_per_hour(it, False)
            notes = []
            if sp and not sp.verified:
                notes.append("(untested)")
            if sp and sp.gpus > 1:
                notes.append(f"{sp.gpus} GPUs: needs `--allow-multigpu`, {sp.gpus - 1} GPUs bill idle")
            gpu = f"{sp.gpus} x {sp.gpu.replace('_', ' ')}" if sp else "?"
            out.append(f"| {i} | `{it}` | {gpu} | {f'{price:.1f}' if price else 'n/a'} | `{instance_table.quota_names(it, False)[0]}` | {' '.join(notes)} |")
        out.append("")
    return "\n".join(out).rstrip("\n")


def _estimate(s: dict, instance: str) -> str:
    rate = instance_table.step_rate(s, instance)
    row = plan.policy_row(s["name"], {}, s, {"smoke": {"enabled": True}}, "", instance, False, rate, instance_table.spec(instance))
    if row["est_h"] is None:
        return "-"
    usd = f" / ~${row['est_usd']:.0f}" if row["est_usd"] is not None else ""
    return f"~{row['est_h']:.1f} h{usd}"


def render(specs: list, cfg: dict) -> str:
    lines = [HEADER_TOP, disclaimer(specs), HEADER,
             "| policy | tier | status | deliverable | base checkpoint | weights licence | commercial use | inputs | action space | VRAM | GPU and default instance | default budget (steps x batch) | est. time / cost | notes |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in specs:
        lic = s.get("licence", {}); hw = s.get("hardware", {}); d = s.get("defaults", {})
        base = s.get("base_checkpoint") or "none (from scratch)"
        lic_txt = f"[{lic.get('class')}]({lic['url']})" if lic.get("url") else str(lic.get("class"))
        if lic.get("spdx"):
            lic_txt += f" · {lic['spdx']}"
        sw = switch_text(s) if not s.get("out_of_scope") and lic.get("class") != "blocked" else ""
        if sw:
            lic_txt += f" · {sw}"
        vram = hw.get("vram_gb")
        gpu = f"{vram} GB" if vram else (f"≥ {hw.get('min_gpu_memory_gb')} GB" if hw.get("min_gpu_memory_gb") else "-")
        notes = s.get("stub_reason") or ""
        if s.get("smoke_candidate") and s.get("status") == "stub":
            notes = (notes + " " if notes else "") + "(smoke planned)"
        inst = default_instance(s, cfg)
        budget = f"{d.get('steps', '?')} x {d.get('batch_size', '?')}"
        lines.append(f"| `{s['name']}` | {s.get('tier')} | {s.get('status')} | {'yes' if s.get('deliverable') else 'no'} | {base} | {lic_txt} | "
                     f"{lic.get('commercial_use', '?')} | {inputs_cell(s)} | {action_cell(s)} | {gpu} | {gpu_cell(s, inst)} | {budget} | "
                     f"{_estimate(s, inst)} | {notes.strip()} |")
    lines.append("")
    lines.append("## Which policy for which situation\n")
    lines.append("The same table as in the README. Every policy named is offered; the rows are data in `pipeline/policies_doc.py` and\n"
                 "tests check them against the registry (licence class, GPU memory, action width).\n")
    lines.append(scenario_table())
    lines.append("")
    lines.append("## Per-policy notes\n")
    for s in specs:
        lic = s.get("licence", {})
        lines.append(f"### `{s['name']}`: {s.get('display_name', '')}\n")
        lines.append(f"- lerobot type `{s.get('lerobot_type')}`, precision `{s.get('precision')}`, tier {TIER_LABEL.get(s.get('tier'), s.get('tier'))}")
        if s.get("hub_artifacts"):
            arts = ", ".join(f"`{a['repo']}`{' (gated)' if a.get('gated') else ''}" for a in s["hub_artifacts"])
            lines.append(f"- downloaded into your bucket by the staging job: {arts}")
        if lic.get("notice"):
            lines.append(f"- licence: {str(lic['notice']).strip()} (class verified {lic.get('verified_on', '?')})")
        hw = s.get("hardware", {})
        if hw.get("instance_types"):
            lines.append(f"- instances: {', '.join(hw['instance_types'])}")
        ev = s.get("eval", {})
        if ev:
            lines.append(f"- held-out evaluation: `eval_openloop.py --policy {ev.get('policy_arg')}`, {ev.get('n_samples', 1)} action draw(s) averaged per frame, "
                         f"{ev.get('history_frames', 1)} observation frame(s) in the input window")
        sv = s.get("serving", {})
        if sv.get("notes"):
            lines.append(f"- serving: {sv['notes']}")
        lines.append("")
    return "\n".join(lines)


# -- README blocks -----------------------------------------------------------------------------------------------------

README_BLOCKS = {
    "licence-disclaimer": lambda specs, cfg: disclaimer(specs),
    "capability-matrix": capability_matrix,
    "scenario-fit": lambda specs, cfg: scenario_table(),
    "instance-fallback": lambda specs, cfg: fallback_tables(specs),
}
_BEGIN = "<!-- BEGIN GENERATED {name}: edit pipeline/policies_doc.py or trainer/policies/*.yaml, then run python3 -m pipeline.policies_doc -->"
_END = "<!-- END GENERATED {name} -->"


def render_readme(text: str, specs: list, cfg: dict) -> str:
    """The README with every marked block replaced by its rendering. Raises when a marker pair is missing."""
    for name, fn in README_BLOCKS.items():
        begin, end = _BEGIN.format(name=name), _END.format(name=name)
        pat = re.compile(re.escape(begin) + r"\n.*?\n" + re.escape(end), re.S)
        if not pat.search(text):
            raise ValueError(f"README.md has no '{begin}' ... '{end}' block")
        text = pat.sub(lambda m, body=fn(specs, cfg): f"{begin}\n{body}\n{end}", text, count=1)
    return text


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    specs, cfg = load_all(), load_project_config()
    text = render(specs, cfg)
    readme_now = README.read_text()
    try:
        readme_new = render_readme(readme_now, specs, cfg)
    except ValueError as e:   # a README marker pair is missing: report it like a stale file, not as a traceback
        print(e, file=sys.stderr)
        return 1
    if a.check:
        stale = [p.relative_to(REPO) for p, cur, new in ((OUT, OUT.read_text() if OUT.exists() else None, text), (README, readme_now, readme_new)) if cur != new]
        if stale:
            print(f"{', '.join(map(str, stale))} stale: run python3 -m pipeline.policies_doc", file=sys.stderr)
            return 1
        print("docs/POLICIES.md up to date; README blocks up to date")
        return 0
    OUT.write_text(text)
    README.write_text(readme_new)
    print(f"wrote {OUT.relative_to(REPO)} ({len(specs)} policies) and {len(README_BLOCKS)} README blocks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
