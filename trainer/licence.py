#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Licence gate for registry policies, and the notice written next to every published checkpoint.

    licence.py act                       # exit 0: clean
    licence.py xvla                      # exit 3: class 'terms' needs --accept (accept_licence: true in project_config.json)
    licence.py xvla --accept             # exit 0 (the acceptance is recorded in the job's MANIFEST/SUMMARY)
    licence.py wall_x --accept           # exit 3: class 'blocked' cannot be accepted

Acknowledging a licence grants nothing the publisher has not granted; this gate only prevents accidental use of a
restricted base and leaves an audit trail. Classes: clean | terms | research_only | non_commercial | none | unknown | blocked.
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

from render_flags import POLICIES_DIR, RenderError, load_policy

ACCEPTABLE = ("terms", "research_only", "non_commercial", "none", "unknown")


def gate(spec: dict, *, accept: bool = False, approval_ref: str = None) -> tuple:
    """(ok, message). Blocked never passes; non-clean classes need accept (and an approval_ref when the entry demands one)."""
    lic = spec["licence"]
    cls = lic.get("class", "unknown")
    name = spec["name"]
    if cls == "clean":
        return True, f"{name}: licence class clean ({lic.get('spdx', '?')})"
    if cls == "blocked":
        return False, f"{name}: licence class blocked -- {lic.get('notice', 'not offered')}; this policy cannot be run from this repo"
    if cls not in ACCEPTABLE:
        return False, f"{name}: unknown licence class {cls!r} in the registry"
    if not accept:
        return False, (f"{name}: licence class {cls} ({lic.get('spdx') or 'no SPDX id'}); read {lic.get('url', '?')} and set "
                       f"policies.{name}.accept_licence: true in project_config.json to proceed. Commercial use: {lic.get('commercial_use', 'unknown')}.")
    if lic.get("requires_approval") and not approval_ref:
        return False, f"{name}: this entry requires an approval reference (policies.{name}.approval_ref) in addition to accept_licence"
    return True, f"{name}: licence class {cls} accepted" + (f" (approval {approval_ref})" if approval_ref else "")


def notice_text(spec: dict, *, dataset_name: str = "", image_digest: str = "", accepted_by: str = "",
                run_status: str = "", licence_status: str = "") -> str:
    """The LICENCE_NOTICE.md text. The registry fields (tier, status, deliverable, class) are those baked into the
    training image; the run's own provenance (dataset fingerprint, image digest, run status, gate outcome) is passed in
    by publish.py so the notice names the exact inputs the checkpoint came from."""
    lic = spec["licence"]
    lines = [
        f"# Licence notice for the {spec.get('display_name', spec['name'])} checkpoint",
        "",
        f"Policy type: {spec['lerobot_type']} (lerobot 0.6.1)  |  tier {spec.get('tier')}  |  registry status {spec.get('status')} "
        f"(as built into the training image)  |  deliverable: {'yes' if spec.get('deliverable') else 'no'}",
        f"Weights licence class: {lic.get('class')}  |  SPDX: {lic.get('spdx') or 'none declared'}  |  commercial use: {lic.get('commercial_use', 'unknown')}",
        f"Instrument: {lic.get('url', 'n/a')}  (class verified on {lic.get('verified_on', '?')})",
        f"Base checkpoint: {spec.get('base_checkpoint') or 'none (trained from scratch)'}",
    ]
    if run_status or licence_status:
        lines.append(f"Run: status {run_status or 'n/a'}  |  licence gate: {licence_status or 'n/a'}")
    lines += [
        "",
        str(lic.get("notice", "")).strip(),
        "",
        "This repository contains no model weights. Any base checkpoint above was downloaded by the training job into your own AWS",
        "account directly from its publisher (the exact commit is recorded in the staging manifest), and remains under that publisher's licence; fine-tuned outputs",
        "inherit the base's restrictions. The MIT-0 licence of the repository covers only its code.",
        "",
        f"Dataset fingerprint: {dataset_name or 'n/a'}  |  Image: {image_digest or 'n/a'}  |  Generated: {dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M UTC}",
    ]
    if accepted_by:
        lines.append(f"Licence acknowledged via project_config.json ({accepted_by}).")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("policy")
    ap.add_argument("--accept", action="store_true")
    ap.add_argument("--approval-ref")
    ap.add_argument("--policies-dir", default=str(POLICIES_DIR))
    ap.add_argument("--notice", action="store_true", help="print the LICENCE_NOTICE.md text instead of gating")
    ap.add_argument("--dataset", default="", help="--notice: dataset fingerprint of the run")
    ap.add_argument("--image", default="", help="--notice: training image digest of the run")
    ap.add_argument("--run-status", default="", help="--notice: completed | interrupted | failed")
    ap.add_argument("--licence-status", default="", help="--notice: the gate's outcome for this run (accepted | refused)")
    ap.add_argument("--accepted-by", default="", help="--notice: how the licence was acknowledged (project_config.json key, approval ref)")
    a = ap.parse_args(argv)
    try:
        spec = load_policy(a.policy, Path(a.policies_dir))
    except RenderError as e:
        print(f"licence: {e}", file=sys.stderr)
        return 2
    if a.notice:
        print(notice_text(spec, dataset_name=a.dataset, image_digest=a.image, accepted_by=a.accepted_by,
                          run_status=a.run_status, licence_status=a.licence_status), end="")
        return 0
    ok, msg = gate(spec, accept=a.accept, approval_ref=a.approval_ref)
    print(msg, file=sys.stderr)
    return 0 if ok else 3


if __name__ == "__main__":
    sys.exit(main())
