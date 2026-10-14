# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Every shell script in the tree parses (`bash -n`, the CI step) and the maintainer script answers --help without side effects.

Scripts are found by globbing, not `git ls-files`, so the test also covers a script that is not committed yet and a
checkout without git; virtualenvs and CDK output directories are excluded.
"""
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
EXCLUDE = (".venv", "cdk.out", "node_modules", ".git")
SCRIPTS = sorted(str(p.relative_to(REPO)) for p in REPO.rglob("*.sh") if not any(part.startswith(EXCLUDE) for part in p.parts))


@pytest.mark.parametrize("script", SCRIPTS)
def test_bash_syntax(script):
    r = subprocess.run(["bash", "-n", script], cwd=REPO, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_the_maintainer_scripts_are_found():
    assert "scripts/verify_fresh_clone.sh" in SCRIPTS and "scripts/scrub_check.sh" in SCRIPTS


def test_verify_fresh_clone_help_exits_zero():
    r = subprocess.run(["bash", "scripts/verify_fresh_clone.sh", "--help"], cwd=REPO, capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr
    assert r.stdout.startswith("usage: scripts/verify_fresh_clone.sh") and "--denylist" in r.stdout and "--skip-cdk" in r.stdout


def test_verify_fresh_clone_rejects_unknown_arguments():
    r = subprocess.run(["bash", "scripts/verify_fresh_clone.sh", "--bogus"], cwd=REPO, capture_output=True, text=True, timeout=30)
    assert r.returncode == 2 and "unknown argument" in r.stderr


def test_verify_fresh_clone_is_documented_in_contributing():
    assert "scripts/verify_fresh_clone.sh" in (REPO / "CONTRIBUTING.md").read_text()


A = "1" * 4 + "2" * 4 + "3" * 4  # built at runtime so no 12-digit literal sits in a tracked file (scrub gate)
B = "4" * 4 + "5" * 4 + "6" * 4
MASK_PROBES = {  # what the report masker must hide (True) or leave alone (False), mirroring scrub_check.ACCOUNT_RE
    f"LEAK pipeline/x.py:1: account id {A}": True,
    f"arn:aws:iam::{A}:role/x": True,
    f"{A}.dkr.ecr.eu-central-1.amazonaws.com/img": True,
    f"s3://{A}-lerobot-bucket/x": True,
    f"s3://lerobot-{A}/ds": True,
    f"https://{A}.signin.aws.amazon.com/console": True,
    f"'Bucket': 'lerobot-{A}-us-east-1'": True,
    f"two {A} {B} ids": True,
    f"sha256:5e0ed2da{A}c1a2b3": False,
    "0.000107444750": False,
    "107444750123.5": False,
    "1234567890123 thirteen digits": False,
}


def test_verify_fresh_clone_report_mask_hides_every_account_id():
    """The report is meant to be pasted into an issue: every 12-digit id the scrub gate would flag must be masked."""
    script = (REPO / "scripts/verify_fresh_clone.sh").read_text()
    start = script.index("\nmask() {")
    mask_fn = script[start:script.index("\n}\n", start) + 3]
    probe = "\n".join(MASK_PROBES) + "\n"
    r = subprocess.run(["bash", "-c", mask_fn + "\nmask"], input=probe, capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr
    for line, (probe_line, must_mask) in zip(r.stdout.splitlines(), MASK_PROBES.items()):
        if must_mask:
            assert A not in line and B not in line, line
            assert "************" in line, line
        else:
            assert line == probe_line, line


def test_destroy_help_prints_the_whole_comment_block():
    r = subprocess.run(["bash", "scripts/destroy.sh", "--help"], cwd=REPO, capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr
    assert "--delete-secret" in r.stdout and len(r.stdout.splitlines()) >= 8, r.stdout
