# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""The detect-secrets baseline must cover the current CDK asset ids: every template re-export changes them, and a stale
baseline fails the CI secret scan (it happened once when a re-export merged from a parallel branch)."""
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def test_cdk_asset_ids_are_audited_in_the_secrets_baseline():
    baseline = json.loads((REPO / ".secrets.baseline").read_text())["results"]
    text = (REPO / "cdk" / "template" / "assets.json").read_text()
    import re
    ids = set(re.findall(r'"[0-9a-f]{64}"', text))
    assert ids, "no 64-hex asset id in cdk/template/assets.json"
    audited = {s["hashed_secret"] for s in baseline.get("cdk/template/assets.json", []) if s.get("is_secret") is False}
    # detect-secrets stores SHA-1 of each flagged string; this reproduces its lookup key, not a security hash
    missing = [i for i in ids if hashlib.sha1(i.strip('"').encode(), usedforsecurity=False).hexdigest() not in audited]  # nosemgrep: python.lang.security.insecure-hash-algorithms.insecure-hash-algorithm-sha1
    assert not missing, f"asset ids not audited in .secrets.baseline (run detect-secrets scan --baseline and audit): {missing}"
    unaudited = [(f, s["line_number"], s["type"]) for f, ss in baseline.items() for s in ss if s.get("is_secret") is not False]
    assert not unaudited, unaudited
