#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Known-vulnerability audit of a pinned requirements file against the OSV database (https://osv.dev), stdlib only.

    python3 scripts/audit_image_lock.py                                   # trainer/sagemaker/requirements.lock
    python3 scripts/audit_image_lock.py requirements.txt --resolved       # what pip resolved in THIS interpreter
    python3 scripts/audit_image_lock.py --json out.json --fail-on high    # exit 1 on HIGH/CRITICAL advisories

Why not pip-audit: the training image's lock is for the image's Python (3.12); pip-audit resolves it in the laptop's
interpreter and fails when a pin has no wheel for that version. This script only reads `name==version` pins (or, with
--resolved, the versions installed in the running interpreter) and asks OSV for advisories, so it works from any laptop
with network access. Exit 0 when no advisory reaches --fail-on (default: never fail, report only).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import urllib.request

OSV_BATCH = "https://api.osv.dev/v1/querybatch"
OSV_VULN = "https://api.osv.dev/v1/vulns/{}"
LEVELS = {"critical": 4, "high": 3, "moderate": 2, "medium": 2, "low": 1, "": 0}
PIN = re.compile(r"^([A-Za-z0-9_.\-]+)==([^\s;]+)")


def pins_from_file(path: pathlib.Path) -> list:
    out = []
    for line in path.read_text().splitlines():
        m = PIN.match(line.split("#", 1)[0].strip())
        if m:
            out.append((m.group(1), m.group(2).split("+")[0]))   # torch==2.8.0+cu129 -> 2.8.0
    return out


def pins_resolved(path: pathlib.Path) -> list:
    """The versions the running interpreter has installed for the names in `path` (lower-bounded requirements)."""
    import importlib.metadata as md
    names = [re.split(r"[<>=!~\[; ]", l.split("#", 1)[0].strip())[0] for l in path.read_text().splitlines()
             if l.strip() and not l.startswith(("#", "-r"))]
    out = []
    for n in names:
        try:
            out.append((n, md.version(n)))
        except md.PackageNotFoundError:
            print(f"  (not installed here: {n})", file=sys.stderr)
    return out


def _get(url: str, data: bytes = None) -> dict:
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=60) as r:   # nosec B310 # nosemgrep -- dynamic-urllib-use: url is one of the two fixed https OSV endpoints above
        return json.load(r)


def query(pins: list) -> dict:
    """{'name==version': [advisory dicts]} for every pin OSV knows an advisory for."""
    found = {}
    for i in range(0, len(pins), 100):
        chunk = pins[i:i + 100]
        body = json.dumps({"queries": [{"package": {"name": n, "ecosystem": "PyPI"}, "version": v} for n, v in chunk]}).encode()
        for (n, v), res in zip(chunk, _get(OSV_BATCH, body)["results"]):
            ids = [x["id"] for x in res.get("vulns", [])]
            if ids:
                found[f"{n}=={v}"] = ids
    out = {}
    for pkg, ids in sorted(found.items()):
        rows = []
        for vid in ids:
            d = _get(OSV_VULN.format(vid))
            sev = ((d.get("database_specific") or {}).get("severity") or "").lower()
            fixed = sorted({ev["fixed"] for a in d.get("affected", []) if (a.get("package") or {}).get("ecosystem") == "PyPI"
                            for rng in a.get("ranges", []) for ev in rng.get("events", []) if "fixed" in ev})
            rows.append({"id": vid, "severity": sev, "fixed": fixed, "aliases": d.get("aliases", []), "summary": d.get("summary") or ""})
        out[pkg] = rows
    return out


def main(argv: list) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("requirements", nargs="?", default="trainer/sagemaker/requirements.lock")
    ap.add_argument("--resolved", action="store_true", help="audit the versions installed in this interpreter instead of == pins")
    ap.add_argument("--json", help="write the full result here")
    ap.add_argument("--fail-on", choices=["low", "moderate", "high", "critical"], help="exit 1 when an advisory of this severity or higher exists")
    a = ap.parse_args(argv)
    path = pathlib.Path(a.requirements)
    pins = pins_resolved(path) if a.resolved else pins_from_file(path)
    result = query(pins)
    worst = 0
    for pkg, rows in result.items():
        print(pkg)
        for r in rows:
            worst = max(worst, LEVELS.get(r["severity"], 0))
            print(f"    {r['id']:<22} {r['severity'] or '-':<9} fixed: {', '.join(r['fixed']) or '-':<22} {r['summary'][:80]}")
    print(f"{len(pins)} pins checked, {len(result)} with advisories")
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps({"pins": len(pins), "advisories": result}, indent=1))
    return 1 if a.fail_on and worst >= LEVELS[a.fail_on] else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
