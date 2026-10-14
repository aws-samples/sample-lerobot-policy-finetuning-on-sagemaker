#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Pre-publish scrub gate: fail if any tracked file (or, with --history, any commit of any ref) carries an AWS
account id, a credential, a private address, an internal hostname, an account-specific resource id, a string from
the optional deny-list, or if the private working folder is tracked.

    python3 scripts/scrub_check.py              # scan `git ls-files` (CI runs this)
    python3 scripts/scrub_check.py PATH ...     # scan the given files (tests use this)
    python3 scripts/scrub_check.py --history    # scan every commit message and diff of every ref, plus author identities

Exit 0 and print "scrub CLEAN: ..." when nothing is found, exit 1 with one `LEAK` line per finding otherwise,
exit 2 on a bad argument or when git is unusable. Findings never print the matched value in full: only its first characters.

Patterns (PATTERNS below):
  account-id       12-digit runs that are NOT part of a decimal number or a hex digest
                   matches   123456789012.dkr.ecr.eu-central-1.amazonaws.com   arn:aws:iam::123456789012:role/x
                   ignores   0.000107444750 (float in an eval fixture)         sha256:5e0ed2da...c1a2b3 (hex)
  aws-access-key   AKIA/ASIA access key ids;   hf-token  Hugging Face `hf_` tokens;   private-key  PEM headers
  private-ip       RFC 1918 addresses (10/8, 172.16/12, 192.168/16), also in URLs and assignments; version pins
                   such as `==10.3.10.19` are skipped
  internal-host    Amazon-internal hostnames (the internal wiki, code browser, policy and *.dev-style corporate domains)
  amazon-email     addresses at amazon.com (the code-of-conduct mailbox is allow-listed)
  aws-resource-id  17-hex-digit EC2 ids (i-, cr-, ami-, sg-, vpc-, subnet-, vol-, snap-, eni-)
  cdk-bucket-name  CDK-generated bucket names (`...bucket5cb76180-<12 chars>`), which identify one deployment
Allow-listed strings live in scripts/scrub_allowlist.txt (one per line, `value  # reason`): account ids that may appear
(the public AWS Deep Learning Containers registry, the 123456789012 documentation placeholder) and literals such as the
AWS documentation example access key used in test fixtures.
The deny-list (.scrub_denylist, git-ignored, one literal string per line) holds names that must never appear in the
repo: customer names, bucket names, role names, job names, user aliases. It is optional so CI can run without it.
Deny-list matching is a case-insensitive substring test (an entry `Acme` also catches `ACME` and `acme-bucket`);
findings print only the first three characters of the entry.
Only files with a suffix in TEXT_SUFFIXES are read (source, docs, configs, JS, lock files, extension-less scripts);
anything else is skipped as binary; a text file with invalid UTF-8 is still scanned with the bad bytes replaced.

--history scans merge commits against each parent, applies the deny-list and the hostname/key patterns to author and
committer identities as well (a work e-mail address as the author is accepted), reports identities whose e-mail is not a real address (`a@b`, `x@example.com`) and lists the refs that would be pushed with `--all`/`--tags`; run it on the fresh clone that will be pushed publicly.
"""
from __future__ import annotations

import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
ACCOUNT_RE = re.compile(r"(?<![0-9A-Fa-f.])[0-9]{12}(?![0-9A-Fa-f]|\.[0-9])")
PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("account id", ACCOUNT_RE),
    ("aws access key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("hf token", re.compile(r"\bhf_[A-Za-z0-9]{30,}\b")),
    ("private key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("private ip", re.compile(r"(?<![\w.])(?<!==)(?<!>=)(?<!<=)(?<!~=)(?<!!=)(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}"
                              r"|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}|192\.168\.\d{1,3}\.\d{1,3})(?!\w|\.\d)")),
    ("internal host", re.compile(r"\b(?:w|code|wiki|sim|issues|t\.corp|broadcast|quip|drive|sage|policy)\.amazon\.com\b"
                                 r"|\.(?:ant|aka|desktop|corp)\.amazon\.com\b|\bdev-dsk-[\w-]+|\.amazon\.dev\b|\ba2z\.com\b"
                                 r"|\baws\.dev\b|\bmidway-auth\b|\bisengard\b", re.IGNORECASE)),
    ("amazon email", re.compile(r"\b[A-Za-z0-9._%+-]+@amazon\.com\b", re.IGNORECASE)),
    ("aws resource id", re.compile(r"\b(?:i|cr|ami|sg|vpc|subnet|vol|snap|eni)-[0-9a-f]{17}\b")),
    ("cdk bucket name", re.compile(r"\b[a-z0-9-]*bucket[0-9a-f]{8}-[a-z0-9]{12,13}\b")),
]
IDENTITY_RE = re.compile(r"^[^@\s]+@(?!example\.)[^@\s.]+(?:\.[^@\s.]+)*\.[A-Za-z]{2,}$")
NEVER_TRACKED = ("local_folder/",)
TEXT_SUFFIXES = {".py", ".sh", ".md", ".json", ".yaml", ".yml", ".txt", ".toml", ".cfg", ".ini", ".csv", ".env", ".js",
                 ".lock", ""}


def _git(*args: str, cwd: pathlib.Path | None = None) -> str:
    return subprocess.run(["git", *args], cwd=cwd or ROOT, check=True, capture_output=True).stdout.decode("utf-8", "replace")


def tracked_files() -> list[str]:
    out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, check=True, capture_output=True).stdout
    return [p for p in out.decode().split("\0") if p]


def load_list(path: pathlib.Path) -> list[str]:
    if not path.exists():
        return []
    items = []
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            items.append(line)
    return items


def _deny_pairs(deny: list[str]) -> list[tuple[str, str]]:
    seen: dict[str, str] = {}
    for word in deny:
        seen.setdefault(word.lower(), word)  # one match per distinct entry; first spelling wins for the report
    return [(word, low) for low, word in seen.items()]


def _mask(value: str, keep: int = 4) -> str:
    return value[:keep] + "…" if len(value) > keep else value


def scan_line_raw(line: str, allow: set[str], deny_pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Every finding in one line of text as (kind, matched value); deny-list hits carry the deny entry itself."""
    found = []
    for kind, rx in PATTERNS:
        for m in rx.finditer(line):
            if m.group(0) not in allow:
                found.append((kind, m.group(0)))
    line_lower = line.lower()
    for word, word_lower in deny_pairs:
        if word_lower in line_lower:
            found.append(("deny-listed string", word))
    return found


def _label(kind: str, value: str) -> str:
    return f"deny-listed string ({value[:3]}…)" if kind == "deny-listed string" else f"{kind} {_mask(value)}"


def scan_line(line: str, allow: set[str], deny_pairs: list[tuple[str, str]]) -> list[str]:
    """Every finding in one line of text, as `<kind> <masked value>` strings (never the full value)."""
    return [_label(k, v) for k, v in scan_line_raw(line, allow, deny_pairs)]


def scan(paths: list[str], allow: set[str], deny: list[str]) -> list[str]:
    problems = []
    deny_pairs = _deny_pairs(deny)
    for rel in paths:
        if any(rel.startswith(p) for p in NEVER_TRACKED):
            problems.append(f"{rel}: private working folder must never be tracked")
            continue
        p = ROOT / rel
        if not p.is_file() or p.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue  # unreadable: nothing to grep
        for n, line in enumerate(text.splitlines(), 1):
            for hit in scan_line(line, allow, deny_pairs):
                problems.append(f"{rel}:{n}: {hit}")
    return problems


def scan_history(repo: pathlib.Path, allow: set[str], deny: list[str]) -> tuple[list[str], dict[str, int]]:
    """Scan every commit of every ref: subject + body + full diff, then author/committer identities.
    Returns (problems, stats). Each distinct (kind, value) is reported once with the commit that introduced it
    (newest first in `git log` order, so the reported commit is the last one still carrying it)."""
    deny_pairs = _deny_pairs(deny)
    problems: list[str] = []
    # -m: merge commits show their diff against each parent, so a leak that exists only in a conflict resolution is seen
    log = subprocess.run(["git", "log", "--all", "-p", "-m", "--no-color", "--format=%x00commit %H%n%s%n%n%b"],
                         cwd=repo, check=True, capture_output=True).stdout.decode("utf-8", "replace")
    seen: set[tuple[str, str]] = set()
    sha = "?"
    for line in log.split("\n"):
        if line.startswith("\0commit "):
            sha = line[8:15]
            continue
        for hit in scan_line_raw(line, allow, deny_pairs):
            if hit not in seen:
                seen.add(hit)
                problems.append(f"history {sha}: {_label(*hit)}")
    idents = _git("log", "--all", "--format=%an <%ae>%n%cn <%ce>", cwd=repo).splitlines()
    for ident in sorted(set(i for i in idents if i.strip())):
        email = ident.rsplit("<", 1)[-1].rstrip(">")
        if not IDENTITY_RE.match(email):
            problems.append(f"history identity: placeholder author or committer `{_mask(ident, 6)}` (set user.name/user.email before the squash)")
        for hit in scan_line(ident, allow, deny_pairs):  # a deny-listed alias or an internal hostname in the identity itself
            if hit.startswith("amazon email"):   # an employee's work address as the author is the norm on aws-samples, not a leak
                continue
            problems.append(f"history identity `{_mask(ident, 6)}`: {hit}")
    refs = [r for r in _git("for-each-ref", "--format=%(refname)", cwd=repo).splitlines() if r]
    stats = {"commits": int(_git("rev-list", "--all", "--count", cwd=repo).strip() or 0),
             "identities": len(set(idents)), "tags": sum(r.startswith("refs/tags/") for r in refs), "refs": len(refs)}
    return problems, stats


def main(argv: list[str]) -> int:
    allow = set(load_list(ROOT / "scripts" / "scrub_allowlist.txt"))
    deny = load_list(ROOT / ".scrub_denylist")
    note = "" if deny else " (no .scrub_denylist present: names not checked)"
    if argv[:1] == ["--history"]:
        try:
            problems, stats = scan_history(ROOT, allow, deny)
        except (subprocess.CalledProcessError, OSError) as e:
            print(f"scrub ERROR: git history not readable ({e})")
            return 2
        for p in problems:
            print("LEAK", p)
        summary = (f"{stats['commits']} commits, {stats['identities']} identities, {stats['refs']} refs "
                   f"({stats['tags']} tags; push only the branch you mean to publish)")
        if problems:
            print(f"scrub history FAILED: {len(problems)} finding(s) in {summary}")
            return 1
        print(f"scrub history CLEAN: {summary}{note}")
        return 0
    bad = [a for a in argv if a.startswith("-")]
    if bad:
        print(f"scrub ERROR: unknown option {bad[0]} (usage: scrub_check.py [--history | PATH ...])")
        return 2
    missing = [a for a in argv if not (ROOT / a).exists() and not pathlib.Path(a).exists()]
    if missing:
        print(f"scrub ERROR: no such file: {missing[0]}")
        return 2
    try:
        paths = argv or tracked_files()
    except (subprocess.CalledProcessError, OSError) as e:
        print(f"scrub ERROR: git ls-files failed ({e})")
        return 2
    problems = scan(paths, allow, deny)
    for p in problems:
        print("LEAK", p)
    if problems:
        print(f"scrub FAILED: {len(problems)} finding(s) in {len(paths)} files")
        return 1
    print(f"scrub CLEAN: {len(paths)} files{note}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
