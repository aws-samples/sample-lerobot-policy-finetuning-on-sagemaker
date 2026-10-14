# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Every `file.py:NNN` citation in the docs must land on real, non-blank code.

Borrowed from the DreamZero sample. Two additions: a citation whose file does not exist anywhere in the repo fails
(a renamed script must not let citations of its old name pass silently), unless the basename is a
known upstream file we deliberately cite without vendoring (lerobot, torchvision); and zero citations skips instead of
failing, so the test is valid from the first commit.
"""
import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
CITATION = re.compile(r"([A-Za-z0-9_./-]+\.(?:py|sh|json|yaml|yml)):(\d+)(?:-(\d+))?")
SEARCH_DIRS = ("", "trainer", "trainer/sagemaker", "trainer/policies", "pipeline", "cdk", "cdk/lerobot_finetune",
               "scripts", "tests", "inference")
# Upstream files the docs may cite by name without shipping them (lerobot 0.6.1, torchvision, DLC).
UPSTREAM_OK = {"factory.py", "lerobot_train.py", "lerobot_dataset.py", "train_utils.py", "modeling_act.py",
               "configuration_act.py", "pretrained.py", "train.py", "parser.py", "compute_stats.py", "video_utils.py",
               "modeling_pi0.py", "modeling_wall_x.py", "modeling_tdmpc.py", "processor_groot.py", "logging_utils.py",
               "import_utils.py", "accelerator.py", "optim/factory.py"}


def _tracked_markdown():
    out = subprocess.run(["git", "-C", str(REPO), "ls-files", "*.md", "**/*.md"], capture_output=True, text=True)
    if out.returncode != 0:
        return []
    return sorted({REPO / p for p in out.stdout.split()})


def _resolve(cited):
    for d in SEARCH_DIRS:
        p = REPO / d / Path(cited).name if d else REPO / cited
        if p.is_file():
            return p
    return None


def test_doc_line_citations_resolve():
    md_files = _tracked_markdown()
    if not md_files:
        pytest.skip("not a git checkout")
    checked, bad, missing = 0, [], []
    for md in md_files:
        text = md.read_text(errors="replace")
        for m in CITATION.finditer(text):
            cited, start = m.group(1), int(m.group(2))
            target = _resolve(cited)
            rel = md.relative_to(REPO)
            if target is None:
                if Path(cited).name not in UPSTREAM_OK and not cited.startswith("lerobot/"):
                    missing.append(f"{rel} cites {m.group(0)} but no such file exists in the repo")
                continue
            lines = target.read_text(errors="replace").split("\n")
            checked += 1
            if start > len(lines):
                bad.append(f"{rel} cites {m.group(0)} but {target.relative_to(REPO)} has {len(lines)} lines")
            elif not lines[start - 1].strip():
                bad.append(f"{rel} cites {m.group(0)} which is a BLANK line in {target.relative_to(REPO)}")
    assert not missing, "\n  ".join(["citations to files that do not exist:"] + missing)
    assert not bad, "\n  ".join([f"{len(bad)} stale line citation(s):"] + bad)
    if not checked:
        pytest.skip("no file:line citations in the docs yet")
