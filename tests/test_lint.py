# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Undefined names and similar pyflakes errors in every Python file of the repo (a NameError in a rarely exercised branch
of pipeline/run.py once reached a paid run before any test did)."""
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FILES = sorted(str(p) for pat in ("pipeline/*.py", "trainer/*.py", "cdk/app.py", "cdk/lerobot_finetune/*.py", "scripts/*.py",
                                  "inference/*.py", "tests/*.py", "tests/fixtures/*.py") for p in REPO.glob(pat))
IGNORE = ("imported but unused", "unable to detect undefined names", "redefinition of unused", "is assigned to but never used",
          "f-string is missing placeholders")


def test_pyflakes_reports_no_errors():
    r = subprocess.run([sys.executable, "-m", "pyflakes", *FILES], capture_output=True, text=True)
    real = [l for l in (r.stdout + r.stderr).splitlines() if l.strip() and not any(i in l for i in IGNORE)]
    assert not real, "\n".join(real)


HEADER = ("# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.", "# SPDX-License-Identifier: MIT-0")


def test_every_authored_source_file_carries_the_licence_header():
    """The two-line copyright / SPDX header sits at the top (after a shebang when there is one) of every tracked .py, .sh
    and the Dockerfile; the vendored aws-cdk-lib handlers under cdk/template/assets keep their own notice."""
    out = subprocess.run(["git", "ls-files", "*.py", "*.sh", "trainer/sagemaker/Dockerfile"], cwd=REPO, capture_output=True, text=True).stdout
    missing = []
    for rel in filter(None, out.split("\n")):
        if rel.startswith("cdk/template/assets"):
            continue
        head = (REPO / rel).read_text(errors="replace").split("\n")[:3]
        at = 1 if head and head[0].startswith("#!") else 0
        if tuple(head[at:at + 2]) != HEADER:
            missing.append(rel)
    assert not missing, "licence header missing or misplaced in: " + ", ".join(missing)
