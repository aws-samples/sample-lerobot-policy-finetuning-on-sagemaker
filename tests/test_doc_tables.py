# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Every row of every markdown table must have the header's number of cells.

GitHub splits a table row on every unescaped `|`, including one inside a code span, then renders exactly as many cells
as the header has and drops the rest. `docs/RECORDER_CHECK.md` once lost the whole description of its
`max \\|action - state\\|` line that way: the row looked fine in the source and rendered as two truncated fragments.
The fix is `\\|` inside the code span, which this test accepts; fenced code blocks are skipped.
"""
import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
FENCE = re.compile(r"^\s*(```|~~~)")
DELIMITER_ROW = re.compile(r"^\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$")
UNESCAPED_PIPE = re.compile(r"(?<!\\)\|")


def _tracked_markdown():
    out = subprocess.run(["git", "-C", str(REPO), "ls-files", "*.md", "**/*.md"], capture_output=True, text=True)
    return sorted({REPO / p for p in out.stdout.split()}) if out.returncode == 0 else []


def cell_count(row: str) -> int:
    """Cells as GitHub counts them: split on unescaped pipes, ignoring the optional leading and trailing pipe."""
    parts = UNESCAPED_PIPE.split(row.strip())
    if row.strip().startswith("|"):
        parts = parts[1:]
    if row.strip().endswith("|") and not row.strip().endswith("\\|"):
        parts = parts[:-1]
    return len(parts)


def table_rows(text: str):
    """Yield (line_no, header_cells, row_cells) for every body row of every table outside fenced code."""
    lines = text.split("\n")
    in_fence, i = False, 0
    while i < len(lines):
        line = lines[i]
        if FENCE.match(line):
            in_fence = not in_fence
        elif not in_fence and line.lstrip().startswith("|") and i + 1 < len(lines) and DELIMITER_ROW.match(lines[i + 1]):
            header = cell_count(line)
            i += 2
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                yield i + 1, header, cell_count(lines[i])
                i += 1
            continue
        i += 1


def test_every_table_row_has_the_header_cell_count():
    md_files = _tracked_markdown()
    if not md_files:
        pytest.skip("not a git checkout")
    bad = []
    for md in md_files:
        for line_no, header, cells in table_rows(md.read_text(errors="replace")):
            if cells != header:
                bad.append(f"{md.relative_to(REPO)} line {line_no}: {cells} cells, header has {header} "
                           f"(an unescaped | inside a code span? write \\| instead)")
    assert not bad, "\n  ".join([f"{len(bad)} misaligned table row(s):"] + bad)


def test_cell_count_treats_escaped_pipes_as_text():
    assert cell_count("| a | b |") == 2
    assert cell_count("a | b") == 2
    assert cell_count("| `max |x| = D` | text |") == 4
    assert cell_count("| `max \\|x\\| = D` | text |") == 2
