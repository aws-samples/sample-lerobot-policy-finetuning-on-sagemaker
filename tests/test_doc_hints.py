# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""The FAIL hint that docs/RECORDER_CHECK.md quotes must be the one trainer/check_dataset.py emits.

The worked example in the doc quotes the `--force-small` hint verbatim, and the prose under it states which episode the
automatic split actually holds out. The two are checked separately: the quoted line is read out of the trainer source
(the f-string), and the prose is compared with the split the script really emits on the four-episode fixture, so the
doc cannot drift from either the message or the behaviour without this file going red.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from tests.fixtures.make_mini_v3 import make_mini_v3  # noqa: E402

TRAINER = REPO / "trainer" / "check_dataset.py"
DOC = REPO / "docs" / "RECORDER_CHECK.md"
# say("FAIL", f"only {n} episodes: give ... --force-small (holds out ...)")
HINT = re.compile(r'say\("FAIL",\s*f"(only \{n\} episodes: [^"]*--force-small[^"]*)"\)')


def _trainer_hint(n: int) -> str:
    matches = HINT.findall(TRAINER.read_text())
    assert len(matches) == 1, f"expected exactly one --force-small FAIL hint in {TRAINER.name}, found {len(matches)}"
    return matches[0].replace("{n}", str(n))


def test_worked_example_quotes_the_hint_the_code_emits():
    doc = DOC.read_text()
    quoted = [line for line in doc.split("\n") if line.startswith("FAIL only ") and "--force-small" in line]
    assert quoted, "docs/RECORDER_CHECK.md no longer shows the few-episodes FAIL line in its worked example"
    for line in quoted:
        n = int(re.match(r"FAIL only (\d+) episodes", line).group(1))
        assert line == f"FAIL {_trainer_hint(n)}", f"doc quotes\n  {line}\nbut check_dataset.py emits\n  FAIL {_trainer_hint(n)}"


def test_prose_names_the_episode_force_small_really_holds_out(tmp_path):
    """The sentence under the example must name the held-out episode the script emits, whatever the hint text says."""
    root = make_mini_v3(tmp_path / "few", variant="few_episodes")
    r = subprocess.run([sys.executable, str(TRAINER), str(root), "--force-small", "--emit-split"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    held = json.loads(r.stdout)["held"]
    assert len(held) == 1, held
    assert f"`--force-small` holds out episode {held[0]}," in DOC.read_text()
