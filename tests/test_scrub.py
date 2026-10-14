# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""The scrub gate must catch every identifier shape we have leaked before and ignore floats and digests."""
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "scrub_check.py"
sys.path.insert(0, str(SCRIPT.parent))
import scrub_check  # noqa: E402

ACCT = "9" * 12  # built at runtime so the literal never sits in a tracked file
LEAKS = [
    f"image: {ACCT}.dkr.ecr.eu-central-1.amazonaws.com/act-training:v1",
    f"role: arn:aws:iam::{ACCT}:role/some-execution-role",
    f"bucket: s3://sagemaker-project-{ACCT}-ap-northeast-1/datasets/",
    f"account {ACCT} owns it",
]
CLEAN = [
    '"joints_mae": 0.000107444750,',                       # float inside an eval fixture
    "digest sha256:5e0ed2da000000000000c1a2b3d4e5f6",       # hex digest contains 12-digit runs
    "FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.8.0",  # allow-listed DLC registry
    "arn:aws:iam::123456789012:role/example",  # allow-listed documentation placeholder
    "step 1234567890123 is thirteen digits",
]



def test_leaks_are_caught(tmp_path, monkeypatch):
    monkeypatch.setattr(scrub_check, "ROOT", pathlib.Path("/"))
    for line in LEAKS:
        f = tmp_path / "leak.md"
        f.write_text(line + "\n")
        problems = scrub_check.scan([str(f).lstrip("/")], {"763104351884", "123456789012"}, [])
        assert problems, f"not caught: {line}"


def test_clean_lines_pass(tmp_path, monkeypatch):
    monkeypatch.setattr(scrub_check, "ROOT", pathlib.Path("/"))
    f = tmp_path / "clean.md"
    f.write_text("\n".join(CLEAN) + "\n")
    assert scrub_check.scan([str(f).lstrip("/")], {"763104351884", "123456789012"}, []) == []


def test_denylist_words(tmp_path, monkeypatch):
    monkeypatch.setattr(scrub_check, "ROOT", pathlib.Path("/"))
    f = tmp_path / "d.md"
    f.write_text("the bucket example-secret-bucket-name is internal\n")
    assert scrub_check.scan([str(f).lstrip("/")], set(), ["example-secret-bucket-name"])
    assert scrub_check.scan([str(f).lstrip("/")], set(), ["other"]) == []


def test_denylist_is_case_insensitive_substring_match(tmp_path, monkeypatch):
    """A deny entry planted in another case is caught; the match is a plain substring test, so an entry inside a longer
    token is caught too, while a line that merely reuses the entry's letters in other words is clean."""
    monkeypatch.setattr(scrub_check, "ROOT", pathlib.Path("/"))
    f = tmp_path / "case.md"
    f.write_text("role EXAMPLE-SECRET-ROLE is internal\n")
    problems = scrub_check.scan([str(f).lstrip("/")], set(), ["Example-Secret-Role"])
    assert problems == [f"{str(f).lstrip('/')}:1: deny-listed string (Exa…)"]
    f.write_text("customer AcmeCorp signed\n")
    assert scrub_check.scan([str(f).lstrip("/")], set(), ["acmecorp"])
    f.write_text("bucket acmecorp-datasets-eu\n")  # substring semantics: entry inside a longer token
    assert scrub_check.scan([str(f).lstrip("/")], set(), ["ACMECORP"])
    f.write_text("an acme corp example, secret role names are redacted\n")  # letters present, entry absent
    assert scrub_check.scan([str(f).lstrip("/")], set(), ["acmecorp", "example-secret-role"]) == []


def test_js_and_lock_files_are_scanned_but_png_is_not(tmp_path, monkeypatch):
    monkeypatch.setattr(scrub_check, "ROOT", pathlib.Path("/"))
    leak = f"arn:aws:iam::{ACCT}:role/x\n"
    for name in ("handler.js", "requirements.lock", "picture.png"):
        (tmp_path / name).write_text(leak)
    assert {".js", ".lock", ".json", ".cfg"} <= scrub_check.TEXT_SUFFIXES
    for name in ("handler.js", "requirements.lock"):
        problems = scrub_check.scan([str(tmp_path / name).lstrip("/")], set(), [])
        assert problems and "account id" in problems[0], f"{name} not scanned: {problems}"
    assert scrub_check.scan([str(tmp_path / "picture.png").lstrip("/")], set(), []) == []
    # the deny-list check reaches the new suffixes too
    assert scrub_check.scan([str(tmp_path / "handler.js").lstrip("/")], {ACCT}, ["ROLE/X"])


def test_case_duplicate_deny_entries_report_one_finding(tmp_path, monkeypatch):
    """Entries that differ only by case collapse to one match per line; the first spelling is the one redacted."""
    monkeypatch.setattr(scrub_check, "ROOT", pathlib.Path("/"))
    f = tmp_path / "dup.md"
    f.write_text("acme\n")
    problems = scrub_check.scan([str(f).lstrip("/")], set(), ["Acme", "ACME"])
    assert len(problems) == 1
    assert problems == [f"{str(f).lstrip('/')}:1: deny-listed string (Acm…)"]


def test_invalid_utf8_text_file_is_still_scanned(tmp_path, monkeypatch):
    """A stray non-UTF-8 byte must not exempt the rest of the file: bad bytes are replaced, the leak is still found."""
    monkeypatch.setattr(scrub_check, "ROOT", pathlib.Path("/"))
    f = tmp_path / "handler.js"
    f.write_bytes(b"caf\xe9 example-secret-bucket-name arn:aws:iam::" + ACCT.encode() + b":role/x\n")
    problems = scrub_check.scan([str(f).lstrip("/")], set(), ["example-secret-bucket-name"])
    assert any("account id" in p for p in problems), problems
    assert any("deny-listed string (exa…)" in p for p in problems), problems


def test_handoff_folder_never_tracked():
    tracked = subprocess.run(["git", "ls-files", "local_folder"], cwd=ROOT, capture_output=True, text=True).stdout
    assert tracked.strip() == ""
    assert scrub_check.scan(["local_folder/README.md"], set(), []) == ["local_folder/README.md: private working folder must never be tracked"]


def test_repo_is_clean():
    r = subprocess.run([sys.executable, str(SCRIPT)], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


# ---------- patterns beyond account ids (added for the public release) ----------
PATTERN_LEAKS = {
    "aws access key": "aws_access_key_id = AKIA" + "ABCDEFGHIJKLMNOP",
    "hf token": "HF_TOKEN=hf_" + "a" * 34,
    "private key": "-----BEGIN RSA " + "PRIVATE KEY-----",
    "private ip": "alice pts/3 2026-09-22 (172.16." + "5.7)",
    "internal host": "see https://w.amazon" + ".com/bin/view/Some/Page",
    "amazon email": "Author: Some One <someone@amazon" + ".com>",
    "aws resource id": "launched i-0" + "bb04c348ce1561b" + "e into the block",
    "cdk bucket name": "s3://lerobot-finetune-infra-assetsbucket5cb76180-" + "abcdefghij12/x",
}
PATTERN_CLEAN = [
    "nvidia-curand-cu12==10.3.10.19",              # a version string, not an address
    "pip install torch==2.8.0 numpy==192.168.1.1", # still a version-shaped token
    "https://docs.aws.amazon.com/sagemaker/",      # public documentation host
    "https://aws.amazon.com/security/vulnerability-reporting/",
    "opensource-codeofconduct@amazon.com",          # allow-listed literal
    "AKIAIOSFODNN7EXAMPLE",                         # allow-listed AWS documentation example key  # pragma: allowlist secret
    "sha256:ba67992091ad2da0c38917b611831d34583619d16c94dc1de94f1aa32c5a919f",
    "sm-i-12345678 is not an EC2 id; i-1234 is too short",
]


def _allow():
    return set(scrub_check.load_list(ROOT / "scripts" / "scrub_allowlist.txt"))


def test_every_pattern_catches_its_leak():
    for kind, line in PATTERN_LEAKS.items():
        hits = scrub_check.scan_line(line, _allow(), [])
        assert hits and hits[0].startswith(kind), f"{kind}: {line!r} -> {hits}"
        assert line.split()[-1] not in " ".join(hits), f"{kind}: finding prints the value in full: {hits}"


def test_clean_lines_pass_every_pattern():
    for line in PATTERN_CLEAN:
        assert scrub_check.scan_line(line, _allow(), []) == [], line


def test_history_mode_reports_message_leaks_and_placeholder_identities(tmp_path):
    repo = tmp_path / "r"
    repo.mkdir()
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "a@b", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "a@b",
           "HOME": str(tmp_path), "PATH": "/usr/bin:/bin"}
    def git(*a):
        subprocess.run(["git", *a], cwd=repo, check=True, capture_output=True, env=env)
    git("init", "-q")
    (repo / "f.md").write_text("clean line\n")
    git("add", "f.md")
    git("commit", "-q", "-m", "first\n\nwho output pasted: alice pts/0 (172.16." + "9.9)")
    (repo / "f.md").write_text("bucket example-bucket5cb76180-" + "abcdefghij12\n")
    git("commit", "-q", "-am", "second")
    git("tag", "rc1")
    problems, stats = scrub_check.scan_history(repo, set(), ["alice"])
    kinds = " ".join(problems)
    assert "private ip" in kinds and "deny-listed string (ali…)" in kinds and "cdk bucket name" in kinds, problems
    assert any("placeholder author" in p and "t <a@b" in p for p in problems), problems
    assert stats == {"commits": 2, "identities": 1, "tags": 1, "refs": 2}, stats
    assert "172.16." + "9.9" not in kinds and "abcdefghij12" not in kinds  # findings are masked


def test_history_mode_is_clean_for_a_clean_repo_with_a_real_identity(tmp_path):
    repo = tmp_path / "r"
    repo.mkdir()
    env = {"GIT_AUTHOR_NAME": "Dev", "GIT_AUTHOR_EMAIL": "dev@example-company.org", "GIT_COMMITTER_NAME": "Dev",
           "GIT_COMMITTER_EMAIL": "dev@example-company.org", "HOME": str(tmp_path), "PATH": "/usr/bin:/bin"}
    def git(*a):
        subprocess.run(["git", *a], cwd=repo, check=True, capture_output=True, env=env)
    git("init", "-q")
    (repo / "f.md").write_text("Deploys in us-east-1 with arn:aws:iam::123456789012:role/x\n")
    git("add", "f.md")
    git("commit", "-q", "-m", "one neutral commit")
    problems, stats = scrub_check.scan_history(repo, {"123456789012"}, [])
    assert problems == [], problems
    assert stats["commits"] == 1 and stats["tags"] == 0


def test_history_mode_cli_on_this_checkout_runs():
    """Exit 0 or 1 (findings are expected until the history is squashed), never a crash."""
    r = subprocess.run([sys.executable, str(SCRIPT), "--history"], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode in (0, 1), r.stdout + r.stderr
    assert r.stdout.rstrip().splitlines()[-1].startswith("scrub history "), r.stdout[-300:]


def _git_repo(tmp_path, email="dev@example-company.org", name="Dev"):
    repo = tmp_path / "repo"
    repo.mkdir()
    env = {"GIT_AUTHOR_NAME": name, "GIT_AUTHOR_EMAIL": email, "GIT_COMMITTER_NAME": name, "GIT_COMMITTER_EMAIL": email,
           "HOME": str(tmp_path), "PATH": "/usr/bin:/bin"}
    def git(*a):
        subprocess.run(["git", *a], cwd=repo, check=True, capture_output=True, env=env)
    git("init", "-q", "-b", "main")
    return repo, git


def test_history_mode_scans_merge_resolutions(tmp_path):
    """A leak that exists only in a merge commit's conflict resolution is still found (git log -m)."""
    repo, git = _git_repo(tmp_path)
    f = repo / "f.txt"
    f.write_text("base\n"); git("add", "f.txt"); git("commit", "-q", "-m", "base")
    git("checkout", "-q", "-b", "side"); f.write_text("side\n"); git("commit", "-q", "-am", "side")
    git("checkout", "-q", "main"); f.write_text("main\n"); git("commit", "-q", "-am", "main")
    subprocess.run(["git", "merge", "side"], cwd=repo, capture_output=True,
                   env={"GIT_AUTHOR_NAME": "Dev", "GIT_AUTHOR_EMAIL": "dev@example-company.org", "GIT_COMMITTER_NAME": "Dev",
                        "GIT_COMMITTER_EMAIL": "dev@example-company.org", "HOME": str(tmp_path), "PATH": "/usr/bin:/bin"})  # conflicts
    f.write_text("resolved arn:aws:iam::" + "9" * 12 + ":role/evil\n"); git("add", "f.txt"); git("commit", "-q", "-m", "merge side")
    problems, stats = scrub_check.scan_history(repo, set(), [])
    assert any("account id 9999…" in p for p in problems), problems
    assert stats["commits"] == 4


def test_history_mode_applies_patterns_and_denylist_to_identities(tmp_path):
    repo, git = _git_repo(tmp_path, email="jalias@amazon" + ".com", name="Jane Alias")
    (repo / "f.txt").write_text("clean\n"); git("add", "f.txt"); git("commit", "-q", "-m", "clean commit")
    problems, _ = scrub_check.scan_history(repo, set(), ["jalias"])
    kinds = " ".join(problems)
    assert "deny-listed string (jal…)" in kinds, problems          # the alias in the identity is caught via the deny-list
    assert "amazon email" not in kinds                              # a work address as the author is accepted, not a leak
    assert "jalias@amazon" not in kinds and "Jane Alias" not in kinds  # identities are masked in the report
    assert scrub_check.scan_history(repo, set(), [])[0] == []     # without the deny-list the work address alone is clean


def test_history_mode_dedupes_on_the_full_value_not_the_mask(tmp_path):
    repo, git = _git_repo(tmp_path)
    (repo / "f.txt").write_text("a i-0" + "1234567890abcdef" + "\nb i-0" + "fedcba0987654321" + "\n")
    git("add", "f.txt"); git("commit", "-q", "-m", "two ids with the same mask")
    problems, _ = scrub_check.scan_history(repo, set(), [])
    assert sum("aws resource id" in p for p in problems) == 2, problems


def test_private_ip_shapes():
    allow = _allow()
    for line in ("SIM_URL=http://10.0.0." + "5:8003/health", "host: 192.168.1." + "10", "--host=172.31.4." + "4", "the box was 10.0.0." + "7."):
        assert any(h.startswith("private ip") for h in scrub_check.scan_line(line, allow, [])), line
    for line in ("nvidia-curand-cu12==10.3.10.19", "numpy>=10.0.0.1", "8.8.8.8 is public", "v10.0.0.1 is a tag"):
        assert scrub_check.scan_line(line, allow, []) == [], line


def test_cli_rejects_unknown_options_and_missing_paths():
    for args in (["--histroy"], ["no/such/file.md"]):
        r = subprocess.run([sys.executable, str(SCRIPT), *args], cwd=ROOT, capture_output=True, text=True)
        assert r.returncode == 2 and r.stdout.startswith("scrub ERROR"), (args, r.returncode, r.stdout[:200])
