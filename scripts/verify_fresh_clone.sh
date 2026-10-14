#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# Maintainer check of a release candidate on a FRESH CLONE of this repository.
#
# Clones the repository the script lives in at one git ref into a temporary directory (so uncommitted files, the
# git-ignored private working folder and cached virtualenvs can never leak into the result), then mirrors the host job of
# .github/workflows/ci.yml step by step and adds the developer lanes CI cannot run (docker image tests, CDK template
# tests) when the tools are present. Prints a Markdown report with PASS / FAIL / SKIP per step.
#
#   bash scripts/verify_fresh_clone.sh                       # HEAD of this checkout, report to stdout + <tmp>/report.md
#   bash scripts/verify_fresh_clone.sh --ref v1.2.0 --out /tmp/rc.md --denylist ~/.scrub_denylist
#
# Exit code 0 only when no step failed. Optional lanes (docker, cdk) report SKIP, never FAIL, when their tools are
# missing; when they run and fail, the run fails. The only thing the script ever deletes is its own mktemp directory.
set -uo pipefail

usage() {
    cat <<'EOF'
usage: scripts/verify_fresh_clone.sh [--ref <git ref>] [--denylist <path>] [--cdk-python <python>]
                                     [--skip-docker] [--skip-cdk] [--keep] [--out <report.md>]

  --ref <git ref>       commit, branch or tag of THIS repository to verify (default: HEAD of the checkout)
  --denylist <path>     private deny-list copied into the clone as .scrub_denylist (git-ignored) for scripts/scrub_check.sh
  --cdk-python <python> interpreter of a virtualenv that has cdk/requirements.txt installed (default: cdk/.venv/bin/python
                        of this checkout when it exists); the cdk lane also needs the `cdk` CLI on PATH
  --skip-docker         do not run the docker lane (pytest -m docker against the local training image)
  --skip-cdk            do not run the cdk lane (tests/test_cdk_template.py)
  --keep                keep the temporary directory (clone, venv, logs, report) instead of deleting it
  --out <report.md>     write the report here (default: <tmp>/report.md, printed to stdout as it is produced)
  -h, --help            this text

Steps (FAIL in any of them gives exit code 1): fresh clone at --ref; local_folder/ absent; fresh venv with the newest
python3 >= 3.10 on PATH (python3.10 preferred, as in CI) and pip install -r requirements-dev.txt; scripts/scrub_check.sh; detect-secrets against the audited baseline;
scripts/scrub_check.py --history over every commit and ref (ids, keys, tokens, addresses, hosts, identities); pytest (host lanes); policies doc check;
bash -n over tracked *.sh; bandit exactly as ci.yml runs it; docker lane and cdk lane (optional); README-referenced paths;
python -m pipeline.run --help; trainer/check_dataset.py --help; repo stats. Every 12-digit id (bare, in ARNs, URIs,
bucket names, scrub findings) is masked to ************ in the report, so it can be pasted into an issue.
EOF
}

# ---------- arguments ----------
REF=HEAD DENYLIST= CDKPY= SKIP_DOCKER=0 SKIP_CDK=0 KEEP=0 OUT=
while [ $# -gt 0 ]; do
    case "$1" in
        --ref) REF=${2:?--ref needs a value}; shift 2 ;;
        --denylist) DENYLIST=${2:?--denylist needs a path}; shift 2 ;;
        --cdk-python) CDKPY=${2:?--cdk-python needs a path}; shift 2 ;;
        --skip-docker) SKIP_DOCKER=1; shift ;;
        --skip-cdk) SKIP_CDK=1; shift ;;
        --keep) KEEP=1; shift ;;
        --out) OUT=${2:?--out needs a path}; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

# ---------- locate the repository; resolve path options against the caller's cwd before any cd ----------
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P) || { echo "cannot resolve the script directory" >&2; exit 2; }
SRC=$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null) || { echo "$SCRIPT_DIR is not inside a git repository" >&2; exit 2; }
SHA=$(git -C "$SRC" rev-parse --verify --quiet "${REF}^{commit}") || { echo "unknown ref: $REF" >&2; exit 2; }
abspath() { case "$1" in /*) printf '%s' "$1" ;; *) printf '%s/%s' "$PWD" "$1" ;; esac; }
[ -n "$DENYLIST" ] && DENYLIST=$(abspath "$DENYLIST")
[ -n "$CDKPY" ] && CDKPY=$(abspath "$CDKPY")
if [ -n "$DENYLIST" ] && [ ! -f "$DENYLIST" ]; then echo "--denylist $DENYLIST: no such file" >&2; exit 2; fi
if [ -n "$CDKPY" ] && [ ! -x "$CDKPY" ]; then echo "--cdk-python $CDKPY: not executable" >&2; exit 2; fi
if [ -n "$OUT" ]; then
    mkdir -p "$(dirname "$OUT")" && : > "$OUT" || { echo "cannot write $OUT" >&2; exit 2; }
    OUT=$(cd "$(dirname "$OUT")" && pwd -P)/$(basename "$OUT")
fi
# From here on the cwd is the source checkout, which outlives the run (the clone is removed on exit).
cd "$SRC" || exit 2

W=$(mktemp -d "${TMPDIR:-/tmp}/verify_fresh_clone.XXXXXX") || { echo "mktemp failed" >&2; exit 2; }
R=$W/repo
VENV=$W/venv
REPORT=${OUT:-$W/report.md}
: > "$REPORT"

cleanup() {
    cd / || true  # never sit inside the directory being removed
    if [ "$KEEP" = 1 ]; then
        echo "kept: $W" >&2
    else
        case "$W" in
            */verify_fresh_clone.*) [ -d "$W" ] && rm -rf "$W" ;;  # only ever our own mktemp directory
        esac
    fi
}
trap cleanup EXIT

# ---------- report helpers ----------
PASS_N=0 FAIL_N=0 SKIP_N=0
mask() {  # every 12-digit run that scrub_check.py's ACCOUNT_RE would flag (not inside a hex digest, not part of a float)
    # becomes ************ before anything is printed: bare ids in LEAK lines, ARNs, ECR registries, s3:// and https URIs,
    # bucket names. The loop re-applies the rule until no id is left; the replacement has no digits, so it terminates.
    sed -E -e ':b' -e 's/(^|[^0-9A-Fa-f.])[0-9]{12}([^0-9A-Fa-f.]|\.([^0-9]|$)|$)/\1************\2/' -e 't b'
}
emit() { printf '%s\n' "$*" | mask | tee -a "$REPORT"; }
section() { emit ""; emit "## $*"; emit ""; }
step() {  # step <name> <PASS|FAIL|SKIP> <detail>
    case "$2" in PASS) PASS_N=$((PASS_N + 1)) ;; FAIL) FAIL_N=$((FAIL_N + 1)) ;; SKIP) SKIP_N=$((SKIP_N + 1)) ;; esac
    emit "- **$1**: $2 :: $3"
}
detail() { while IFS= read -r line; do emit "    $line"; done; }  # indented block under a step
last_line() { printf '%s' "$1" | tail -n 1 | cut -c1-300; }
pytest_extras() { printf '%s\n' "$1" | grep -E '^(SKIPPED|FAILED|ERROR)' | detail; }

emit "# Fresh-clone verification of $REF ($(date -u +%FT%TZ))"
emit ""

# ---------- clone ----------
section "Fresh clone"
if git clone -q "$SRC" "$R" 2> "$W/clone.log" && git -C "$R" checkout -q --detach "$SHA" 2>> "$W/clone.log"; then
    step "clone at $REF" PASS "$SHA ($(git -C "$R" log -1 --format=%s | cut -c1-90))"
else
    step "clone at $REF" FAIL "$(last_line "$(cat "$W/clone.log")")"
    emit ""; emit "RESULT: FAIL (clone failed, nothing else was checked)"; exit 1
fi
if [ -e "$R/local_folder" ] || git -C "$R" ls-files --error-unmatch local_folder >/dev/null 2>&1; then
    step "local_folder/" FAIL "present in the clone (the private working folder must never be tracked)"
else
    step "local_folder/" PASS "absent from the clone"
fi
if [ -n "$DENYLIST" ]; then
    if cp "$DENYLIST" "$R/.scrub_denylist"; then
        step "deny-list" PASS "copied into the clone as .scrub_denylist ($(grep -c -v -E '^[[:space:]]*(#|$)' "$DENYLIST") entries)"
    else
        step "deny-list" FAIL "could not copy $DENYLIST"
    fi
else
    step "deny-list" SKIP "no --denylist given: the scrub gate checks its built-in patterns only, not private names"
fi

# ---------- python environment ----------
section "Fresh Python environment (CI host-tests job)"
pick_python() {
    local c v
    c=$(command -v python3.10 2>/dev/null) && { printf '%s' "$c"; return 0; }
    for v in 14 13 12 11; do
        c=$(command -v "python3.$v" 2>/dev/null) && { printf '%s' "$c"; return 0; }
    done
    c=$(command -v python3 2>/dev/null) && "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null \
        && { printf '%s' "$c"; return 0; }
    return 1
}
PY_OK=0
if HOSTPY=$(pick_python); then
    if "$HOSTPY" -m venv "$VENV" > "$W/venv.log" 2>&1 && "$VENV/bin/python" -m pip install -q --upgrade pip >> "$W/venv.log" 2>&1; then
        step "venv" PASS "$("$HOSTPY" --version 2>&1) at $HOSTPY"
        if "$VENV/bin/python" -m pip install -q -r "$R/requirements-dev.txt" > "$W/pip.log" 2>&1; then
            PY_OK=1
            step "pip install -r requirements-dev.txt" PASS "$("$VENV/bin/python" -m pip list 2>/dev/null \
                | grep -i -E '^(pyarrow|numpy|boto3|jsonschema|pytest|bandit|pyyaml) ' | tr -s ' ' | tr '\n' ';')"
        else
            step "pip install -r requirements-dev.txt" FAIL "$(tail -n 3 "$W/pip.log" | tr '\n' ' ' | cut -c1-300)"
        fi
    else
        step "venv" FAIL "$(last_line "$(cat "$W/venv.log")")"
    fi
else
    step "venv" FAIL "no python3 >= 3.10 on PATH"
fi
PY=$VENV/bin/python
run_py() {  # run_py <name> <cmd...>: a venv-dependent step; SKIP when the venv is unusable
    if [ "$PY_OK" = 1 ]; then return 0; fi
    step "$1" SKIP "no usable venv"; return 1
}

cd "$R" || exit 1

# ---------- scrub gate ----------
section "Scrub gate"
if run_py "scripts/scrub_check.sh"; then
    out=$(PATH="$VENV/bin:$PATH" bash scripts/scrub_check.sh 2>&1); rc=$?
    if [ $rc -eq 0 ]; then step "scripts/scrub_check.sh" PASS "$(last_line "$out")"
    else step "scripts/scrub_check.sh" FAIL "rc=$rc :: $(last_line "$out")"; printf '%s\n' "$out" | grep '^LEAK' | head -n 20 | detail; fi
fi
# detect-secrets against the audited baseline, as ci.yml runs it: a finding not audited as a false positive fails.
if run_py "detect-secrets (baseline audited)"; then
    if PATH="$VENV/bin:$PATH" detect-secrets scan --baseline .secrets.baseline > "$W/ds_scan.log" 2>&1; then
        un=$("$VENV/bin/python" -c 'import json; r = json.load(open(".secrets.baseline"))["results"]; print(sum(1 for f, ss in r.items() for s in ss if s.get("is_secret") is not False))' 2>/dev/null || echo "?")
        if [ "$un" = 0 ]; then step "detect-secrets (baseline audited)" PASS "no unaudited finding in .secrets.baseline"
        else step "detect-secrets (baseline audited)" FAIL "$un unaudited finding(s) in .secrets.baseline (detect-secrets audit .secrets.baseline)"; fi
    else
        step "detect-secrets (baseline audited)" FAIL "scan did not run: $(last_line "$(cat "$W/ds_scan.log")")"
    fi
fi
# The gate's --history mode over every commit message and diff of ALL refs of the clone (anything that might be
# pushed): account ids, access keys, tokens, private addresses, internal hostnames, EC2 and CDK bucket ids, deny-listed
# names, and author/committer identities that are placeholders. Findings are masked by the gate itself.
# Needs only git and the standard library, so it runs even when the venv could not be built.
HIST_STEP="git history of all refs (scripts/scrub_check.py --history)"
out=$("${HOSTPY:-python3}" scripts/scrub_check.py --history 2>&1); rc=$?
if [ $rc -eq 0 ]; then step "$HIST_STEP" PASS "$(last_line "$out")"
elif [ $rc -eq 1 ]; then step "$HIST_STEP" FAIL "$(last_line "$out")"; printf '%s\n' "$out" | grep '^LEAK' | head -n 20 | detail
else step "$HIST_STEP" FAIL "scan did not run: $(last_line "$out")"; fi

# ---------- tests and static checks ----------
section "Host test suite and CI static checks"
if run_py "pytest"; then
    out=$(PYTHONDONTWRITEBYTECODE=1 "$PY" -m pytest -p no:cacheprovider -rfEs 2>&1); rc=$?
    if [ $rc -eq 0 ]; then step "pytest (pytest.ini lanes: not docker, not aws)" PASS "$(last_line "$out")"
    else step "pytest (pytest.ini lanes: not docker, not aws)" FAIL "rc=$rc :: $(last_line "$out")"; fi
    pytest_extras "$out"
fi
if run_py "python -m pipeline.policies_doc --check"; then
    out=$("$PY" -m pipeline.policies_doc --check 2>&1); rc=$?
    if [ $rc -eq 0 ]; then step "python -m pipeline.policies_doc --check" PASS "docs/POLICIES.md matches the registry"
    else step "python -m pipeline.policies_doc --check" FAIL "rc=$rc :: $(last_line "$out")"; fi
fi
fails=0; total=0; bad=""
while IFS= read -r -d '' f; do
    total=$((total + 1))
    bash -n "$f" 2>/dev/null || { fails=$((fails + 1)); bad="$bad $f"; }
done < <(git ls-files -z '*.sh')
if [ $fails -eq 0 ]; then step "bash -n over $total tracked *.sh" PASS "no syntax errors"
else step "bash -n over $total tracked *.sh" FAIL "$fails file(s):$bad"; fi
BANDIT_CMD=$(grep -E '^[[:space:]]*run: bandit ' .github/workflows/ci.yml 2>/dev/null | head -n 1 \
    | sed -E 's/^[[:space:]]*run: //; s/[[:space:]]+#.*$//')
BANDIT_SRC="from .github/workflows/ci.yml"
if [ -z "$BANDIT_CMD" ]; then
    BANDIT_CMD="bandit -q -r pipeline trainer cdk scripts inference -x tests,cdk/.venv,cdk/cdk.out,cdk/cdk.out-test -ll"
    BANDIT_SRC="ci.yml has no bandit step; using the last known command"
fi
if run_py "bandit"; then
    out=$(PATH="$VENV/bin:$PATH" bash -c "$BANDIT_CMD" 2>&1); rc=$?
    if [ $rc -eq 0 ]; then step "bandit ($BANDIT_SRC)" PASS "\`$BANDIT_CMD\` :: clean at the CI threshold"
    else step "bandit ($BANDIT_SRC)" FAIL "rc=$rc :: \`$BANDIT_CMD\`"; printf '%s\n' "$out" | grep -E '^(>> Issue|   Severity|   Location)' | head -n 30 | detail; fi
fi

# ---------- developer lanes ----------
section "Developer lanes (optional: SKIP when the tools are missing)"
IMAGE=${LEROBOT_TRAIN_IMAGE:-act-sagemaker-training:dev}
if [ "$SKIP_DOCKER" = 1 ]; then
    step "pytest -m docker" SKIP "--skip-docker"
elif ! command -v docker >/dev/null 2>&1; then
    step "pytest -m docker" SKIP "docker is not on PATH"
elif ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    step "pytest -m docker" SKIP "image $IMAGE not present (or daemon unreachable)"
elif run_py "pytest -m docker"; then
    out=$(PYTHONDONTWRITEBYTECODE=1 "$PY" -m pytest -p no:cacheprovider -o addopts="" -q -m docker -rfEs 2>&1); rc=$?
    if [ $rc -eq 0 ]; then step "pytest -m docker ($IMAGE)" PASS "$(last_line "$out")"
    else step "pytest -m docker ($IMAGE)" FAIL "rc=$rc :: $(last_line "$out")"; fi
    pytest_extras "$out"
fi

CDK_EXPLICIT=1
if [ -z "$CDKPY" ]; then CDK_EXPLICIT=0; [ -x "$SRC/cdk/.venv/bin/python" ] && CDKPY=$SRC/cdk/.venv/bin/python; fi
cdk_unavailable() {  # reason, or nothing when the lane can run
    if [ -z "$CDKPY" ]; then echo "no --cdk-python and no cdk/.venv in the checkout"; return; fi
    command -v cdk >/dev/null 2>&1 || { echo "the cdk CLI is not on PATH (npm install -g aws-cdk)"; return; }
    "$CDKPY" -c 'import aws_cdk, cdk_nag, pytest' >/dev/null 2>&1 || echo "$CDKPY lacks aws_cdk/cdk_nag/pytest (pip install -r requirements-dev.txt -r cdk/requirements.txt)"
}
if [ "$SKIP_CDK" = 1 ]; then
    step "cdk template tests" SKIP "--skip-cdk"
else
    reason=$(cdk_unavailable)
    if [ -n "$reason" ]; then
        if [ "$CDK_EXPLICIT" = 1 ]; then step "cdk template tests" FAIL "$reason"; else step "cdk template tests" SKIP "$reason"; fi
    else
        ver=$("$CDKPY" -c 'import importlib.metadata as m; print(m.version("aws-cdk-lib"))' 2>/dev/null)
        out=$(PYTHONDONTWRITEBYTECODE=1 "$CDKPY" -m pytest -p no:cacheprovider -o addopts="" -q -rfEs tests/test_cdk_template.py -m "not docker and not aws" 2>&1); rc=$?
        if [ $rc -eq 0 ]; then step "cdk template tests (aws-cdk-lib $ver, cdk CLI $(cdk --version 2>/dev/null | cut -d' ' -f1))" PASS "$(last_line "$out")"
        else step "cdk template tests (aws-cdk-lib $ver)" FAIL "rc=$rc :: $(last_line "$out")"; fi
        pytest_extras "$out"
    fi
fi

# ---------- quick-start surface ----------
section "Quick-start surface (mechanical checks; the walk-through itself is done by the reviewer)"
# Paths in README backticks must exist when they are repo-anchored: a single-component name (finetune.sh,
# project_config.json) or a path under a tracked top-level entry. Dataset- and job-relative names (_STAGED.json,
# SUMMARY.json, MANIFEST.json, LICENCE_NOTICE.md, meta/info.json, data/..., videos/...) are listed for information only. No matching token at all is a FAIL (the README lost its paths).
top=$(git ls-files | cut -d/ -f1 | sort -u)
missing="" other="" checked=0
while IFS= read -r p; do
    [ -n "$p" ] || continue
    p=${p#./}
    first=${p%%/*}
    case "$p" in
        _STAGED.json|SUMMARY.json|MANIFEST.json|LICENCE_NOTICE.md|meta/*|data/*|videos/*) other="$other $p" ;;   # job and dataset artefacts, not repo paths
        *'*'*) if compgen -G "$p" >/dev/null; then checked=$((checked + 1)); else missing="$missing $p"; fi ;;
        *) if [ -e "$p" ]; then checked=$((checked + 1))
           elif [ "$first" = "$p" ] || printf '%s\n' "$top" | grep -q -x -F "$first"; then missing="$missing $p"
           else other="$other $p"; fi ;;
    esac
done < <(grep -o -E '`[^`]+`' README.md | tr -d '`' | grep -E '^(\./)?[A-Za-z0-9_./*-]+\.(py|sh|md|json|yaml|yml)$' | sort -u)
if [ -n "$missing" ]; then step "README-referenced paths" FAIL "missing:$missing (not repo-anchored:${other:- none})"
elif [ "$checked" -eq 0 ]; then step "README-referenced paths" FAIL "no repo path in README.md backticks (expected finetune.sh, project_config.json, ...)"
else step "README-referenced paths" PASS "$checked exist; not repo-anchored (dataset-relative):${other:- none}"; fi
if run_py "python -m pipeline.run --help"; then
    out=$("$PY" -m pipeline.run --help 2>&1); rc=$?
    if [ $rc -eq 0 ]; then step "python -m pipeline.run --help" PASS "$(printf '%s' "$out" | head -n 1 | cut -c1-120)"
    else step "python -m pipeline.run --help" FAIL "rc=$rc :: $(last_line "$out")"; fi
fi
if run_py "trainer/check_dataset.py --help"; then
    out=$("$PY" trainer/check_dataset.py --help 2>&1); rc=$?
    if [ $rc -eq 0 ]; then step "trainer/check_dataset.py --help" PASS "$(printf '%s' "$out" | head -n 1 | cut -c1-120)"
    else step "trainer/check_dataset.py --help" FAIL "rc=$rc :: $(last_line "$out")"; fi
fi
if [ -f tests/fixtures/make_mini_v3.py ] && run_py "make_mini_v3 --list-variants"; then
    out=$("$PY" tests/fixtures/make_mini_v3.py --list-variants 2>&1); rc=$?
    if [ $rc -eq 0 ]; then step "make_mini_v3 --list-variants" PASS "$(printf '%s\n' "$out" | grep -c .) variants"
    else step "make_mini_v3 --list-variants" FAIL "rc=$rc :: $(last_line "$out")"; fi
fi

# ---------- stats and summary ----------
section "Repo stats"
emit "- tracked files: $(git ls-files | wc -l | tr -d ' ')"
emit "- tests: $(git ls-files 'tests/test_*.py' | wc -l | tr -d ' ') files"
emit "- policies: $(git ls-files 'trainer/policies/*.yaml' | wc -l | tr -d ' ') YAMLs"
emit "- shell scripts: $(git ls-files '*.sh' | wc -l | tr -d ' ')"

section "Summary"
emit "- steps: $PASS_N PASS, $FAIL_N FAIL, $SKIP_N SKIP"
if [ "$FAIL_N" -eq 0 ]; then emit "- RESULT: PASS ($SHA)"; else emit "- RESULT: FAIL ($SHA)"; fi
if [ -n "$OUT" ]; then echo "report: $OUT" >&2
elif [ "$KEEP" = 1 ]; then echo "report: $REPORT" >&2
else echo "report printed above; pass --out <file> or --keep to retain a copy" >&2; fi
[ "$FAIL_N" -eq 0 ]
