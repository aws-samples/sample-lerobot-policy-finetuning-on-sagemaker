#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# Fine-tune the policies enabled in project_config.json on the dataset in datasets/<name>/ (see pipeline/run.py --help).
# Needs: Python >= 3.10 with `pip install -r requirements.txt`, the AWS CLI v2, and a deployed CDK stack (cdk/).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
PY="${PYTHON:-python3}"
# Python 3.9 is end-of-life and resolves dependency versions with known advisories (botocore keeps urllib3 1.26 there);
# the pipeline still runs on it, but the supported floor is 3.10 (README, requirements.txt, CI).
"$PY" -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" 2>/dev/null || echo "WARNING: $PY is older than Python 3.10; upgrade or set PYTHON=<python3.10+> (see docs/SECURITY.md, Dependencies)" >&2
"$PY" -c "import boto3, yaml, pyarrow, jsonschema, huggingface_hub" 2>/dev/null || { echo "missing Python packages: run  $PY -m pip install -r requirements.txt"; exit 2; }
# The dataset reaches the bucket through `aws s3 sync` (pipeline/run.py, stage_dataset), which runs after the plan is
# confirmed. Say so before anything starts, but only when an upload from this machine is pending: a dataset whose source
# is `s3` is not uploaded, and --plan, --compat, --dry-run, --train-job and an early --stop-after never reach that step.
if ! command -v aws >/dev/null 2>&1 && "$PY" - "$@" <<'EOF'
import sys
from pathlib import Path
args = sys.argv[1:]
if any(a in ("--plan", "--compat", "--dry-run", "--train-job") or a.startswith("--train-job=") for a in args):
    sys.exit(1)
for i, a in enumerate(args):
    stop = a.split("=", 1)[1] if a.startswith("--stop-after=") else (args[i + 1] if a == "--stop-after" and i + 1 < len(args) else None)
    if stop in ("config", "dataset", "stack", "preflight"):
        sys.exit(1)
cfg_path = Path("project_config.json")
for i, a in enumerate(args):
    if a == "--config" and i + 1 < len(args):
        cfg_path = Path(args[i + 1])
    elif a.startswith("--config="):
        cfg_path = Path(a.split("=", 1)[1])
try:
    from pipeline.config import load_project_config
    cfg = load_project_config(cfg_path)
except Exception:
    sys.exit(1)   # a broken config is run.py's error to report
chosen = [args[i + 1] for i, a in enumerate(args) if a == "--policy" and i + 1 < len(args)] + [a.split("=", 1)[1] for a in args if a.startswith("--policy=")]
pols = {k: v for k, v in cfg["policies"].items() if (k in chosen if chosen else v.get("enabled"))}
sys.exit(0 if any({**cfg["dataset"], **(v.get("dataset") or {})}.get("source") != "s3" for v in pols.values()) else 1)
EOF
then
    echo "the AWS CLI v2 is not on PATH and the dataset upload needs it (aws s3 sync): install it from https://docs.aws.amazon.com/cli/ and run again" >&2
    exit 2
fi
exec "$PY" pipeline/run.py "$@"
