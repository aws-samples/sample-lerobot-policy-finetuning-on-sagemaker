#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# Wrapper so the gate is one command in CI and in the pre-commit hook.
set -euo pipefail
exec python3 "$(dirname "${BASH_SOURCE[0]}")/scrub_check.py" "$@"
