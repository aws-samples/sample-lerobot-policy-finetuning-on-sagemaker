#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# 20-step CPU smoke of the published image on SageMaker (~$0.30). Wrapper: see pipeline/smoke_aws.py --help.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
exec python3 -m pipeline.smoke_aws "$@"
