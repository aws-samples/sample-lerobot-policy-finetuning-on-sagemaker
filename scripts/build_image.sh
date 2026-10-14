#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# Build the training image on CodeBuild from the current trainer/ tree. Wrapper: see pipeline/build_image.py --help.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
exec python3 -m pipeline.build_image "$@"
