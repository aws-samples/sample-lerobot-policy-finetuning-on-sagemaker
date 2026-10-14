#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# Build the LeRobot policy training image (one image for every supported policy) and push it to ECR.
# Developer fallback: the CDK stack's CodeBuild project (scripts/build_image.sh, pipeline/build_image.py) is the normal path.
#   ./build_and_push.sh --profile <aws profile> --region <region> [--repo <ecr repository>] [--tag v1]
# --repo defaults to the repository the CDK stack creates, <project>-sagemaker-training, with <project> read from the
# repo-root project_config.json (lerobot-finetune-sagemaker-training when the file is absent). Creates the ECR repository
# if it does not exist. The DLC base image is pulled from the same region.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; CTX="$(dirname "$HERE")"   # build context = trainer/
PROFILE=""; REGION=""; REPO=""; TAG="v1"
while [[ $# -gt 0 ]]; do case "$1" in
  --profile) PROFILE="$2"; shift 2;; --region) REGION="$2"; shift 2;; --repo) REPO="$2"; shift 2;; --tag) TAG="$2"; shift 2;;
  *) echo "unknown arg $1"; exit 2;; esac; done
: "${REGION:?--region required}"
if [[ -z "$REPO" ]]; then
  PROJECT=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('project', 'lerobot-finetune'))" "$CTX/../project_config.json" 2>/dev/null || echo lerobot-finetune)
  REPO="$PROJECT-sagemaker-training"
fi
P=(); [[ -n "$PROFILE" ]] && P=(--profile "$PROFILE")
ACCOUNT=$(aws sts get-caller-identity ${P[@]+"${P[@]}"} --query Account --output text)
REG="$ACCOUNT.dkr.ecr.$REGION.amazonaws.com"
IMAGE="$REG/$REPO:$TAG"

echo "account $ACCOUNT  region $REGION  image $IMAGE"
aws ecr get-login-password --region "$REGION" ${P[@]+"${P[@]}"} | docker login --username AWS --password-stdin "763104351884.dkr.ecr.$REGION.amazonaws.com"
aws ecr get-login-password --region "$REGION" ${P[@]+"${P[@]}"} | docker login --username AWS --password-stdin "$REG"
aws ecr describe-repositories --repository-names "$REPO" --region "$REGION" ${P[@]+"${P[@]}"} >/dev/null 2>&1 \
  || aws ecr create-repository --repository-name "$REPO" --region "$REGION" ${P[@]+"${P[@]}"} \
       --image-scanning-configuration scanOnPush=true --query repository.repositoryUri --output text
docker build --platform linux/amd64 -f "$HERE/Dockerfile" --build-arg DLC_REGION="$REGION" -t "$IMAGE" "$CTX"
docker push "$IMAGE"
echo "pushed: $IMAGE"
