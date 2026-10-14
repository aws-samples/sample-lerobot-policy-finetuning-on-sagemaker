# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# LeRobot 0.6.1 policy training image for SageMaker training jobs (ACT validated; the other policies via the registry).
# Base: SageMaker PyTorch DLC, torch 2.8.0+cu129 / Python 3.12, pinned by digest (the tag is informational only):
#   pytorch-training:2.8.0-gpu-py312-cu129-ubuntu22.04-sagemaker, pushed 2026-08-03, same digest in us-east-1,
#   ap-northeast-1 and eu-central-1 (verified 2026-09-10). The DLC registry region follows the build region.
# Build context is the trainer/ folder:  docker build -f sagemaker/Dockerfile --build-arg DLC_REGION=<region> .
# checkov:skip=CKV_DOCKER_2: a SageMaker training container is a batch job, not a service; there is no health endpoint to check
# checkov:skip=CKV_DOCKER_3: the SageMaker training runtime starts the container as root and mounts /opt/ml for it (the DLC
#   base does the same); the job runs with network isolation and no inbound path, see docs/SECURITY.md
ARG DLC_REGION=us-east-1
FROM 763104351884.dkr.ecr.${DLC_REGION}.amazonaws.com/pytorch-training@sha256:2d26d620845e569b448904e9a6f82daf2513d6900a77825ad7db170ba28760e9

# ffmpeg: the in-job downscale (libx264, GPL-configured Ubuntu build) and torchcodec's runtime FFmpeg libs.
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/ml/code
# Two pin files, never reorder these COPY lines (they are the pip layer's cache key):
#   constraints.txt   the six pins pip must not move (torch trio, torchcodec, numpy, lerobot)
#   requirements.lock the full freeze of the validated build; LOCK_MODE=enforce (default) installs against it and fails
#                     the build if the freeze differs; LOCK_MODE=generate ignores it and produces a new one
COPY sagemaker/constraints.txt sagemaker/requirements.lock /opt/ml/code/
ARG LOCK_MODE=enforce
# One image for every supported policy family. `lerobot[groot]` is deliberately NOT used: it drags decord 0.6.0, whose
# py3.12 wheel fails `pip check`, while the GR00T code never imports it; its real needs (dm-tree, timm, peft,
# transformers, diffusers) are installed explicitly. torchcodec is resolved for the install, then removed: its prebuilt
# libs cannot load against the DLC's statically linked Python, so video decoding goes through PyAV.
RUN set -e; \
    if [ "$LOCK_MODE" = "enforce" ]; then C="-c /opt/ml/code/constraints.txt -c /opt/ml/code/requirements.lock"; \
    else C="-c /opt/ml/code/constraints.txt"; fi; \
    pip install --no-cache-dir $C "lerobot[training,diffusion,smolvla,pi,xvla,peft]==0.6.1" pyarrow "dm-tree>=0.1.8,<1.0.0" "timm>=1.0.0,<1.1.0"; \
    pip freeze --all | grep -Ev '^(-e |#)|( @ )' > /opt/ml/code/requirements.lock.built; \
    if [ "$LOCK_MODE" = "enforce" ]; then \
      grep -v '^#' /opt/ml/code/requirements.lock > /tmp/lock.expected; \
      diff -u /tmp/lock.expected /opt/ml/code/requirements.lock.built \
        || { echo "FATAL: pip freeze differs from sagemaker/requirements.lock (see diff above); rebuild with LOCK_MODE=generate to refresh the lock"; exit 1; }; \
    fi; \
    pip uninstall -y torchcodec; \
    pip check || echo "NOTE: pip check reported conflicts inherited from the DLC base (listed above)"
COPY sagemaker/image_asserts.py /opt/ml/code/image_asserts.py
RUN python /opt/ml/code/image_asserts.py
ENV VIDEO_BACKEND=pyav

# ACT constructs its ResNet18 backbone from torchvision's ImageNet weights at policy-creation time
# (46,830,571 bytes from download.pytorch.org). Pre-stage them so the job never reaches out for them, and verify the
# full sha256 of the file: torchvision names a weight file <name>-<first 8 hex of its sha256>.pth, and the digest below
# was computed from the downloaded file (its first 8 hex match the name).
ENV TORCH_HOME=/opt/ml/code/torchcache
RUN mkdir -p ${TORCH_HOME}/hub/checkpoints \
    && curl -fL --retry 3 -o ${TORCH_HOME}/hub/checkpoints/resnet18-f37072fd.pth \
       https://download.pytorch.org/models/resnet18-f37072fd.pth \
    && echo "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec  ${TORCH_HOME}/hub/checkpoints/resnet18-f37072fd.pth" \
       | sha256sum -c -  # pragma: allowlist secret

COPY train_policy.sh eval_openloop.py prepare_dataset_360.sh check_dataset.py downscale_plan.py render_flags.py licence.py amp_driver.py summary.py publish.py smoke_checks.py reload_check.py hub_stage.py select_camera.py ensure_quantiles.py dataset_env.py patch_base_config.py prune_checkpoints.py local_repo_args.py /opt/ml/code/
COPY policies/ /opt/ml/code/policies/
COPY sagemaker/train_entrypoint.sh /opt/ml/code/train_entrypoint.sh
RUN chmod +x /opt/ml/code/*.sh

ENV PYTHONUNBUFFERED=1 TQDM_DISABLE=1
# SageMaker invokes `docker run <image> train`; the argument is ignored.
# The container runs as the SageMaker training runtime starts it (root, like the DLC base): documented in docs/SECURITY.md
# and in the checkov note at the top of this file; a trailing comment is not allowed on an exec-form ENTRYPOINT line.
ENTRYPOINT ["/opt/ml/code/train_entrypoint.sh"]
