# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Live view of a running job: tail train.log from the checkpoint prefix and cross-check liveness with checkpoint keys.

lerobot prints `step:8K` from step 1000 on and writes carriage returns; both are handled. Liveness is never inferred
from the metric line alone: the newest checkpoints/<step>/ key is the ground truth.
"""
from __future__ import annotations

import re
from typing import Optional

STEP_RE = re.compile(r"step:([0-9.]+)(K?) .*?loss:([0-9.eE+-]+)")


def parse_step(tok: str, k: str = "") -> int:
    return int(round(float(tok) * (1000 if k else 1)))


def split_lines(chunk: str) -> list:
    return [l for l in re.split(r"[\r\n]+", chunk) if l.strip()]


def latest_metric(lines) -> Optional[dict]:
    for line in reversed(list(lines)):
        m = STEP_RE.search(line)
        if m:
            return {"step": parse_step(m.group(1), m.group(2)), "loss": float(m.group(3)), "line": line.strip()[:200]}
    return None


class LogTail:
    """Incremental Range reads of s3://bucket/key; call read() repeatedly."""

    def __init__(self, s3, bucket: str, key: str):
        self.s3, self.bucket, self.key, self.offset = s3, bucket, key, 0

    def read(self) -> list:
        try:
            resp = self.s3.get_object(Bucket=self.bucket, Key=self.key, Range=f"bytes={self.offset}-")
        except self.s3.exceptions.NoSuchKey:
            return []
        except Exception as e:  # 416 InvalidRange (nothing new) or a transient error
            if "InvalidRange" in str(e) or "416" in str(e):
                return []
            raise
        body = resp["Body"].read()
        self.offset += len(body)
        return split_lines(body.decode(errors="replace"))


def newest_checkpoint(s3, bucket: str, prefix: str) -> Optional[dict]:
    """{'step': int, 'last_modified': datetime} of the newest checkpoints/<step>/ under <prefix>, else None."""
    paginator = s3.get_paginator("list_objects_v2")
    best = None
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix.rstrip("/") + "/train/checkpoints/", Delimiter="/"):
        for cp in page.get("CommonPrefixes", []):
            name = cp["Prefix"].rstrip("/").rsplit("/", 1)[-1]
            if name.isdigit() and (best is None or int(name) > best["step"]):
                best = {"step": int(name), "prefix": cp["Prefix"]}
    return best
