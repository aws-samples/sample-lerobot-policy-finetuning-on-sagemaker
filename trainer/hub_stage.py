#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Stage a policy's Hugging Face artefacts into S3 so training jobs can run with network isolation ON.

Runs as a small CPU SageMaker job with the training image and an overridden container entrypoint:
    python3 /opt/ml/code/hub_stage.py --out /opt/ml/checkpoints --artifacts '<json list>' [--token-secret NAME]

Each artefact {repo, revision, role, gated} is downloaded with snapshot_download into the Hugging Face **cache layout**
(<out>/hub/models--org--repo/snapshots/<sha>/ + refs/main), so by-name loads (tokenizers, VLM backbones) resolve offline
when the training job mounts the prefix as channel "hub" and sets HF_HOME to it. The resolved commit sha of every
artefact is written to <out>/_STAGED.json. A token, when a repo is gated, is read from AWS Secrets Manager (the job role
may read that one secret); it never appears in the job spec or the logs.

Only what the policy loads is downloaded (download_patterns): a `tokenizer` / `text_tokenizer` role gets the tokenizer
files alone (facebook/bart-large ships 5.5 GB of Flax/TF/Rust/PyTorch weights next to 2.7 MB of tokenizer files;
google/paligemma-3b-pt-224 11.7 GB of weights the pi0 processor never reads); every other role gets the full snapshot
minus the alternative-framework weights (ONNX exports, *.msgpack, *.h5, *.ot, *.gguf) and, when the repo also ships
safetensors, the PyTorch *.bin duplicates. The patterns used are recorded per artefact in _STAGED.json; a registry entry
may override them with its own `allow_patterns` / `ignore_patterns`.

The pattern filtering applies to the download; so that an earlier unfiltered layout does not persist, <out> is cleared at
the start of every staging run. SageMaker pre-populates <out> from the prefix's current contents, and snapshot_download
keeps any file already present under snapshots/<sha>/, so without the wipe the files the patterns now exclude would
survive a re-stage. The checkpoint sync agent mirrors the deletion to the prefix (the CDK role allows delete under hub/),
and pipeline/hub.py deletes the stale prefix S3-side before submitting the job as well. Manifest schema 3 = patterns
recorded and prefix cleared; pipeline/hub.py re-stages any prefix whose manifest is older.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys


# roles whose artefact is only ever loaded with AutoTokenizer.from_pretrained (lerobot's TokenizerProcessorStep and the
# pi0 / pi0.5 processors); AutoProcessor / trust_remote_code roles (action tokenizers) are NOT in this set on purpose
TOKENIZER_ONLY_ROLES = ("tokenizer", "text_tokenizer")
# what AutoTokenizer resolves offline (verified in the release image against the staged paligemma and bart-large snapshots)
TOKENIZER_PATTERNS = ["tokenizer*", "vocab*", "merges*", "special_tokens_map.json", "added_tokens.json", "config.json",
                      "spiece.model", "sentencepiece*", "chat_template*"]
# files no lerobot policy loads: alternative-framework weights and ONNX exports published next to the safetensors
IGNORE_ALWAYS = ["onnx/*", "*.onnx", "*.msgpack", "*.h5", "*.ot", "*.gguf", "coreml/*", "openvino/*"]
# PyTorch pickles, redundant when the same repo ships safetensors (transformers prefers safetensors when both exist)
TORCH_BIN_PATTERNS = ["*.bin", "pytorch_model.bin.index.json"]


def download_patterns(artefact: dict, repo_files=None) -> tuple:
    """(allow_patterns, ignore_patterns) for snapshot_download, from the artefact's role and the repo's file list.

    repo_files is the listing from the Hub (None when it could not be fetched: then the *.bin duplicates are kept,
    since nothing proves a safetensors copy exists). Explicit allow_patterns / ignore_patterns on the artefact win.
    """
    if artefact.get("allow_patterns") is not None or artefact.get("ignore_patterns") is not None:
        return artefact.get("allow_patterns"), artefact.get("ignore_patterns")
    role = str(artefact.get("role") or "").strip().lower()
    if role in TOKENIZER_ONLY_ROLES:
        return list(TOKENIZER_PATTERNS), None
    ignore = list(IGNORE_ALWAYS)
    if repo_files is not None and any(str(f).endswith(".safetensors") for f in repo_files):
        ignore += TORCH_BIN_PATTERNS
    return None, ignore


def list_repo_files(repo_id: str, revision, repo_type: str, token):
    """The repo's file list from the Hub, or None when the listing fails (the download itself will then say why)."""
    try:
        from huggingface_hub import HfApi
        return HfApi(token=token).list_repo_files(repo_id, revision=revision, repo_type=repo_type)
    except Exception as e:  # noqa: BLE001  (a listing failure must not abort the staging job)
        print(f"list_repo_files {repo_id}: {type(e).__name__}: {e}", file=sys.stderr)
        return None


def read_secret(name: str) -> str:
    import boto3
    from botocore.config import Config
    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    # USER_AGENT_STRING is the solution user-agent token pipeline/hub.py put in the job environment (no repo to read it
    # from in here); unset means an unlabelled call, never a failure
    sm = boto3.client("secretsmanager", region_name=region, config=Config(user_agent_extra=os.environ.get("USER_AGENT_STRING", "")))
    v = sm.get_secret_value(SecretId=name)
    s = v.get("SecretString") or ""
    try:
        d = json.loads(s)
        return d.get("token") or d.get("HF_TOKEN") or next(iter(d.values()))
    except (json.JSONDecodeError, StopIteration, AttributeError):
        return s


def materialize(cache_dir: str) -> int:
    """Replace every symlink under models--*/snapshots/ by the blob it points to and drop blobs/ and .locks/.

    The Hugging Face cache stores files once under blobs/ and links them from snapshots/<sha>/; the S3 mirror of the
    checkpoint directory keeps regular files only (a first staging run lost every snapshot and the training job saw an
    empty cache). Returns the number of files materialised.
    """
    import shutil
    n = 0
    for repo_dir in sorted(os.scandir(cache_dir), key=lambda e: e.name):
        if not repo_dir.is_dir() or not repo_dir.name.startswith(("models--", "datasets--")):
            continue
        snaps = os.path.join(repo_dir.path, "snapshots")
        if not os.path.isdir(snaps):
            continue
        for root, _, files in os.walk(snaps):
            for f in files:
                link = os.path.join(root, f)
                if os.path.islink(link):
                    target = os.path.realpath(link)
                    os.unlink(link)
                    shutil.copy2(target, link)
                    n += 1
        for junk in ("blobs", ".no_exist"):
            shutil.rmtree(os.path.join(repo_dir.path, junk), ignore_errors=True)
    shutil.rmtree(os.path.join(cache_dir, ".locks"), ignore_errors=True)
    return n


def stage(artifacts: list, out: str, token: str = None, downloader=None, lister=None) -> dict:
    if downloader is None:  # lazy: the laptop test suite runs without huggingface_hub
        from huggingface_hub import snapshot_download
        downloader = snapshot_download
    if lister is None:
        lister = list_repo_files
    import shutil
    cache_dir = os.path.join(out, "hub")
    # start from an empty layout: SageMaker pre-populates <out> from the prefix, and snapshot_download keeps any file
    # already present under snapshots/<sha>/, so files the patterns now exclude (and a stale hub/hub/ layer) would survive
    shutil.rmtree(cache_dir, ignore_errors=True)
    marker = os.path.join(out, "_STAGED.json")
    if os.path.exists(marker):
        os.remove(marker)
    os.makedirs(cache_dir, exist_ok=True)
    staged = []
    for a in artifacts:
        repo, rev = a["repo"], a.get("revision") or None
        kind = a.get("repo_type", "model")
        tok = token if a.get("gated") or token else None
        allow, ignore = download_patterns(a, None if str(a.get("role") or "").strip().lower() in TOKENIZER_ONLY_ROLES
                                          else lister(repo, rev, kind, tok))
        # revision comes from the registry (may be null = default branch); the resolved sha is recorded in _STAGED.json
        path = downloader(repo_id=repo, revision=rev, cache_dir=cache_dir, token=tok, repo_type=kind,
                          allow_patterns=allow, ignore_patterns=ignore)  # nosec B615
        sha = os.path.basename(os.path.normpath(path))
        refs = os.path.join(os.path.dirname(os.path.dirname(os.path.normpath(path))), "refs")
        os.makedirs(refs, exist_ok=True)
        if not os.path.exists(os.path.join(refs, "main")):   # a sha-pinned download writes no refs/main; offline by-name loads need it
            with open(os.path.join(refs, "main"), "w") as f:
                f.write(sha)
        size = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(path, followlinks=True) for f in fs)
        staged.append({"repo": repo, "requested_revision": rev, "resolved_sha": sha, "role": a.get("role"), "bytes": size, "path": path,
                       "allow_patterns": allow, "ignore_patterns": ignore})
        print(f"staged {repo}@{sha} ({size / 1e6:.1f} MB, role {a.get('role')}, allow {allow}, ignore {ignore})", flush=True)
    materialised = materialize(cache_dir)
    print(f"materialised {materialised} snapshot file(s); blobs/ removed", flush=True)
    manifest = {"schema": 3, "materialized": True, "cleared": True, "staged_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "hf_home": out, "artifacts": staged, "request": artifacts}
    with open(os.path.join(out, "_STAGED.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # = CheckpointConfig.LocalPath: SageMaker mirrors its CONTENTS to the hub prefix, so the training job sees <prefix>/hub/models--...
    # and <prefix>/_STAGED.json, and HF_HOME=<mounted prefix> resolves the cache (a nested hub/hub/ layout broke the first smoke)
    ap.add_argument("--out", default="/opt/ml/checkpoints")
    ap.add_argument("--artifacts", default=os.environ.get("HUB_ARTIFACTS", "[]"), help="JSON list of {repo, revision, role, gated}")
    ap.add_argument("--token-secret", default=os.environ.get("HF_TOKEN_SECRET", ""))
    a = ap.parse_args(argv)
    hp_path = "/opt/ml/input/config/hyperparameters.json"
    if os.path.exists(hp_path):  # SageMaker passes hyperparameters as strings
        hp = json.load(open(hp_path))
        a.artifacts = hp.get("hub_artifacts", a.artifacts)
        a.token_secret = hp.get("hf_token_secret", a.token_secret)
    artifacts = json.loads(a.artifacts)
    if not artifacts:
        print("nothing to stage", file=sys.stderr)
        return 0
    token = read_secret(a.token_secret) if a.token_secret else None
    m = stage(artifacts, a.out, token)
    print(json.dumps({k: v for k, v in m.items() if k != "request"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
