# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""The Hugging Face token lives in AWS Secrets Manager, never in the repo or the chat.

Default secret name: <project>/hf_token (project_config secrets.hf_token overrides it). `finetune.sh` (run.py) and, when run
with `-c hfTokenPrompt=true`, `cdk deploy` (cdk/app.py) call `ensure()`: when the token is needed (a selected policy stages
gated base weights, or the operator asked for the deploy-time prompt) and the secret is missing or rejected by the Hub (401), the operator is prompted once
with a hidden input, the token is validated against https://huggingface.co/api/whoami-v2 and stored as {"token": ...};
the hub staging job reads that one secret (trainer/hub_stage.py) and training jobs never see it.
"""
import getpass
import json
import os
import sys
import urllib.error
import urllib.request

from botocore.exceptions import ClientError

WHOAMI = "https://huggingface.co/api/whoami-v2"


def secret_name(cfg: dict) -> str:
    return (cfg.get("secrets") or {}).get("hf_token") or f"{cfg['project']}/hf_token"


def whoami(token: str, opener=None) -> dict:
    """{'ok': bool, 'user': str|None, 'status': int}; never raises on HTTP errors."""
    req = urllib.request.Request(WHOAMI, headers={"Authorization": f"Bearer {token}"})
    try:
        with (opener or urllib.request.urlopen)(req, timeout=15) as r:
            body = json.loads(r.read().decode())
            return {"ok": True, "user": body.get("name"), "status": 200}
    except urllib.error.HTTPError as e:
        return {"ok": False, "user": None, "status": e.code}
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        return {"ok": False, "user": None, "status": 0, "error": str(e)}


def read_token(sm, name: str):
    """The token string, or None when the secret does not exist."""
    try:
        s = sm.get_secret_value(SecretId=name).get("SecretString") or ""
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in ("ResourceNotFoundException", "AccessDeniedException"):
            return None
        raise
    try:
        d = json.loads(s)
        return d.get("token") or d.get("HF_TOKEN") or next(iter(d.values()), None)
    except (json.JSONDecodeError, TypeError):
        return s.strip() or None


def store_token(sm, name: str, token: str) -> None:
    body = json.dumps({"token": token})
    try:
        sm.create_secret(Name=name, Description="Hugging Face token for the LeRobot fine-tuning base-weight staging job (read by hub_stage.py only)", SecretString=body)
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") != "ResourceExistsException":
            raise
        sm.put_secret_value(SecretId=name, SecretString=body)


def status(sm, name: str, opener=None) -> dict:
    token = read_token(sm, name)
    if token is None:
        return {"state": "missing", "user": None}
    w = whoami(token, opener)
    if w["ok"]:
        return {"state": "ok", "user": w["user"]}
    if w["status"] in (401, 403):
        return {"state": "invalid", "user": None}
    return {"state": "unverified", "user": None, "error": w.get("error") or w["status"]}   # Hub unreachable: keep the stored token


def interactive() -> bool:
    if os.environ.get("HF_TOKEN_PROMPT", "1") in ("0", "false", "no") or os.environ.get("CI"):
        return False
    try:
        return os.isatty(sys.stdin.fileno()) or os.path.exists("/dev/tty")
    except (OSError, ValueError):
        return False


def ensure(sm, name: str, *, log=print, prompt=None, opener=None, required: bool = True) -> dict:
    """Make sure `name` holds a token the Hub accepts. Prompts (hidden input) when missing/invalid and a terminal is present;
    returns the final status. `required=False` only reports."""
    st = status(sm, name, opener)
    if st["state"] == "ok":
        log(f"  OK   hf token     {name}: valid (Hugging Face user {st['user']})")
        return st
    if st["state"] == "unverified":
        log(f"  WARN hf token     {name}: stored but the Hub could not be reached ({st.get('error')}); using it as is")
        return st
    why = "not found" if st["state"] == "missing" else "rejected by the Hub (401): rotated or revoked"
    if not required:
        log(f"  NOTE hf token     {name}: {why}; reported only: a real run asks for it before staging (hidden input) or, without a terminal, exits 4")
        return st
    if not (prompt or (interactive() and getpass.getpass)):
        raise RuntimeError(f"Hugging Face token secret {name} {why}. Store one: aws secretsmanager put-secret-value --secret-id {name} "
                           f"--secret-string '<token>' (or create-secret --name {name} ...) and re-run; the account must have accepted the gated repos' terms.")
    ask = prompt or (lambda msg: getpass.getpass(msg))
    for attempt in range(3):
        token = ask(f"Hugging Face token for {name} ({why}; input hidden, stored in Secrets Manager, never printed): ").strip()
        if not token:
            continue
        w = whoami(token, opener)
        if w["ok"]:
            store_token(sm, name, token)
            log(f"  OK   hf token     {name}: stored and valid (Hugging Face user {w['user']})")
            return {"state": "ok", "user": w["user"]}
        log(f"  FAIL hf token     the Hub rejected that token (HTTP {w['status']}); try again")
    raise RuntimeError(f"no valid Hugging Face token stored in {name} after 3 attempts")
