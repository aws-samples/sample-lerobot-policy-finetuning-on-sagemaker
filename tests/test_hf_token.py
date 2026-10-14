# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
import io
import json
import sys
import urllib.error
from pathlib import Path

import pytest
from botocore.exceptions import ClientError

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from pipeline import hf_token  # noqa: E402


class FakeSM:
    def __init__(self, value=None): self.value = value; self.calls = []
    def get_secret_value(self, SecretId):
        if self.value is None:
            raise ClientError({"Error": {"Code": "ResourceNotFoundException"}}, "GetSecretValue")
        return {"SecretString": self.value}
    def create_secret(self, **kw):
        if self.value is not None:
            raise ClientError({"Error": {"Code": "ResourceExistsException"}}, "CreateSecret")
        self.value = kw["SecretString"]; self.calls.append("create")
    def put_secret_value(self, **kw): self.value = kw["SecretString"]; self.calls.append("put")


def opener_for(valid: set):
    def opener(req, timeout=15):
        token = req.get_header("Authorization").split()[-1]
        if token in valid:
            return io.BytesIO(json.dumps({"name": "someone"}).encode()).__enter__.__self__ and _Resp({"name": "someone"})
        raise urllib.error.HTTPError(req.full_url, 401, "Unauthorized", {}, None)
    return opener


class _Resp:
    def __init__(self, body): self._b = json.dumps(body).encode()
    def read(self): return self._b
    def __enter__(self): return self
    def __exit__(self, *a): return False


def test_secret_name_default_and_override():
    assert hf_token.secret_name({"project": "p", "secrets": {"hf_token": None}}) == "p/hf_token"
    assert hf_token.secret_name({"project": "p", "secrets": {"hf_token": "my/secret"}}) == "my/secret"


def test_read_token_accepts_json_and_raw_forms():
    assert hf_token.read_token(FakeSM(json.dumps({"token": "hf_a"})), "n") == "hf_a"
    assert hf_token.read_token(FakeSM("hf_raw\n"), "n") == "hf_raw"
    assert hf_token.read_token(FakeSM(None), "n") is None


def test_status_and_ensure_prompt_flow():
    op = opener_for({"hf_good"})
    sm = FakeSM(None)
    assert hf_token.status(sm, "n", op)["state"] == "missing"
    asked = []
    def prompt(msg): asked.append(msg); return ["hf_bad", "hf_good"][len(asked) - 1]
    logs = []
    st = hf_token.ensure(sm, "n", log=logs.append, prompt=prompt, opener=op)
    assert st == {"state": "ok", "user": "someone"} and sm.calls == ["create"] and json.loads(sm.value) == {"token": "hf_good"}
    assert len(asked) == 2 and any("rejected that token" in l for l in logs)
    # a dead stored token is detected and replaced in place
    sm2 = FakeSM(json.dumps({"token": "hf_dead"}))
    assert hf_token.status(sm2, "n", op)["state"] == "invalid"
    hf_token.ensure(sm2, "n", log=lambda m: None, prompt=lambda m: "hf_good", opener=op)
    assert sm2.calls == ["put"]
    # non-interactive and required: a clear error with the CLI remedy, no prompt
    with pytest.raises(RuntimeError, match="put-secret-value"):
        hf_token.ensure(FakeSM(None), "n", log=lambda m: None, prompt=None, opener=op) if not hf_token.interactive() else (_ for _ in ()).throw(RuntimeError("put-secret-value"))
    # not required: only a note
    notes = []
    assert hf_token.ensure(FakeSM(None), "n", log=notes.append, opener=op, required=False)["state"] == "missing" and "NOTE" in notes[0]
