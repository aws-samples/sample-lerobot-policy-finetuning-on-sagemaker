# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""The artefact contract, proven offline in the training image: a checkpoint trained by trainer/train_policy.sh,
published by trainer/publish.py the way the job publishes it, loads in inference/predict.py and yields an action
chunk of the trained shape.

Docker lane (`pytest -m docker`; skips when the image or the daemon is missing, see conftest.py). Everything runs
with `--network none`: the fixture videos are encoded through lerobot, the ResNet18 initialisation comes from the
torchvision cache pre-staged in the image, and nothing is fetched from a Hub. One module-scoped container run builds
the fixture, trains ACT for a few CPU steps and publishes; each predict case is its own container on the same mount.

CPU threads: the container inherits every core of the host as OMP threads. On a shared 32-core box that gave 30 s per
ACT step (thread oversubscription); OMP_NUM_THREADS=8 gives well under a second. The trainer itself is untouched.

The negative cases keep the test honest: a truncated model.safetensors must make predict.py fail, and the xfail
records (and fails on anything other than that exact behaviour or its fix) that a corrupted payload (safetensors has no
checksum) currently yields an all-NaN chunk with exit 0.
"""
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import time
import uuid

import pytest

from tests.conftest import TRAIN_IMAGE, docker_lane_unavailable

pytestmark = pytest.mark.docker

REPO = pathlib.Path(__file__).resolve().parent.parent
POLICY = "act"
STEPS, SAVE_FREQ, LOG_FREQ = 20, 10, 5          # two numbered checkpoints: publish must pick the newer one
BATCH, CHUNK = 2, 8
HEIGHT, WIDTH, ACTION_DIM = 144, 256, 16          # container-mode fixture: 16:9 like the validated 360x640, 12 episodes
HELDOUT = "0"
IMAGE_DIGEST = "sha256:" + "0" * 64               # what the entrypoint exports before publish.py (hyperparameter)
DATASET_FINGERPRINT = "fixture-" + "0" * 12
OMP_THREADS = "8"
HEADLINE = re.compile(r"^policy (\w+): (\d+) future actions x (\d+) dims for episode (\d+) frame (\d+)$", re.M)
STEP_LINE = re.compile(r"^STEP (\w+) rc=(\d+) (\d+) s$", re.M)
PRETRAINED_FILES = ("config.json", "model.safetensors", "train_config.json",
                    "policy_preprocessor.json", "policy_postprocessor.json")   # predict.py needs the processors too

# One bash script per container: `step NAME cmd...` logs to /out/logs/NAME.log, prints a parsable timing line and
# stops at the first failure (its log tail goes to stdout so the pytest failure message says why).
STEP_FN = r"""
set -u
mkdir -p /out/logs
step() {
  local name=$1; shift; local t=$SECONDS
  "$@" > "/out/logs/$name.log" 2>&1; local rc=$?
  echo "STEP $name rc=$rc $((SECONDS - t)) s"
  [ "$rc" -eq 0 ] || { echo "--- tail of /out/logs/$name.log"; tail -n 40 "/out/logs/$name.log"; exit "$rc"; }
}
"""

TRAIN_ENV = dict(DS_ROOT="/out/ds", OUT_DIR="/out/run", POLICY=POLICY, STEPS=STEPS, SAVE_FREQ=SAVE_FREQ, LOG_FREQ=LOG_FREQ,
                 BATCH=BATCH, CHUNK=CHUNK, DEVICE="cpu", NUM_WORKERS=0, SEED=1000, HELDOUT=HELDOUT,
                 FORCE_RES=1,                 # the trainer insists on TARGET_H=360 unless told the resolution is deliberate
                 EVAL_CKPTS="last", EVAL_STRIDE=10)   # the held-out evaluation the job runs, on a handful of frames

BUILD_SCRIPT = STEP_FN + f"""
step offline python -c "import socket, sys
try:
    socket.getaddrinfo('huggingface.co', 443)
except socket.gaierror:
    print('offline'); sys.exit(0)
sys.exit(1)"
step fixture python /repo/tests/fixtures/make_mini_v3.py /out/ds --container --height {HEIGHT} --width {WIDTH}
step train env {" ".join(f"{k}={v}" for k, v in TRAIN_ENV.items())} bash /repo/trainer/train_policy.sh
step publish env IMAGE_DIGEST={IMAGE_DIGEST} DATASET_FINGERPRINT={DATASET_FINGERPRINT} \
    python /repo/trainer/publish.py --out-dir /out/run --model-dir /out/model --output-data /out/outdata --policy {POLICY} --status completed
"""


def _docker_bash(workdir, script, timeout):
    """Run `bash -c script` in the training image, offline, as the calling user, with the repo read-only at /repo.
    A timeout force-removes the container, so it cannot keep training or writing into the tmp dir after the test failed."""
    name = "predict-docker-" + uuid.uuid4().hex[:12]
    cmd = ["docker", "run", "--rm", "--name", name, "--network", "none", "--user", "%d:%d" % (os.getuid(), os.getgid()),
           "-e", "HOME=/tmp", "-e", "HF_HOME=/tmp/hf", "-e", "HF_LEROBOT_HOME=/tmp/lerobot", "-e", "HF_HUB_OFFLINE=1",
           "-e", "OMP_NUM_THREADS=" + OMP_THREADS,
           "-v", "%s:/repo:ro" % REPO, "-v", "%s:/out" % workdir, "--entrypoint", "bash", TRAIN_IMAGE, "-c", script]
    assert "none" == cmd[cmd.index("--network") + 1]
    t0 = time.monotonic()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)
        raise
    return r, time.monotonic() - t0


def _timings(stdout):
    return {m.group(1): (int(m.group(2)), int(m.group(3))) for m in STEP_LINE.finditer(stdout)}


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


@pytest.fixture(scope="module")
def artefact(tmp_path_factory):
    """fixture -> train_policy.sh (ACT, CPU, a few steps) -> publish.py, once per module; removed afterwards (~600 MB)."""
    reason = docker_lane_unavailable()
    if reason:
        pytest.skip(reason)
    workdir = tmp_path_factory.mktemp("predict_docker")
    r, wall = _docker_bash(workdir, BUILD_SCRIPT, timeout=420)
    timings = _timings(r.stdout)
    print("\n[predict_docker] build container %.0f s: %s" % (wall, ", ".join("%s %d s" % (k, v[1]) for k, v in timings.items())))
    assert r.returncode == 0, "build container failed (rc %d)\n%s%s" % (r.returncode, r.stdout, r.stderr)
    assert [k for k in timings] == ["offline", "fixture", "train", "publish"] and all(v[0] == 0 for v in timings.values()), r.stdout
    assert "offline" in (workdir / "logs" / "offline.log").read_text()
    yield workdir
    shutil.rmtree(workdir, ignore_errors=True)


def _predict(workdir, model_subdir, name):
    script = STEP_FN + (f"step {name} python /repo/inference/predict.py --model-dir /out/{model_subdir} "
                        f"--dataset-root /out/ds --episode 0 --frame 0 --device cpu")
    r, wall = _docker_bash(workdir, script, timeout=180)
    assert "STEP %s rc=" % name in r.stdout, "container did not run the predict step:\n" + r.stdout + r.stderr
    print("[predict_docker] %s container %.0f s (rc %d)" % (name, wall, r.returncode))
    log = workdir / "logs" / ("%s.log" % name)
    return r.returncode, (log.read_text() if log.is_file() else r.stdout + r.stderr)


def _corrupt_copy(workdir, name, mutate):
    """Copy the published model dir and let `mutate(path_to_model_safetensors)` damage the weights file."""
    dst = workdir / name
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(workdir / "model", dst)
    mutate(dst / "pretrained_model" / "model.safetensors")
    return name


# ---------------------------------------------------------------------------------------------- published artefact
def test_published_checkpoint_layout_and_summary(artefact):
    model, run = artefact / "model", artefact / "run"
    pm = model / "pretrained_model"
    for f in PRETRAINED_FILES:
        assert (pm / f).is_file() and (pm / f).stat().st_size > 0, f
    cfg = json.load(open(pm / "config.json"))
    assert cfg["type"] == POLICY and cfg["chunk_size"] == CHUNK and cfg["n_action_steps"] == CHUNK
    tc = json.load(open(pm / "train_config.json"))
    assert tc["policy"]["type"] == POLICY and tc["steps"] == STEPS and tc["batch_size"] == BATCH
    assert tc["dataset"]["video_backend"] == "pyav"                      # the validated argv (image env VIDEO_BACKEND)

    # the newest numbered checkpoint was published, not the older one and not `last`
    ckpts = sorted(p.name for p in (run / "train" / "checkpoints").iterdir() if p.name.isdigit())
    assert ckpts == ["%06d" % SAVE_FREQ, "%06d" % STEPS]
    published = _sha256(pm / "model.safetensors")
    assert published == _sha256(run / "train" / "checkpoints" / ckpts[-1] / "pretrained_model" / "model.safetensors")
    assert published != _sha256(run / "train" / "checkpoints" / ckpts[0] / "pretrained_model" / "model.safetensors")

    s = json.load(open(model / "SUMMARY.json"))
    assert s["policy"] == POLICY and s["status"] == "completed" and s["checkpoint"] == "pretrained_model/"
    assert s["checkpoint_step"] == STEPS and s["model_sha256"] == published and s["train_exitcode"] == 0
    assert s["image_digest"] == IMAGE_DIGEST and s["dataset_fingerprint"] == DATASET_FINGERPRINT
    assert s["licence_status"] == "accepted" and s["versions"]["lerobot"] == "0.6.1"
    assert s["split"]["held"] == [int(HELDOUT)] and len(s["split"]["train"]) == 11 and s["split"]["action_dim"] == ACTION_DIM
    ev = s["evals"]["eval_last.json"]                                     # the held-out evaluation shipped next to the weights
    assert ev["episodes"] == [int(HELDOUT)] and ev["chunk"] == CHUNK and ev["stride"] == TRAIN_ENV["EVAL_STRIDE"]
    assert ev["joints_mae"] is not None and ev["joints_hold"] is not None
    assert (model / "eval" / "eval_last.json").is_file()
    assert json.load(open(run / "SUMMARY.json")) == s                    # the copy the laptop reads from the checkpoint prefix

    notice = (model / "LICENCE_NOTICE.md").read_text()
    assert "Policy type: act" in notice and "Apache-2.0" in notice
    for name in ("train.log", "losscurve.txt", "split.json", "exitcode"):
        assert (artefact / "outdata" / name).is_file(), name
    assert (artefact / "outdata" / "exitcode").read_text().strip() == "0"
    assert len((artefact / "outdata" / "losscurve.txt").read_text().splitlines()) == STEPS // LOG_FREQ


# ---------------------------------------------------------------------------------------------- predict.py
def test_predict_yields_an_action_chunk_of_the_trained_shape(artefact):
    rc, out = _predict(artefact, "model", "predict")
    assert rc == 0, out
    m = HEADLINE.search(out)
    assert m, out
    assert m.groups() == (POLICY, str(CHUNK), str(ACTION_DIM), "0", "0")
    rows = [line.split(":", 1)[1].split() for line in out.splitlines() if re.match(r"^\s+t\+\d+:", line)]
    assert len(rows) == min(5, CHUNK) and all(len(r) == ACTION_DIM for r in rows)
    values = [float(v) for r in rows for v in r]
    assert all(v == v and abs(v) < 1e6 for v in values), "non-finite or absurd action values"
    assert len(set(values)) > 1, "a constant chunk means the unnormaliser or the head did nothing"
    names = json.load(open(artefact / "ds" / "meta" / "info.json"))["features"]["action"]["names"]
    names = names["axes"] if isinstance(names, dict) else names
    assert "  dims: " + ", ".join(names) in out
    assert "Traceback" not in out


def test_predict_fails_on_a_truncated_weights_file(artefact):
    def truncate(p):
        size = p.stat().st_size
        with open(p, "r+b") as f:
            f.truncate(size // 2)
    rc, out = _predict(artefact, _corrupt_copy(artefact, "model_truncated", truncate), "predict_truncated")
    assert rc != 0 and "SafetensorError" in out, out
    assert not HEADLINE.search(out)


def test_predict_rejects_a_non_finite_chunk(artefact):
    """Recorded gap: predict.py accepts a corrupted safetensors payload (no checksum) and prints an all-NaN chunk
    with exit 0. Passes once it refuses a non-finite chunk (exit != 0, no chunk header, no Traceback, no load error, and a
    message naming the non-finite chunk)."""
    def poison(p):                              # 0xff bytes inside the tensor payload decode as float32 NaN
        size = p.stat().st_size
        with open(p, "r+b") as f:
            f.seek(size // 2)
            f.write(b"\xff" * 4096)
    rc, out = _predict(artefact, _corrupt_copy(artefact, "model_poisoned", poison), "predict_poisoned")
    m = HEADLINE.search(out)
    if rc != 0:
        assert not m and "SafetensorError" not in out and "Traceback" not in out and re.search(r"non-finite|NaN", out), \
            "refused, but not for a non-finite chunk:\n" + out
        return                                   # the fix landed: delete the xfail below
    rows = [line.split(":", 1)[1].split() for line in out.splitlines() if re.match(r"^\s+t\+\d+:", line)]
    assert m and rows and all(v == "nan" for r in rows for v in r), "exit 0 with a chunk that is not all-NaN:\n" + out
    pytest.xfail("inference/predict.py accepts a corrupted safetensors payload (no checksum) and prints an all-NaN chunk with exit 0")
