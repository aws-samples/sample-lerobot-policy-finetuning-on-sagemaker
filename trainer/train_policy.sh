#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# Policy training + held-out evaluation on a LeRobot v3.0 dataset, driven by the environment variables listed below.
# Steps: validate the dataset and choose the split (check_dataset.py), derive the camera-subset and stats views a policy
# may need (select_camera.py, ensure_quantiles.py), run the licence gate (licence.py), render the lerobot-train argv from
# the registry (policies/<POLICY>.yaml via render_flags.py), train fresh or resume, then evaluate the requested
# checkpoints open-loop on the held-out episodes (eval_openloop.py). With POLICY=act and the defaults below the argv is
# the validated ACT recipe (100K steps, batch 8, chunk 24 @ 25 fps = 0.96 s; docs/RESULTS.md).
# Runs in the foreground: sagemaker/train_entrypoint.sh calls it inside the SageMaker job; it also runs directly on any
# GPU host with lerobot 0.6.1.
#
# Required
#   DS_ROOT        dataset dir at 360x640 (make it with ./prepare_dataset_360.sh)
#   OUT_DIR        output dir -> train/ (lerobot's own output dir: checkpoints/<step>/, last), eval/, train.log,
#                  losscurve.txt, split.json, exitcode
# Split (explicit ids are validated against the dataset; the auto split needs >= 10 episodes)
#   HELDOUT        space- or comma-separated held-out episode ids, e.g. "0 13 26"
#   EPISODES       explicit JSON list of TRAINING ids, e.g. "[1,2,3]" (then held-out = the rest)
#   (default)      ~10% of episodes (min 1, max n-1), evenly spaced
# Policy
#   POLICY=act             registry entry under policies/ (lerobot policy type)
#   OVERRIDES=""           extra lerobot-train flags appended last (forbidden flags are refused)
#   ACCEPT_LICENCE=0/1 APPROVAL_REF=  licence gate inputs for non-clean policies (see licence.py)
#   EXTEND_STEPS=N         with RESUME=1: continue a finished run to N steps (constant-LR policies only add --steps)
#   RENDER_ONLY=1          print the rendered argv and exit (tests)
# Training knobs
#   STEPS=100000 BATCH=8 CHUNK=24 SAVE_FREQ=10000 LOG_FREQ=200 NUM_WORKERS=12 SEED=1000 DEVICE=cuda
#   REPO_ID        default local/<basename of DS_ROOT>
#   VIDEO_BACKEND  torchcodec | pyav (unset = lerobot default)
#   RESUME=1       continue from the newest checkpoint under OUT_DIR/train/checkpoints (loud fresh start if none)
# Evaluation (non-fatal: a failing eval is reported, the checkpoint is kept)
#   EVAL_CKPTS="last"   checkpoint names under OUT_DIR/train/checkpoints to evaluate; "" skips evaluation
#   EVAL_STRIDE=1       1 = every held-out frame (what the reference runs in docs/RESULTS.md used); 5 is fine for comparisons
#   SKIP_TRAIN=1        only run the evaluation on an existing OUT_DIR
# Environment
#   PYTHON         interpreter with lerobot 0.6.1 (default: python3 on PATH); lerobot-train is taken from its bin dir
#   TORCH_HOME     torchvision cache; pre-stage hub/checkpoints/resnet18-f37072fd.pth on offline boxes
#   TARGET_H=360             height every camera must have (prepare_dataset_360.sh makes it so)
#   FORCE_RES=1              accept cameras at another height (ACT has no resize; expect worse numbers)
#   FORCE_ACTION_EQ_STATE=1  proceed although action == observation.state bit-exactly (nothing to learn)
#   FORCE_SMALL=1            auto-split fewer than 10 episodes
#   REQUIRE_FLAT_NAMES=1     FAIL on the dict form of action.names (name-based policies)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${DS_ROOT:?set DS_ROOT to the 360x640 LeRobot dataset dir}"
: "${OUT_DIR:?set OUT_DIR}"
POLICY="${POLICY:-act}"
STEPS="${STEPS:-100000}"; BATCH="${BATCH:-8}"; CHUNK="${CHUNK:-24}"
SAVE_FREQ="${SAVE_FREQ:-10000}"; LOG_FREQ="${LOG_FREQ:-200}"; NUM_WORKERS="${NUM_WORKERS:-12}"
SEED="${SEED:-1000}"; DEVICE="${DEVICE:-cuda}"
REPO_ID="${REPO_ID:-local/$(basename "$DS_ROOT")}"
EVAL_CKPTS="${EVAL_CKPTS-last}"; EVAL_STRIDE="${EVAL_STRIDE:-1}"
PYTHON="${PYTHON:-$(command -v python3)}"
LEROBOT_TRAIN="${LEROBOT_TRAIN:-$(dirname "$PYTHON")/lerobot-train}"
[[ -x "$LEROBOT_TRAIN" ]] || LEROBOT_TRAIN="$(command -v lerobot-train || true)"
[[ -x "$LEROBOT_TRAIN" || "${RENDER_ONLY:-0}" == "1" ]] || { echo "FATAL: lerobot-train not found next to $PYTHON or on PATH"; exit 2; }
export TQDM_DISABLE=1 TOKENIZERS_PARALLELISM=false
say() { printf '\n=== %s ===\n' "$*"; }

mkdir -p "$OUT_DIR"
LR_OUT="$OUT_DIR/train"   # lerobot refuses an existing output_dir, so it gets its own subfolder (SageMaker mirrors /opt/ml/checkpoints to S3 itself)

say "dataset sanity + split: $DS_ROOT"
# One validator for the recording host, the laptop and the job (see check_dataset.py --help). Report -> stderr, split -> stdout.
CK=(); [[ -n "${HELDOUT:-}" ]] && CK+=(--heldout "$HELDOUT"); [[ -n "${EPISODES:-}" ]] && CK+=(--episodes "$EPISODES")
[[ "${FORCE_ACTION_EQ_STATE:-0}" == "1" ]] && CK+=(--force-action-eq-state)
[[ "${FORCE_SMALL:-0}" == "1" ]] && CK+=(--force-small)
[[ "${FORCE_RES:-0}" == "1" ]] && CK+=(--force-res)
[[ "${REQUIRE_FLAT_NAMES:-0}" == "1" ]] && CK+=(--require-flat-names)
SPLIT_JSON=$("$PYTHON" "$HERE/check_dataset.py" "$DS_ROOT" --emit-split ${CK[@]+"${CK[@]}"}) || { echo "FATAL: dataset check failed (see report above)"; exit 3; }
# ACT has no resize: every camera must already be at the target height (the SageMaker entrypoint downscales first;
# outside the entrypoint run prepare_dataset_360.sh first). Width follows the aspect ratio: 640 for 16:9, 480 for 4:3.
"$PYTHON" - "$DS_ROOT" "${FORCE_RES:-0}" "${TARGET_H:-360}" <<'PY'
import json, sys
root, force, H = sys.argv[1], sys.argv[2] == "1", int(sys.argv[3])
info = json.load(open(f"{root}/meta/info.json"))
cams = {k: v["shape"] for k, v in info["features"].items() if v.get("dtype") in ("video", "image")}
bad = {k: s for k, s in cams.items() if [d for d in s if d != 3][0] != H}   # HWC or CHW
if bad and not force:
    sys.exit(f"FATAL: cameras not at height {H}: {bad} -- run prepare_dataset_360.sh first (or FORCE_RES=1)")
for k, s in cams.items():
    hw = [d for d in s if d != 3]   # HWC or CHW
    if 640 not in hw:
        print(f"NOTE: {k} is {hw[0]}x{hw[1]}; widths other than 640 are fine (the downscale keeps the aspect ratio)", file=sys.stderr)
PY
echo "$SPLIT_JSON" > "$OUT_DIR/split.json"
# Single-image policies (registry single_camera_required: VQ-BeT, TD-MPC) raise "You must provide only one image among the
# inputs" on a multi-camera dataset. select_camera.py derives a metadata-only one-camera view; CAMERA (project_config
# policies.<name>.camera) picks the camera, default the first in sorted order. The view lives outside OUT_DIR (never mirrored).
KEEP_ARGS=()
if [[ -n "${CAMERAS:-}" ]]; then   # policies.<name>.cameras: an explicit camera subset for any policy (FastWAM needs a count that divides 448)
  IFS=',' read -r -a _cams <<< "$CAMERAS"; for c in "${_cams[@]}"; do KEEP_ARGS+=(--keep "$c"); done
elif [[ "$("$PYTHON" "$HERE/render_flags.py" --policy "$POLICY" --print single_camera_required 2>/dev/null || echo False)" == "True" ]]; then
  KEEP_ARGS=(--single); [[ -n "${CAMERA:-}" ]] && KEEP_ARGS=(--keep "$CAMERA")
fi
if [[ ${#KEEP_ARGS[@]} -gt 0 ]]; then
  [[ "${KEEP_ARGS[0]}" == "--single" ]] && KEEP_ARGS=()
  ONE_CAM_ROOT=$("$PYTHON" "$HERE/select_camera.py" "$DS_ROOT" ${KEEP_ARGS[@]+"${KEEP_ARGS[@]}"} --out "${ONE_CAM_DIR:-/tmp}/$(basename "$DS_ROOT")_cams") || { echo "FATAL: camera selection failed"; exit 3; }
  if [[ "$ONE_CAM_ROOT" != "$DS_ROOT" ]]; then say "camera subset: training and evaluating on $ONE_CAM_ROOT"; export DS_ROOT="$ONE_CAM_ROOT"; fi
fi
# QUANTILES policies (registry requires_quantile_stats: pi0.5, MolmoAct2) raise on the first batch when meta/stats.json lacks
# q01/q99 (after the multi-GB base load). ensure_quantiles.py completes the stats in a metadata-only view of the dataset.
# Every policy: stats.json must have an entry per camera (lerobot indexes stats[<camera>] to apply ImageNet mean/std; some
# converters leave cameras out). QUANTILES policies additionally get q01/q99. Both land in the same metadata-only view.
QFLAG=(); [[ "$("$PYTHON" "$HERE/render_flags.py" --policy "$POLICY" --print requires_quantile_stats 2>/dev/null || echo False)" == "True" ]] && QFLAG=(--quantiles)
Q_ROOT=$("$PYTHON" "$HERE/ensure_quantiles.py" "$DS_ROOT" --out "${ONE_CAM_DIR:-/tmp}/$(basename "$DS_ROOT")_q" ${QFLAG[@]+"${QFLAG[@]}"}) || { echo "FATAL: stats completion failed"; exit 3; }
if [[ "$Q_ROOT" != "$DS_ROOT" ]]; then say "stats.json completed (cameras / q01-q99) -> $Q_ROOT"; export DS_ROOT="$Q_ROOT"; fi
echo "$DS_ROOT" > "$OUT_DIR/dataset_root.txt"   # the entrypoint's artefact reload check must load the same root
# dataset-derived values a registry argv may use: CAMERA_KEYS = image features in info.json order (comma-joined, for
# --policy.obs_cam_keys=[${CAMERA_KEYS}]), ACTION_CHANNEL_IDS = [0..D-1] (for --policy.used_action_channel_ids)
eval "$("$PYTHON" "$HERE/dataset_env.py" "$DS_ROOT")"
# registry base_config_patch / base_patches: edit files of the staged base checkpoint (hub cache snapshot) before training.
# base_config_patch JSON-merges into config.json so a base trained with other cameras / action widths rebuilds its features
# from this dataset (LingBot-VA); base_patches edits other files (pi0-FAST: tokenizer repo id -> staged snapshot path).
BASE_PATCH=$("$PYTHON" "$HERE/render_flags.py" --policy "$POLICY" --print base_config_patch 2>/dev/null || true)
BASE_PATCHES=$("$PYTHON" "$HERE/render_flags.py" --policy "$POLICY" --print base_patches 2>/dev/null || true)
if [[ -n "${HF_HOME:-}" ]] && { [[ -n "$BASE_PATCH" && "$BASE_PATCH" != "None" ]] || [[ -n "$BASE_PATCHES" && "$BASE_PATCHES" != "None" && "$BASE_PATCHES" != "[]" ]]; }; then
  "$PYTHON" "$HERE/patch_base_config.py" --policy "$POLICY" --hf-home "$HF_HOME" || { echo "FATAL: base config patch failed"; exit 3; }
fi
# registry train.local_repo_args: repo ids whose loaders cannot resolve offline -> the staged snapshot directory instead
if [[ -n "${HF_HOME:-}" ]]; then
  LOCAL_ARGS=$("$PYTHON" "$HERE/local_repo_args.py" --policy "$POLICY" --hf-home "$HF_HOME" | tr '\n' ' ') || { echo "FATAL: local repo args failed"; exit 3; }
  [[ -n "${LOCAL_ARGS// /}" ]] && { echo "local repo paths: $LOCAL_ARGS"; export OVERRIDES="${OVERRIDES:-} $LOCAL_ARGS"; }
fi
LG=(); [[ "${ACCEPT_LICENCE:-0}" == "1" ]] && LG+=(--accept); [[ -n "${APPROVAL_REF:-}" ]] && LG+=(--approval-ref "$APPROVAL_REF")
"$PYTHON" "$HERE/licence.py" "$POLICY" ${LG[@]+"${LG[@]}"} || { echo "FATAL: licence gate refused policy $POLICY"; exit 3; }
EVAL_POLICY=$("$PYTHON" "$HERE/render_flags.py" --policy "$POLICY" --print lerobot_type)
EVAL_SAMPLES=$("$PYTHON" "$HERE/render_flags.py" --policy "$POLICY" --print eval.n_samples 2>/dev/null || echo 1)
EVAL_EXTRA=()   # eval protocol keys of the registry (see eval_openloop.predict_chunk): queue policies, single image key, chunk axis, no history
[[ "$("$PYTHON" "$HERE/render_flags.py" --policy "$POLICY" --print eval.history_via_queues 2>/dev/null || echo False)" == "True" ]] && EVAL_EXTRA+=(--history-via-queues)
[[ "$("$PYTHON" "$HERE/render_flags.py" --policy "$POLICY" --print eval.chunk_first 2>/dev/null || echo False)" == "True" ]] && EVAL_EXTRA+=(--chunk-first)
[[ "$("$PYTHON" "$HERE/render_flags.py" --policy "$POLICY" --print eval.no_history 2>/dev/null || echo False)" == "True" ]] && EVAL_EXTRA+=(--no-history)
QIK=$("$PYTHON" "$HERE/render_flags.py" --policy "$POLICY" --print eval.queue_image_key 2>/dev/null || true); [[ -n "$QIK" && "$QIK" != "None" ]] && EVAL_EXTRA+=(--queue-image-key "$QIK")
TOL=$("$PYTHON" "$HERE/render_flags.py" --policy "$POLICY" --print eval.tolerance_s 2>/dev/null || true); [[ -n "$TOL" && "$TOL" != "None" ]] && EVAL_EXTRA+=(--tolerance-s "$TOL")
[[ "$("$PYTHON" "$HERE/render_flags.py" --policy "$POLICY" --print eval.reset_per_call 2>/dev/null || echo False)" == "True" ]] && EVAL_EXTRA+=(--reset-per-call)
PB=$("$PYTHON" "$HERE/render_flags.py" --policy "$POLICY" --print base_checkpoint 2>/dev/null || true); [[ -n "$PB" && "$PB" != "None" ]] && EVAL_EXTRA+=(--peft-base "$PB")
# precision: policies trained in bf16 upstream must NOT be hard-cast (--policy.dtype=bfloat16 discards AdamW updates below the
# bf16 ULP); the registry says fp32_masters_bf16_autocast and amp_driver.py runs lerobot's train() with fp32 masters + autocast
PRECISION=$("$PYTHON" "$HERE/render_flags.py" --policy "$POLICY" --print precision)
if [[ "$PRECISION" == "fp32_masters_bf16_autocast" ]]; then LAUNCH=("$PYTHON" "$HERE/amp_driver.py"); else LAUNCH=("$LEROBOT_TRAIN"); fi
echo "launcher: ${LAUNCH[*]} (precision $PRECISION)"
TRAIN_EPS=$("$PYTHON" -c "import json,sys;print(json.dumps(json.loads(sys.argv[1])['train']).replace(' ',''))" "$SPLIT_JSON")
HELD_EPS=$("$PYTHON" -c "import json,sys;print(' '.join(map(str,json.loads(sys.argv[1])['held'])))" "$SPLIT_JSON")
FPS=$("$PYTHON" -c "import json,sys;print(json.loads(sys.argv[1])['fps'])" "$SPLIT_JSON")
echo "training episodes : $("$PYTHON" -c "import json,sys;print(len(json.loads(sys.argv[1])['train']))" "$SPLIT_JSON")"
echo "held-out episodes : ${HELD_EPS:-(none)}"
echo "chunk $CHUNK @ ${FPS} fps = $("$PYTHON" -c "print(round($CHUNK/$FPS,3))") s horizon"

if [[ "${SKIP_TRAIN:-0}" != "1" ]]; then
  say "training -> $LR_OUT"
  NEWEST=""
  if [[ "${RESUME:-0}" == "1" ]]; then
    # lerobot zero-pads step directories, so lexical order == numeric order
    for c in "$LR_OUT"/checkpoints/[0-9]*/; do [[ -f "${c}pretrained_model/train_config.json" && -d "${c}training_state" ]] && NEWEST="${c%/}"; done   # complete saves only
  fi
  rc=0
  if [[ -n "$NEWEST" ]]; then
    # An S3 round trip (aws s3 sync back, SageMaker checkpoint restore) turns lerobot's relative `last`
    # symlink into a real directory or drops it; lerobot's update_last_checkpoint() then raises at the first
    # save. Put a proper symlink back and resume from the newest numbered checkpoint, never from `last`.
    LAST="$LR_OUT/checkpoints/last"
    if [[ ! -L "$LAST" ]]; then rm -rf "$LAST"; ln -s "$(basename "$NEWEST")" "$LAST"; fi
    echo "RESUME=1: continuing from $NEWEST"
    RM=(--mode resume); [[ -n "${EXTEND_STEPS:-}" ]] && RM=(--mode extend --steps "$EXTEND_STEPS")
    # capture first, then split: `mapfile < <(cmd) || ...` would ignore cmd's exit status and run lerobot-train with no flags
    RENDERED=$("$PYTHON" "$HERE/render_flags.py" --policy "$POLICY" "${RM[@]}" \
      --config-path "$NEWEST/pretrained_model/train_config.json" --overrides="${OVERRIDES:-}") || { echo "FATAL: render_flags failed"; exit 2; }
    [[ -n "$RENDERED" ]] || { echo "FATAL: render_flags produced no flags"; exit 2; }
    mapfile -t FLAGS <<< "$RENDERED"
    [[ "${RENDER_ONLY:-0}" == "1" ]] && { printf '%s\n' "${FLAGS[@]}"; exit 0; }
    set -x
    "${LAUNCH[@]}" "${FLAGS[@]}" 2>&1 | tr '\r' '\n' | tee -a "$OUT_DIR/train.log" || rc=${PIPESTATUS[0]}
    set +x
  else
    [[ "${RESUME:-0}" == "1" ]] && echo "RESUME=1 but no checkpoint under $LR_OUT/checkpoints -- starting fresh"
    rm -rf "$LR_OUT" "$OUT_DIR/train.log"
    # every ${VAR} the registry argv may reference (policies/<POLICY>.yaml train.argv)
    export REPO_ID DS_ROOT TRAIN_EPS CHUNK DEVICE BATCH STEPS SAVE_FREQ LOG_FREQ NUM_WORKERS SEED LR_OUT
    export JOB_NAME="$(basename "$OUT_DIR")" VIDEO_BACKEND="${VIDEO_BACKEND:-}"
    RENDERED=$("$PYTHON" "$HERE/render_flags.py" --policy "$POLICY" --mode fresh --overrides="${OVERRIDES:-}") || { echo "FATAL: render_flags failed"; exit 2; }
    [[ -n "$RENDERED" ]] || { echo "FATAL: render_flags produced no flags"; exit 2; }
    mapfile -t FLAGS <<< "$RENDERED"
    [[ "${RENDER_ONLY:-0}" == "1" ]] && { printf '%s\n' "${FLAGS[@]}"; exit 0; }
    set -x
    # tr: lerobot's progress bar writes carriage returns even with TQDM_DISABLE (it passes an explicit disable flag);
    # one record per line keeps losscurve.txt and the live tail honest. PIPESTATUS[0] is still lerobot-train's exit code.
    "${LAUNCH[@]}" "${FLAGS[@]}" 2>&1 | tr '\r' '\n' | tee "$OUT_DIR/train.log" || rc=${PIPESTATUS[0]}
    set +x
  fi
  echo "$rc" > "$OUT_DIR/exitcode"
  grep -oE "step:[0-9.]+K? smpl.*" "$OUT_DIR/train.log" > "$OUT_DIR/losscurve.txt" || true
  [[ "$rc" == "0" ]] || { echo "FATAL: lerobot-train exited $rc (see $OUT_DIR/train.log)"; exit "$rc"; }
fi

if [[ -z "$HELD_EPS" ]]; then
  say "no held-out episodes -> evaluation skipped"
elif [[ -z "$EVAL_CKPTS" ]]; then
  say "EVAL_CKPTS empty -> evaluation skipped"
else
  say "held-out evaluation (stride $EVAL_STRIDE) on episodes: $HELD_EPS"
  mkdir -p "$OUT_DIR/eval"
  for CK in $EVAL_CKPTS; do
    CKPT="$LR_OUT/checkpoints/$CK/pretrained_model"
    [[ -d "$CKPT" ]] || { echo "SKIP $CK: $CKPT missing"; continue; }
    erc=0
    "$PYTHON" "$HERE/eval_openloop.py" --ckpt "$CKPT" --policy "${EVAL_POLICY:-$POLICY}" --repo-id "$REPO_ID" --dataset-root "$DS_ROOT" \
      --episodes $HELD_EPS --chunk "$CHUNK" --stride "$EVAL_STRIDE" --device "$DEVICE" --n-samples "${EVAL_SAMPLES:-1}" ${EVAL_EXTRA[@]+"${EVAL_EXTRA[@]}"} \
      --out "$OUT_DIR/eval/eval_$CK.json" --save-preds "$OUT_DIR/eval/eval_${CK}_preds.npz" \
      2>&1 | grep -vE '^  [0-9]+/[0-9]+$' || erc=${PIPESTATUS[0]}
    if [[ "$erc" != "0" ]]; then echo "WARNING: evaluation of $CK failed (rc=$erc); the checkpoint itself is fine"; continue; fi
    "$PYTHON" - "$OUT_DIR/eval/eval_$CK.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1])); a = d["mae"]["all"]; g = d["gripper_timing"]
def line(label, x):
    if x["model"] is None: return
    ratio = f"({x['model']/x['hold']:.3f}x hold)" if x["hold"] else ""
    print(f"  {label:8s} MAE {x['model']:.5f} rad  vs hold {x['hold'] if x['hold'] is not None else float('nan'):.5f}  {ratio}")
line("joints", a["all_joints"]); line("grippers", a["all_grippers"])
if g["total"]: print(f"  gripper timing {g['hits']}/{g['total']} within +/-0.5 s")
print("  (<1.0x hold = better than predicting no motion)")
PY
  done
fi
say "done: $OUT_DIR"
