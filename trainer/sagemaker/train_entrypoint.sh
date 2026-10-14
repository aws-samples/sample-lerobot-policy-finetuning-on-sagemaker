#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# SageMaker training-job entrypoint: maps hyperparameters.json + the SageMaker paths onto train_policy.sh.
#
#   input channel "train"  -> /opt/ml/input/data/train      LeRobot v3.0 dataset (720x1280 or 360x640)
#   /opt/ml/checkpoints    -> CheckpointConfig.S3Uri         <policy>/train/checkpoints/*, synced continuously (resumable)
#   /opt/ml/model          -> model.tar.gz in S3OutputPath   pretrained_model/ + eval/ + SUMMARY.json (small)
#   /opt/ml/output/data    -> output.tar.gz                  train.log, losscurve.txt, split.json
#
# Hyperparameters (all strings; every one optional):
#   steps batch_size chunk save_freq log_freq num_workers (default nproc-4, floor 2, cap 16) seed heldout episodes repo_id video_backend
#   eval_ckpts eval_stride device downscale (auto|1|0) downscale_height (360) resume (auto|1|0) extend_steps
#   policy (act) overrides accept_licence (0|1) approval_ref smoke (0|1) reload_check (1|0) merge_adapter (1|0; see publish.py) image_digest dataset_fingerprint
#   camera cameras (camera subset for the policy; see train_policy.sh) keep_checkpoints (newest K survive; 0 = keep every one)
#   force_res force_action_eq_state force_small require_flat_names (each 0|1; see check_dataset.py --help)
#   resume=auto (default) continues whenever a checkpoint was restored into /opt/ml/checkpoints -- which is
#   exactly what happens after a managed-spot interruption or with ./finetune.sh --resume-from <job>.
set -euo pipefail
# ML and CODE are overridable so the entrypoint can be exercised outside SageMaker (tests); the defaults are the SageMaker contract.
ML="${ML:-/opt/ml}"
CODE="${CODE:-$ML/code}"
IN="$ML/input/data/train"
HP="$ML/input/config/hyperparameters.json"
mkdir -p "$ML/model" "$ML/output/data" "$ML/checkpoints"
trap 'rc=$?; if [[ $rc -ne 0 && ! -f "$ML/output/failure" ]]; then echo "train_entrypoint failed rc=$rc (see CloudWatch log)" > "$ML/output/failure"; fi' EXIT

hp() { python3 -c "import json,sys,os; d=json.load(open('$HP')) if os.path.exists('$HP') else {}; v=d.get(sys.argv[1], sys.argv[2]); print(v.strip('\"') if isinstance(v,str) else v)" "$1" "${2:-}"; }
export STEPS="$(hp steps 100000)" BATCH="$(hp batch_size 8)" CHUNK="$(hp chunk 24)" SAVE_FREQ="$(hp save_freq 10000)"
export SEED="$(hp seed 1000)"
# num_workers is capped at 16: 188 persistent workers on a 192-vCPU host took 35 min to spawn
export LOG_FREQ="$(hp log_freq 200)" NUM_WORKERS="$(hp num_workers "$(( $(nproc) > 20 ? 16 : ($(nproc) > 4 ? $(nproc) - 4 : 2) ))")"
export HELDOUT="$(hp heldout "")" EPISODES="$(hp episodes "")" EVAL_CKPTS="$(hp eval_ckpts last)" EVAL_STRIDE="$(hp eval_stride 1)"
export DEVICE="$(hp device cuda)"
export POLICY="$(hp policy act)" OVERRIDES="$(hp overrides "")" ACCEPT_LICENCE="$(hp accept_licence 0)" APPROVAL_REF="$(hp approval_ref "")" EXTEND_STEPS="$(hp extend_steps "")"
export CAMERA="$(hp camera "")" CAMERAS="$(hp cameras "")" SMOKE="$(hp smoke 0)" KEEP_CHECKPOINTS="$(hp keep_checkpoints 0)"   # 0 = keep every checkpoint   # SMOKE=1 lets render_flags append the registry's smoke-only argv
export FORCE_RES="$(hp force_res 0)" FORCE_ACTION_EQ_STATE="$(hp force_action_eq_state 0)" FORCE_SMALL="$(hp force_small 0)" REQUIRE_FLAT_NAMES="$(hp require_flat_names 0)"
[[ "$(hp force "")" == "" ]] || { echo "FATAL: hyperparameter 'force' was split into force_res / force_action_eq_state / force_small (each 0|1)"; exit 2; }
VB="$(hp video_backend "${VIDEO_BACKEND:-}")"; [[ -n "$VB" ]] && export VIDEO_BACKEND="$VB"
MA="$(hp merge_adapter "${MERGE_ADAPTER:-}")"; [[ -n "$MA" ]] && export MERGE_ADAPTER="$MA"   # publish.py: merge a LoRA adapter into full weights (registry publish.merge_adapter does the same)
RID="$(hp repo_id "")"; DOWNSCALE="$(hp downscale auto)"; RESUME_HP="$(hp resume auto)"
export TARGET_H="$(hp downscale_height 360)"
export PYTHON="${PYTHON:-/usr/local/bin/python}" TORCH_HOME="${TORCH_HOME:-$CODE/torchcache}"
# base weights staged by pipeline/hub.py arrive as channel "hub" in HF cache layout; point the Hub client at them, offline
if [[ -d "$ML/input/data/hub/hub" ]]; then
  export HF_HOME="$ML/input/data/hub" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  echo "hub channel mounted: HF_HOME=$HF_HOME (offline)"
fi
[[ -x "$PYTHON" ]] || export PYTHON="$(command -v python3)"

echo "=== SageMaker LeRobot training job: policy $POLICY ==="; echo "hyperparameters: $(cat "$HP" 2>/dev/null || echo '{}')"
[[ -f "$IN/meta/info.json" ]] || { echo "FATAL: no LeRobot dataset at $IN (meta/info.json missing). Point the 'train' channel at the dataset root."; exit 3; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "(no GPU visible)"

# Scratch for the 360x640 copy: prefer a sibling of the channel dir (on the job's EBS volume when SageMaker
# mounts /opt/ml/input/data from it), else the container filesystem. df tells you which one you got.
WORK="$ML/input/data/scratch_360"; mkdir -p "$WORK" 2>/dev/null || WORK="$ML/work"; mkdir -p "$WORK"
df -h "$WORK" "$ML/checkpoints" "$ML/model" 2>/dev/null | sed 's/^/  /'

# ---- dataset: downscale to 360x640 unless it already is ----
is_height() {  # 1 if every camera in <dir>/meta/info.json already has the target height (prepare_dataset_360.sh patches
              # info.json LAST, so a half-finished copy still reads as "not done" and gets redone)
  python3 -c "
import json,sys
try: d=json.load(open('$1/meta/info.json'))
except Exception: print('0'); sys.exit()
H=int('$TARGET_H')
s=[v['shape'] for v in d['features'].values() if v.get('dtype') in ('video','image')]
print('1' if s and all([d for d in x if d != 3][0]==H for x in s) else '0')"; }
IS360=$(is_height "$IN")
if [[ "$DOWNSCALE" == "1" || ( "$DOWNSCALE" == "auto" && "$IS360" == "0" ) ]]; then
  export DS_ROOT="$WORK/$(basename "${RID:-dataset}" | tr '/' '_')_360"
  if [[ "$(is_height "$DS_ROOT")" == "1" ]]; then echo "reusing verified height-$TARGET_H copy at $DS_ROOT"; else
    rm -rf "$DS_ROOT"
    echo "=== downscaling $IN -> $DS_ROOT (height $TARGET_H, aspect preserved) ==="
    "$CODE/prepare_dataset_360.sh" "$IN" "$DS_ROOT"
  fi
else
  export DS_ROOT="$IN"
fi
export REPO_ID="${RID:-local/$(basename "$DS_ROOT")}"
export ONE_CAM_DIR="$WORK"   # select_camera.py writes the one-camera view here (scratch volume, not mirrored)

# ---- resume: auto = whenever SageMaker restored a checkpoint (spot relaunch / --resume-from) ----
export OUT_DIR="$ML/checkpoints/$POLICY"
shopt -s nullglob; CKS=("$OUT_DIR"/train/checkpoints/[0-9]*/); shopt -u nullglob; RESTORED=${#CKS[@]}   # (ls | wc under set -e exits when the dir is absent)
case "${RESTORED}_${RESUME_HP}" in
  *_1) export RESUME=1 ;;
  *_0) export RESUME=0 ;;
  *)   if [[ "$RESTORED" -gt 0 ]]; then export RESUME=1; echo "resume=auto: $RESTORED restored checkpoint(s) found -> RESUME=1"; else export RESUME=0; fi ;;
esac

# ---- train (train_policy.sh evaluates too, non-fatally) ----
# SageMaker sends SIGTERM (StopTrainingJob, MaxRuntimeExceeded, Spot reclaim) and waits 120 s. bash is PID 1 here (exec-form
# ENTRYPOINT) and ignores SIGTERM unless a handler is installed, so: trap first, run the trainer in its own process group,
# `wait` in a loop (wait returns when the trap fires), kill the whole group (dataloader workers are spawned), then publish
# the newest checkpoint with status=interrupted so model.tar.gz is never empty.
INTERRUPTED=0
TRAIN_PID=""
on_term() {
  echo "=== SIGTERM received: stopping the trainer and publishing the newest checkpoint ==="
  INTERRUPTED=1
  [[ -n "${TRAIN_PID:-}" ]] && kill -TERM -- "-$TRAIN_PID" 2>/dev/null || true
  for _ in $(seq 1 30); do kill -0 "$TRAIN_PID" 2>/dev/null || break; sleep 2; done
  kill -KILL -- "-$TRAIN_PID" 2>/dev/null || true
}
trap on_term TERM INT
setsid "$CODE/train_policy.sh" &
TRAIN_PID=$!
PRUNE_PID=""
if [[ "${KEEP_CHECKPOINTS:-0}" -gt 0 ]]; then   # compute.keep_checkpoints: newest K complete checkpoints survive (S3 mirror follows the deletes)
  ( while kill -0 "$TRAIN_PID" 2>/dev/null; do sleep 120; python3 "$CODE/prune_checkpoints.py" --root "$OUT_DIR/train/checkpoints" --keep "$KEEP_CHECKPOINTS" || true; done ) &
  PRUNE_PID=$!
fi
TRAIN_RC=0
# `|| TRAIN_RC=$?`: under set -e a bare `wait` interrupted by the trap (status 143) would abort the script before publishing
while kill -0 "$TRAIN_PID" 2>/dev/null; do TRAIN_RC=0; wait "$TRAIN_PID" 2>/dev/null || TRAIN_RC=$?; done
trap - TERM INT
[[ -n "$PRUNE_PID" ]] && { kill "$PRUNE_PID" 2>/dev/null || true; python3 "$CODE/prune_checkpoints.py" --root "$OUT_DIR/train/checkpoints" --keep "$KEEP_CHECKPOINTS" || true; }
if [[ "$INTERRUPTED" == "1" ]]; then
  export IMAGE_DIGEST="$(hp image_digest "")" DATASET_FINGERPRINT="$(hp dataset_fingerprint "")"
  python3 "$CODE/publish.py" --out-dir "$OUT_DIR" --model-dir "$ML/model" --output-data "$ML/output/data" --policy "$POLICY" --status interrupted || true
  echo "interrupted by SIGTERM after publishing the newest checkpoint" > "$ML/output/failure"
  exit 143
fi
[[ "$TRAIN_RC" == "0" ]] || { echo "train_policy.sh exited $TRAIN_RC"; exit "$TRAIN_RC"; }

# ---- smoke-only checks (hp smoke=1): weights moved vs fresh init, zero bf16 trainable params, loaded keys ----
if [[ "$(hp smoke 0)" == "1" ]]; then
  CK=$(ls -d "$OUT_DIR"/train/checkpoints/[0-9]*/pretrained_model 2>/dev/null | tail -n 1)
  [[ -n "$CK" ]] && python3 "$CODE/smoke_checks.py" --ckpt "$CK" --policy "$(python3 "$CODE/render_flags.py" --policy "$POLICY" --print lerobot_type)" --device "$(python3 "$CODE/render_flags.py" --policy "$POLICY" --print smoke.checks_device 2>/dev/null || echo "$DEVICE")" --out "$OUT_DIR/smoke_checks.json" || echo "WARNING: smoke_checks failed"
fi

# ---- publish: newest checkpoint + eval + logs + licence notice + SUMMARY.json (publish.py; also used by the SIGTERM path) ----
export IMAGE_DIGEST="$(hp image_digest "")" DATASET_FINGERPRINT="$(hp dataset_fingerprint "")"
python3 "$CODE/publish.py" --out-dir "$OUT_DIR" --model-dir "$ML/model" --output-data "$ML/output/data" --policy "$POLICY" --status completed
# ---- artefact proof: the published checkpoint loads offline with stock lerobot and reproduces the eval's first prediction ----
# (reload_check.py prefers pretrained_model_merged/ when publish.py merged a LoRA adapter, and then ignores --peft-base)
if [[ "$(hp reload_check 1)" == "1" && -d "$ML/model/pretrained_model" ]]; then
  HELD=$(python3 -c "import json,sys; print(' '.join(map(str, json.load(open(sys.argv[1]))['held'])))" "$OUT_DIR/split.json" 2>/dev/null || true)
  if [[ -n "$HELD" ]]; then
    # shellcheck disable=SC2086
    [[ -f "$OUT_DIR/dataset_root.txt" ]] && DS_ROOT="$(cat "$OUT_DIR/dataset_root.txt")"   # single-camera policies trained on a derived view
    HQ=(); [[ "$(python3 "$CODE/render_flags.py" --policy "$POLICY" --print eval.history_via_queues 2>/dev/null || echo False)" == "True" ]] && HQ+=(--history-via-queues)
    [[ "$(python3 "$CODE/render_flags.py" --policy "$POLICY" --print eval.chunk_first 2>/dev/null || echo False)" == "True" ]] && HQ+=(--chunk-first)
    [[ "$(python3 "$CODE/render_flags.py" --policy "$POLICY" --print eval.no_history 2>/dev/null || echo False)" == "True" ]] && HQ+=(--no-history)
    QIK=$(python3 "$CODE/render_flags.py" --policy "$POLICY" --print eval.queue_image_key 2>/dev/null || true); [[ -n "$QIK" && "$QIK" != "None" ]] && HQ+=(--queue-image-key "$QIK")
    TOL=$(python3 "$CODE/render_flags.py" --policy "$POLICY" --print eval.tolerance_s 2>/dev/null || true); [[ -n "$TOL" && "$TOL" != "None" ]] && HQ+=(--tolerance-s "$TOL")
    [[ "$(python3 "$CODE/render_flags.py" --policy "$POLICY" --print eval.reset_per_call 2>/dev/null || echo False)" == "True" ]] && HQ+=(--reset-per-call)
    PB=$(python3 "$CODE/render_flags.py" --policy "$POLICY" --print base_checkpoint 2>/dev/null || true); [[ -n "$PB" && "$PB" != "None" ]] && HQ+=(--peft-base "$PB")
    NS=$(python3 -c "import yaml,sys; print((yaml.safe_load(open(sys.argv[1])).get('eval') or {}).get('n_samples', 1))" "$CODE/policies/$POLICY.yaml" 2>/dev/null || echo 1)
    # eval.stochastic: per-policy opt-in for a recipe whose output genuinely differs between calls on one checkpoint (fresh
    # noise that no seed fixes); the reload check then compares shapes and reports the diff without judging it. No shipped
    # recipe sets it: a reload spread once blamed on sampling was a recipe defect, since fixed (docs/TRAPS.md)
    SST=$(python3 -c "import yaml,sys; print((yaml.safe_load(open(sys.argv[1])).get('eval') or {}).get('stochastic', False))" "$CODE/policies/$POLICY.yaml" 2>/dev/null || echo False)
    STOCH=(); [[ "$NS" -gt 1 || "$SST" == "True" ]] && STOCH=(--stochastic)
    # shellcheck disable=SC2086
    python3 "$CODE/reload_check.py" --model-dir "$ML/model" --dataset-root "$DS_ROOT" --episodes $HELD --repo-id "$REPO_ID" ${STOCH[@]+"${STOCH[@]}"} ${HQ[@]+"${HQ[@]}"} \
      --policy "$(python3 "$CODE/render_flags.py" --policy "$POLICY" --print lerobot_type)" --device "$DEVICE" --out "$OUT_DIR/reload_check.json" || echo "WARNING: reload_check failed to run"
    python3 "$CODE/summary.py" --out-dir "$OUT_DIR" --model-dir "$ML/model" --policy "$POLICY" --status completed --write >/dev/null
  fi
fi
cat "$ML/model/SUMMARY.json"
echo "=== done ==="
