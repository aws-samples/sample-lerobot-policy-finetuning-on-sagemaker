#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# Downscale a LeRobot v3.0 dataset's videos to a target height (default 360, aspect preserved, even width),
# verify frame-count parity per video, and patch meta/info.json to the new resolution.
# 16:9 sources become 640x360 (the validated configuration). Other aspect ratios are downscaled with their own width
# (a 4:3 camera becomes 480x360) and reported by downscale_plan.py; any width is accepted downstream.
#
#   ./prepare_dataset_360.sh <src_dataset_dir> <dst_dataset_dir>     (relative or absolute paths)
#   TARGET_H=360   target height
#   JOBS=<n>       parallel ffmpeg processes (default: nproc)
#   FORCE_RES=1    kept for compatibility; no longer needed (non-16:9 sources are accepted with a warning)
set -euo pipefail
SRC_IN="${1:?usage: prepare_dataset_360.sh <src_dataset_dir> <dst_dataset_dir>}"
DST_IN="${2:?usage: prepare_dataset_360.sh <src_dataset_dir> <dst_dataset_dir>}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$(cd "$SRC_IN" && pwd)"
mkdir -p "$DST_IN"; DST="$(cd "$DST_IN" && pwd)"
[[ "$SRC" != "$DST" ]] || { echo "FATAL: src and dst are the same directory"; exit 2; }
command -v ffmpeg >/dev/null && command -v ffprobe >/dev/null || { echo "FATAL: ffmpeg/ffprobe not on PATH"; exit 2; }
TARGET_H="${TARGET_H:-360}"
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 4)}"

# ---- plan the per-camera target size BEFORE the re-encode (fails in seconds, not after minutes of ffmpeg) ----
PLAN="$(python3 "$HERE/downscale_plan.py" "$SRC" --height "$TARGET_H")"
[[ -n "$PLAN" ]] || { echo "FATAL: empty downscale plan"; exit 1; }
width_for() {  # camera key -> target width
  awk -v k="$1" '$1 == k { print $2 }' <<< "$PLAN"
}

# ffmpeg >= 5.1 spells the passthrough flag -fps_mode; 4.x (Ubuntu 22.04) only knows -vsync
if ffmpeg -hide_banner -h full 2>/dev/null | grep -q -- '-fps_mode'; then PASS="-fps_mode passthrough"; else PASS="-vsync passthrough"; fi

cp -r "$SRC/data" "$SRC/meta" "$DST/"

# one file: re-encode at the camera's target width, then verify the packet count matches the source
encode_one() {
  local f="$1" rel cam w a b
  rel="${f#"$SRC"/}"                       # videos/<camera>/chunk-000/file-000.mp4
  cam="$(cut -d/ -f2 <<< "$rel")"
  w="$(width_for "$cam")"
  [[ -n "$w" ]] || { echo "FATAL: no plan for camera $cam ($rel)"; return 1; }
  mkdir -p "$DST/$(dirname "$rel")"
  # shellcheck disable=SC2086  # $PASS is two words on purpose
  ffmpeg -nostdin -y -loglevel error -i "$f" -vf "scale=${w}:${TARGET_H}:flags=bicubic" \
    -c:v libx264 -preset veryfast -crf 18 -g 2 -pix_fmt yuv420p $PASS "$DST/$rel"
  a=$(ffprobe -v error -count_packets -select_streams v:0 -show_entries stream=nb_read_packets -of csv=p=0 "$f")
  b=$(ffprobe -v error -count_packets -select_streams v:0 -show_entries stream=nb_read_packets -of csv=p=0 "$DST/$rel")
  [[ "$a" == "$b" ]] || { echo "FRAME MISMATCH $rel ($a vs $b)"; return 1; }
  echo "ok $rel ($a frames, ${w}x${TARGET_H})"
}
export -f encode_one width_for
export SRC DST PLAN PASS TARGET_H
n=$(find "$SRC/videos" -name '*.mp4' | wc -l)
[[ "$n" -gt 0 ]] || { echo "FATAL: no .mp4 under $SRC/videos"; exit 1; }
find "$SRC/videos" -name '*.mp4' -print0 | sort -z | xargs -0 -P "$JOBS" -I{} bash -c 'encode_one "$1"' _ {}

# ---- patch the recorded resolution per camera, keeping each feature's axis order ----
PLAN="$PLAN" python3 - "$DST" "$TARGET_H" <<'PY'
import json, os, sys
p = f'{sys.argv[1]}/meta/info.json'; d = json.load(open(p)); H = int(sys.argv[2])
widths = {line.split()[0]: int(line.split()[1]) for line in os.environ["PLAN"].splitlines() if line.strip()}
for k, v in d['features'].items():
    if v.get('dtype') not in ('video', 'image'):
        continue
    W = widths[k]
    shape = list(v['shape']); hw_pos = [i for i, x in enumerate(shape) if x != 3]
    shape[hw_pos[0]], shape[hw_pos[1]] = H, W          # height then width, wherever they sit
    v['shape'] = shape
    info = v.setdefault('info', {})
    info['video.height'], info['video.width'] = H, W
    print(f"{k}: shape -> {shape}")
json.dump(d, open(p, 'w'), indent=4)
print("patched", p)
PY
echo "done: $DST ($n videos at height $TARGET_H, $JOBS parallel)"
