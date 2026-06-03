#!/usr/bin/env bash
set -euo pipefail

# Batch pipeline: segment ALL images first (model loaded once), then run each
# downstream stage once over all images via --start_index/--end_index so every
# model is loaded a single time instead of once per image.
#
# Usage:
#   bash run_batch_full_pipeline.sh /abs/img1.jpg /abs/img2.jpg ...
#   bash run_batch_full_pipeline.sh /abs/dir_of_images
#
# Optional env (same meaning as the single-image pipeline):
#   GPU_IDX=0 OBJ_REC=amodal3r MIN_MASK_AREA=800 USE_YOLO_SEG=1 \
#   YOLO_SEG_MODEL=yoloe-26l-seg.pt KEEP_LABELS=...

if [[ $# -lt 1 ]]; then
  echo "Usage: bash run_batch_full_pipeline.sh /abs/img1.jpg [/abs/img2.jpg ...] | /abs/dir"
  exit 1
fi

GPU_IDX="${GPU_IDX:-0}"
OBJ_REC="${OBJ_REC:-amodal3r}"
MIN_MASK_AREA="${MIN_MASK_AREA:-800}"
KEEP_LABELS="${KEEP_LABELS:-}"
USE_YOLO_SEG="${USE_YOLO_SEG:-0}"
YOLO_SEG_MODEL="${YOLO_SEG_MODEL:-yoloe-26l-seg.pt}"
YOLO_CONF="${YOLO_CONF:-0.45}"
YOLO_IOU="${YOLO_IOU:-0.55}"
YOLO_MAX_DET="${YOLO_MAX_DET:-300}"
YOLO_CLASSES="${YOLO_CLASSES:-}"
YOLO_CLASS_PRESET="${YOLO_CLASS_PRESET:-indoor}"

INDOOR_CLASSES_DEFAULT="chair,table,sofa,bed,desk,mattress,cabinet,shelf,drawer,tv,monitor,refrigerator,microwave,washing machine,oven,bench,couch,bookcase,fan,storage_box,box,closet,air conditioner,cooker,wardrobe,dresser,pantry shelf,piano,coffee table,low table,television, furniture,"
export INDOOR_CLASSES_DEFAULT MIN_MASK_AREA KEEP_LABELS USE_YOLO_SEG YOLO_SEG_MODEL \
       YOLO_CONF YOLO_IOU YOLO_MAX_DET YOLO_CLASSES YOLO_CLASS_PRESET

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- collect image paths (expand a single directory argument) ----
IMAGES=()
if [[ $# -eq 1 && -d "$1" ]]; then
  while IFS= read -r -d '' f; do IMAGES+=("$(realpath "$f")"); done \
    < <(find "$1" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) -print0 | sort -z)
else
  for arg in "$@"; do
    if [[ ! -f "$arg" ]]; then echo "Image not found: $arg"; exit 1; fi
    IMAGES+=("$(realpath "$arg")")
  done
fi

if [[ ${#IMAGES[@]} -eq 0 ]]; then
  echo "No images to process."
  exit 1
fi
echo "Batch: ${#IMAGES[@]} image(s)"

RESULT_ROOT="$REPO/experimental_results/single"
LOG_DIR="$RESULT_ROOT/val/_batch_logs"
mkdir -p "$LOG_DIR"
TIMING_FILE="$LOG_DIR/timing.txt"
> "$TIMING_FILE"

run_logged() {
  local label="$1"; shift
  local logfile="$LOG_DIR/${label}.log"
  local t_start=$SECONDS
  echo "[$(date +%H:%M:%S)] START $label"
  if "$@" > "$logfile" 2>&1; then
    local elapsed=$(( SECONDS - t_start ))
    echo "[$(date +%H:%M:%S)] DONE  $label (${elapsed}s)"
    echo "${label}=${elapsed}" >> "$TIMING_FILE"
  else
    local rc=$?
    echo "[$(date +%H:%M:%S)] FAIL  $label (exit $rc). See $logfile"
    echo "${label}=FAILED" >> "$TIMING_FILE"
    return $rc
  fi
}

cd "$REPO/src"
PIPELINE_START=$SECONDS

# ============================================================
# [1] Batch segmentation (one process, model loaded once)
# ============================================================
echo "[1/7] Batch segmentation over ${#IMAGES[@]} images..."
SEG_LOG="$LOG_DIR/1_segmentation.log"
T1_START=$SECONDS
python batch_scripts/segmentation_batch.py \
  --images "${IMAGES[@]}" \
  --save_dir ../experimental_results/single \
  --split val 2>&1 | tee "$SEG_LOG"
echo "1_segmentation=$(( SECONDS - T1_START ))" >> "$TIMING_FILE"

N=$(grep -oP 'BATCH_NUM_IMAGES=\K[0-9]+' "$SEG_LOG" | tail -1)
if [[ -z "${N:-}" || "$N" -eq 0 ]]; then
  echo "Segmentation produced no images. Aborting."
  exit 1
fi
echo "Segmented $N image(s)."

# ============================================================
# [2-5] Parallel: depth  ||  crop -> (completion || elevation)
#       each stage runs ONCE over all N images
# ============================================================
echo "=== Parallel: [2] depth || [3->4,5] crop->completion,elevation (N=$N) ==="
FAIL_PAR=0

run_logged "2_depth" \
  python batch_scripts/depth.py --gpu_idx "$GPU_IDX" \
    --start_index 0 --end_index "$N" --split val \
    --save_dir ../experimental_results/single &
PID_DEPTH=$!

(
  run_logged "3_cropping" \
    python batch_scripts/get_crops_enhanced.py --gpu_idx "$GPU_IDX" \
      --start_index 0 --end_index "$N" --split val \
      --save_dir ../experimental_results/single || exit 1

  run_logged "4_completion" \
    python batch_scripts/completion.py --gpu_idx "$GPU_IDX" \
      --start_index 0 --end_index "$N" --split val \
      --save_dir ../experimental_results/single &
  PID_COMP=$!

  run_logged "5_elevation" \
    python batch_scripts/elevation.py --gpu_idx "$GPU_IDX" \
      --start_index 0 --end_index "$N" --split val \
      --save_dir ../experimental_results/single &
  PID_ELEV=$!

  INNER_FAIL=0
  wait $PID_COMP || INNER_FAIL=1
  wait $PID_ELEV || INNER_FAIL=1
  exit $INNER_FAIL
) &
PID_BRANCH_B=$!

wait $PID_DEPTH    || FAIL_PAR=1
wait $PID_BRANCH_B || FAIL_PAR=1
if [[ $FAIL_PAR -ne 0 ]]; then
  echo "ERROR: Parallel block had failures. Check logs in $LOG_DIR"
  exit 1
fi

# ============================================================
# [6] 3D reconstruction (once over all N)
# ============================================================
echo "[6/7] 3D reconstruction (N=$N)..."
export CC="$(which gcc)"
export CXX="$(which g++)"
run_logged "6_reconstruction" \
  python batch_scripts/reconstruction.py --gpu_idx "$GPU_IDX" \
    --start_index 0 --end_index "$N" --split val \
    --save_dir ../experimental_results/single --obj_rec "$OBJ_REC"

# ============================================================
# [7] Scene layout alignment (once over all N)
# ============================================================
echo "[7/7] Scene layout alignment (N=$N)..."
run_logged "7_alignment" \
  python batch_scripts/whole.py --gpu_idx "$GPU_IDX" \
    --start_index 0 --end_index "$N" --split val \
    --save_dir ../experimental_results/single

echo "total=$(( SECONDS - PIPELINE_START ))" >> "$TIMING_FILE"
echo "========================================================="
cat "$TIMING_FILE"
echo "========================================================="

# ============================================================
# Per-scene volume
# ============================================================
echo "=== 부피 계산 ==="
for img in "${IMAGES[@]}"; do
  name="$(basename "$img")"
  stem="${name%.*}"
  scene="${stem//\//_}"; scene="${scene//-/_}"
  bbox="$RESULT_ROOT/val/$scene/3dbbox.json"
  echo "--- $scene ---"
  if [[ -f "$bbox" ]]; then
    python3 "$REPO/src/calc_volume.py" "$bbox" || true
  else
    echo "감지된 가구가 없어 부피 계산을 건너뜁니다."
  fi
done
