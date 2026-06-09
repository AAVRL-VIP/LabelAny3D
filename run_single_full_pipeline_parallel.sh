#!/usr/bin/env bash
set -euo pipefail

# Parallelized single-image pipeline:
#
# Sequential: [1] segmentation
# Parallel-1: [2] depth estimation  ||  [3] object cropping
# Parallel-2: [4] amodal completion ||  [5] elevation estimation (uses _reproj.png fallback)
# Sequential: [6] 3D reconstruction
# Sequential: [7] scene layout alignment
#
# Segmentation is handled exclusively by SAM3.
#
# Usage:
#   bash run_single_full_pipeline_parallel.sh /abs/path/image.jpg
# Optional env:
#   GPU_IDX=0 OBJ_REC=amodal3r MIN_MASK_AREA=800 SAM3_PROMPTS=... SAM3_CONF=0.5
#   SAM_PYTHON=/opt/conda/envs/sam/bin/python

if [[ $# -lt 1 ]]; then
  echo "Usage: bash run_single_full_pipeline_parallel.sh /abs/path/image.jpg"
  exit 1
fi

IMAGE_PATH="$1"
GPU_IDX="${GPU_IDX:-0}"
OBJ_REC="${OBJ_REC:-amodal3r}"
MIN_MASK_AREA="${MIN_MASK_AREA:-800}"
SKIP_SEG="${SKIP_SEG:-0}"
KEEP_LABELS="${KEEP_LABELS:-}"
SAM_PYTHON="${SAM_PYTHON:-/opt/conda/envs/sam/bin/python}"

# ============================================================
# Single source of truth: indoor furniture keyword list.
# Used as the default SAM3 prompt set (SAM3_PROMPTS).
# ============================================================
INDOOR_CLASSES_DEFAULT="chair,table,sofa,bed,desk,cabinet,drawer,tv,monitor,refrigerator,microwave,washing machine,oven,bench,couch,bookcase,closet,air conditioner,cooker,wardrobe,dresser,piano,coffee table,low table,television"
export INDOOR_CLASSES_DEFAULT

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ ! -f "$IMAGE_PATH" ]]; then
  echo "Image not found: $IMAGE_PATH"
  exit 1
fi

IMG_NAME="$(basename "$IMAGE_PATH")"
SCENE_NAME="${IMG_NAME%.*}"

COCO_IMG_DIR="$REPO/dataset/coco/images/val2017"
COCO_ANN_DIR="$REPO/dataset/coco/annotations"
COCO_VAL_JSON="$COCO_ANN_DIR/coconut_val.json"
RESULT_ROOT="$REPO/experimental_results/single"
RESULT_SCENE_DIR="$RESULT_ROOT/val/$SCENE_NAME"
SEG_JSON="${SEG_JSON:-$COCO_VAL_JSON}"

mkdir -p "$COCO_IMG_DIR" "$COCO_ANN_DIR" "$RESULT_SCENE_DIR"

export IMAGE_PATH
export COCO_DST="$COCO_IMG_DIR/$IMG_NAME"
export INPUT_DST="$RESULT_SCENE_DIR/input.png"
export MIN_MASK_AREA
export RESULT_SCENE_DIR
export KEEP_LABELS

python - <<'PY'
from PIL import Image, ImageOps
import os

src = os.environ["IMAGE_PATH"]
coco_dst = os.environ["COCO_DST"]
input_dst = os.environ["INPUT_DST"]

img = Image.open(src)
img = ImageOps.exif_transpose(img).convert("RGB")

# COCO image path: keep original filename/extension
img.save(coco_dst)

# Unified pipeline input: always save normalized PNG
img.save(input_dst)

print(f"Saved normalized image to: {coco_dst}")
print(f"Saved normalized image to: {input_dst}")
print(f"Normalized image size: {img.size}")
PY

cd "$REPO/src"

LOG_DIR="$RESULT_SCENE_DIR/logs"
mkdir -p "$LOG_DIR"

TIMING_FILE="$RESULT_SCENE_DIR/timing.txt"
> "$TIMING_FILE"  # clear

PIPELINE_START=$SECONDS

# Helper: run a command, log output, measure elapsed time, and propagate failures
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
    local elapsed=$(( SECONDS - t_start ))
    echo "[$(date +%H:%M:%S)] FAIL  $label (exit $rc, ${elapsed}s). See $logfile"
    echo "${label}=${elapsed} FAILED" >> "$TIMING_FILE"
    return $rc
  fi
}

# ============================================================
# [1/7] Segmentation (sequential — everything depends on this)
# ============================================================
T1_START=$SECONDS
if [[ "$SKIP_SEG" == "1" ]]; then
  echo "[1/7] Skipping segmentation. Using existing annotation JSON: $SEG_JSON"
  if [[ ! -f "$SEG_JSON" ]]; then
    echo "Annotation JSON not found: $SEG_JSON"
    exit 1
  fi
  if [[ "$SEG_JSON" != "$COCO_VAL_JSON" ]]; then
    cp -f "$SEG_JSON" "$COCO_VAL_JSON"
  fi
else
  echo "[1/7] Generating single-image COCONUT-style annotation via SAM3 (sam env)..."
  SAM3_PROMPTS="${SAM3_PROMPTS:-$INDOOR_CLASSES_DEFAULT}"
  SAM3_CONF="${SAM3_CONF:-0.5}"
  INTERACTIVE_SAM3="${INTERACTIVE_SAM3:-0}"
  SAM3_SCRIPT="$REPO/src/sam3_seg_for_la3d.py"
  if [[ ! -f "$SAM3_SCRIPT" ]]; then
    echo "SAM3 script not found: $SAM3_SCRIPT"
    exit 1
  fi
  env -u PYTHONPATH "$SAM_PYTHON" "$SAM3_SCRIPT" \
    --image "$IMAGE_PATH" \
    --out_json "$COCO_VAL_JSON" \
    --out_seg_dir "$RESULT_SCENE_DIR/segmentation" \
    --prompts "$SAM3_PROMPTS" \
    --keep_labels "$KEEP_LABELS" \
    --min_mask_area "$MIN_MASK_AREA" \
    --confidence "$SAM3_CONF" \
    $( [[ "$INTERACTIVE_SAM3" == "1" ]] && echo "--interactive" )
fi
T1_ELAPSED=$(( SECONDS - T1_START ))
echo "1_segmentation=${T1_ELAPSED}" >> "$TIMING_FILE"
echo "[1/7] segmentation + annotation : ${T1_ELAPSED}s"

# ============================================================
# UNIFIED PARALLEL BLOCK:
#
#   [2] depth ─────────────────────────────────┐
#   [3] cropping → [4] completion              │ all parallel
#                → [5] elevation               │
#   ───────────────────────────────────────────┘ wait all
#
# Step 2 (depth) is fully independent.
# Steps 4, 5 depend on step 3 (crops), so we chain them.
# Step 5 uses _reproj.png fallback (no need for _rgba.png).
# ============================================================
echo "=== Unified Parallel: [2] depth || [3→4,5] crop→completion,elevation ==="
T_PAR_START=$SECONDS
FAIL_PAR=0

# --- Branch A: depth (independent, runs entire duration) ---
run_logged "2_depth" \
  python batch_scripts/depth.py \
    --gpu_idx "$GPU_IDX" \
    --start_index 0 --end_index 1 \
    --split val \
    --save_dir ../experimental_results/single &
PID_DEPTH=$!

# --- Branch B: cropping → then spawn completion + elevation in parallel ---
(
  # Step 3: cropping (must finish before 4 and 5)
  run_logged "3_cropping" \
    python batch_scripts/get_crops_enhanced.py \
      --gpu_idx "$GPU_IDX" \
      --start_index 0 --end_index 1 \
      --split val \
      --save_dir ../experimental_results/single || exit 1

  # Step 4: amodal completion (uses crops)
  run_logged "4_completion" \
    python batch_scripts/completion.py \
      --gpu_idx "$GPU_IDX" \
      --start_index 0 --end_index 1 \
      --split val \
      --save_dir ../experimental_results/single &
  PID_COMP_INNER=$!

  # Step 5: elevation (uses _reproj.png fallback from crops)
  run_logged "5_elevation" \
    python batch_scripts/elevation.py \
      --gpu_idx "$GPU_IDX" \
      --start_index 0 --end_index 1 \
      --split val \
      --save_dir ../experimental_results/single &
  PID_ELEV_INNER=$!

  # Wait for both 4 and 5
  INNER_FAIL=0
  wait $PID_COMP_INNER || INNER_FAIL=1
  wait $PID_ELEV_INNER || INNER_FAIL=1
  exit $INNER_FAIL
) &
PID_BRANCH_B=$!

# --- Wait for all branches ---
wait $PID_DEPTH    || FAIL_PAR=1
wait $PID_BRANCH_B || FAIL_PAR=1

if [[ $FAIL_PAR -ne 0 ]]; then
  echo "ERROR: Parallel block had failures. Check logs in $LOG_DIR"
  exit 1
fi
T_PAR_ELAPSED=$(( SECONDS - T_PAR_START ))
echo "parallel_block_wall=${T_PAR_ELAPSED}" >> "$TIMING_FILE"

# ============================================================
# [6/7] 3D reconstruction (sequential — needs _rgba.png)
# ============================================================
echo "[6/7] 3D reconstruction..."
T6_START=$SECONDS
export CC="$(which gcc)"
export CXX="$(which g++)"
python batch_scripts/reconstruction.py \
  --gpu_idx "$GPU_IDX" \
  --start_index 0 --end_index 1 \
  --split val \
  --save_dir ../experimental_results/single \
  --obj_rec "$OBJ_REC"
T6_ELAPSED=$(( SECONDS - T6_START ))
echo "6_reconstruction=${T6_ELAPSED}" >> "$TIMING_FILE"

# ============================================================
# [7/7] Scene layout alignment (sequential — needs everything)
# ============================================================
echo "[7/7] Scene layout alignment..."
T7_START=$SECONDS
python batch_scripts/whole.py \
  --gpu_idx "$GPU_IDX" \
  --start_index 0 --end_index 1 \
  --split val \
  --save_dir ../experimental_results/single
T7_ELAPSED=$(( SECONDS - T7_START ))
echo "7_alignment=${T7_ELAPSED}" >> "$TIMING_FILE"

PIPELINE_TOTAL=$(( SECONDS - PIPELINE_START ))
echo "total=${PIPELINE_TOTAL}" >> "$TIMING_FILE"

# ============================================================
# Read individual step times from timing file
# ============================================================
get_time() { grep "^${1}=" "$TIMING_FILE" | head -1 | cut -d= -f2 | awk '{print $1}'; }

T_SEG=$(get_time 1_segmentation)
T_DEPTH=$(get_time 2_depth)
T_CROP=$(get_time 3_cropping)
T_COMP=$(get_time 4_completion)
T_ELEV=$(get_time 5_elevation)
T_RECON=$(get_time 6_reconstruction)
T_ALIGN=$(get_time 7_alignment)
T_PAR_WALL=$(get_time parallel_block_wall)

# Compute what sequential would have been
T_SEQ_EST=$(( T_SEG + T_DEPTH + T_CROP + T_COMP + T_ELEV + T_RECON + T_ALIGN ))
T_SAVED=$(( T_SEQ_EST - PIPELINE_TOTAL ))

echo ""
echo "========================================================="
echo "                    TIMING SUMMARY"
echo "========================================================="
printf " %-38s : %6ss\n" "1. segmentation + annotation"        "$T_SEG"
echo "---------------------------------------------------------"
printf " %-38s : %6ss\n" "2. depth estimation"                  "$T_DEPTH"
printf " %-38s : %6ss\n" "3. object cropping"                   "$T_CROP"
printf " %-38s : %6ss\n" "4. amodal completion"                 "$T_COMP"
printf " %-38s : %6ss\n" "5. elevation estimation"              "$T_ELEV"
printf " %-38s : %6ss\n" "   [2 || 3→4,5] parallel wall-clock" "$T_PAR_WALL"
echo "---------------------------------------------------------"
printf " %-38s : %6ss\n" "6. 3D reconstruction"                 "$T_RECON"
printf " %-38s : %6ss\n" "7. scene layout alignment"            "$T_ALIGN"
echo "========================================================="
printf " %-38s : %6ss\n" "TOTAL (parallel)"                     "$PIPELINE_TOTAL"
printf " %-38s : %6ss\n" "TOTAL (if sequential)"                "$T_SEQ_EST"
printf " %-38s : %6ss\n" "TIME SAVED by parallelism"            "$T_SAVED"
echo "========================================================="
echo ""
echo "Timing saved to: $TIMING_FILE"
echo "Result: $RESULT_SCENE_DIR/3dbbox.json"

echo "=== 부피 계산 ==="
if [ -f "$RESULT_SCENE_DIR/3dbbox.json" ]; then
    python3 "$REPO/src/calc_volume.py" "$RESULT_SCENE_DIR/3dbbox.json"
else
    echo "감지된 가구가 없어 부피 계산을 건너뜁니다."
fi
