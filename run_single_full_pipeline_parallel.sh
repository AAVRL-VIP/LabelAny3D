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
# Usage:
#   bash run_single_full_pipeline_parallel.sh /abs/path/image.jpg
# Optional env:
#   GPU_IDX=0 OBJ_REC=trellis MIN_MASK_AREA=6400 USE_YOLO_SEG=1 YOLO_SEG_MODEL=yoloe-26l-seg.pt

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
USE_YOLO_SEG="${USE_YOLO_SEG:-0}"
YOLO_SEG_MODEL="${YOLO_SEG_MODEL:-yoloe-26l-seg.pt}"
YOLO_CONF="${YOLO_CONF:-0.45}"
YOLO_IOU="${YOLO_IOU:-0.55}"
YOLO_MAX_DET="${YOLO_MAX_DET:-300}"
YOLO_CLASSES="${YOLO_CLASSES:-}"
YOLO_CLASS_PRESET="${YOLO_CLASS_PRESET:-indoor}"

# ============================================================
# Single source of truth: indoor furniture keyword list.
# Used by both YOLOE (USE_YOLO_SEG=1) and SAM3 (USE_SAM3=1) paths.
# Edit this list once → both segmenters reflect the change.
# ============================================================
INDOOR_CLASSES_DEFAULT="chair,table,sofa,bed,desk,cabinet,shelf,drawer,tv,monitor,refrigerator,microwave,washing machine,oven,bench,couch,bookcase,storage_box,closet,air conditioner,cooker,wardrobe,dresser,pantry shelf,piano,coffee table,low table"
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
export USE_YOLO_SEG
export YOLO_SEG_MODEL
export YOLO_CONF
export YOLO_IOU
export YOLO_MAX_DET
export YOLO_CLASSES
export YOLO_CLASS_PRESET

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
elif [[ "${USE_SAM3:-0}" == "1" ]]; then
  echo "[1/7] Generating single-image COCONUT-style annotation via SAM3 (sam env)..."
  SAM3_PROMPTS="${SAM3_PROMPTS:-$INDOOR_CLASSES_DEFAULT}"
  SAM3_CONF="${SAM3_CONF:-0.5}"
  INTERACTIVE_SAM3="${INTERACTIVE_SAM3:-0}"
  SAM3_SCRIPT="$REPO/src/sam3_seg_for_la3d.py"
  if [[ ! -f "$SAM3_SCRIPT" ]]; then
    echo "SAM3 script not found: $SAM3_SCRIPT"
    exit 1
  fi
  env -u PYTHONPATH /opt/conda/envs/sam/bin/python "$SAM3_SCRIPT" \
    --image "$IMAGE_PATH" \
    --out_json "$COCO_VAL_JSON" \
    --out_seg_dir "$RESULT_SCENE_DIR/segmentation" \
    --prompts "$SAM3_PROMPTS" \
    --keep_labels "$KEEP_LABELS" \
    --min_mask_area "$MIN_MASK_AREA" \
    --confidence "$SAM3_CONF" \
    $( [[ "$INTERACTIVE_SAM3" == "1" ]] && echo "--interactive" )
else
  echo "[1/7] Generating single-image COCONUT-style annotation..."
  # (segmentation inline script unchanged — keeping it inline for self-containedness)
python - <<'PY'
import json
import os
from pathlib import Path

import cv2
import numpy as np
import rembg
from PIL import Image, ImageOps
from pycocotools import mask as mask_utils

from model_wrappers import run_entityv2, run_clipseg, run_ovsam, run_yolo_seg

import random
import torch

seed = 7
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


repo = Path.cwd().parent
image_path = Path(os.environ["IMAGE_PATH"]).resolve()
img_name = image_path.name
min_mask_area = int(os.environ.get("MIN_MASK_AREA", "800"))
keep_labels_raw = os.environ.get("KEEP_LABELS", "")
keep_labels = {x.strip().lower() for x in keep_labels_raw.split(",") if x.strip()}
use_yolo_seg = os.environ.get("USE_YOLO_SEG", "0") == "1"
yolo_model = os.environ.get("YOLO_SEG_MODEL", "yoloe-26l-seg.pt")
yolo_conf = float(os.environ.get("YOLO_CONF", "0.45"))
yolo_iou = float(os.environ.get("YOLO_IOU", "0.55"))
yolo_max_det = int(os.environ.get("YOLO_MAX_DET", "300"))
yolo_classes_raw = os.environ.get("YOLO_CLASSES", "")
yolo_classes = [x.strip() for x in yolo_classes_raw.split(",") if x.strip()]
yolo_class_preset = os.environ.get("YOLO_CLASS_PRESET", "indoor").strip().lower()

if not yolo_classes and yolo_class_preset == "indoor":
    # Pull single-source list from INDOOR_CLASSES_DEFAULT exported by the shell.
    _indoor = os.environ.get("INDOOR_CLASSES_DEFAULT", "")
    yolo_classes = [x.strip() for x in _indoor.split(",") if x.strip()]
furniture_keywords = {
    # 소파/의자류
    "furniture", "sofa", "couch", "chair", "stool", "bench",
    "ottoman", "futon", "recliner", "loveseat",
    # 테이블/책상류
    "table", "desk",
    # 수납류
    "cabinet", "drawer", "shelf", "bookcase", "wardrobe",
    "dresser", "cupboard", "locker",
    # 침실류
    "bed", "mattress", "crib",
    # 가전류
    "refrigerator", "washer", "dryer", "dishwasher",
    "microwave", "oven", "stove", "television", "monitor",
    "air_conditioner", "vacuum",
    # 기타
    "piano", "ironing_board",
}

result_scene_dir = Path(os.environ["RESULT_SCENE_DIR"]).resolve()
seg_dir = result_scene_dir / "segmentation"
seg_dir.mkdir(parents=True, exist_ok=True)

ann_path = repo / "dataset" / "coco" / "annotations" / "coconut_val.json"
img = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
img_np = np.array(img)
H, W = img_np.shape[:2]

masks = []
labels = []

if use_yolo_seg:
    try:
        masks, labels = run_yolo_seg(
            img_np,
            model_path=yolo_model,
            conf=yolo_conf,
            iou=yolo_iou,
            classes=yolo_classes if yolo_classes else None,
            max_det=yolo_max_det,
        )
        print(f"YOLO segmentation produced {len(masks)} masks (model={yolo_model}, classes={len(yolo_classes) if yolo_classes else 'all'})")
    except Exception as e:
        print(f"YOLO segmentation failed, switching to EntityV2 pipeline: {e}")
        masks = []
        labels = []

if not use_yolo_seg and len(masks) == 0:
    try:
        seg_masks = run_entityv2(img_np, threshold=0.1, max_size=1500)
        if len(seg_masks) > 0:
            fg_idx, _ = run_clipseg(img, seg_masks)
            if len(fg_idx) > 0:
                masks = [seg_masks[i] for i in fg_idx]
                try:
                    labels = run_ovsam(img, np.array(masks).astype(np.uint8) * 255)
                except Exception as e:
                    print(f"OVSAM failed, falling back to generic labels: {e}")
                    labels = ["object"] * len(masks)
    except Exception as e:
        print(f"EntityV2/CLIPSeg pipeline failed, switching to rembg fallback: {e}")
        masks = []
        labels = []

if len(masks) == 0:
    rgba = np.array(rembg.remove(img, rembg.new_session("isnet-general-use")))
    fg = (rgba[..., 3] > 127).astype(np.uint8)
    num_labels, cc = cv2.connectedComponents(fg, connectivity=8)
    comp_masks = []
    for comp_id in range(1, num_labels):
        m = cc == comp_id
        area = int(m.sum())
        if area >= min_mask_area:
            comp_masks.append(m)
    if len(comp_masks) > 0:
        masks = comp_masks
        labels = ["object"] * len(masks)
        print(f"rembg fallback produced {len(masks)} connected components")
    elif fg.sum() >= min_mask_area:
        masks = [fg.astype(bool)]
        labels = ["object"]

annotations = []
categories = []
category_to_id = {}
ann_id = 1
seg_vis = []

if len(labels) != len(masks):
    labels = ["object"] * len(masks)

for i, m in enumerate(masks):
    m = np.asarray(m).astype(np.uint8)
    if m.sum() < min_mask_area:
        continue
    # 경계에 걸친 마스크 제외
    is_truncated = (
        m[0, :].any() or m[-1, :].any() or
        m[:, 0].any() or m[:, -1].any()
    )
    if is_truncated:
        continue
    ys, xs = np.where(m > 0)
    if len(xs) == 0:
        continue
    raw_label = str(labels[i]).strip() if i < len(labels) else "object"
    label = raw_label if raw_label else "object"
    if keep_labels:
        label_l = label.lower()
        matched = False
        for k in keep_labels:
            if k == "furniture":
                if any(w in label_l for w in furniture_keywords):
                    matched = True
                    break
            if k in label_l:
                matched = True
                break
        if not matched:
            continue
    if label not in category_to_id:
        category_id = len(category_to_id) + 1
        category_to_id[label] = category_id
        categories.append({"id": category_id, "name": label, "supercategory": "object"})
    category_id = category_to_id[label]
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    bbox = [float(x1), float(y1), float(x2 - x1 + 1), float(y2 - y1 + 1)]
    area = float(m.sum())
    rle = mask_utils.encode(np.asfortranarray(m))
    counts = rle["counts"]
    if isinstance(counts, (bytes, bytearray)):
        counts = counts.decode("utf-8")
    annotations.append({
        "id": ann_id, "image_id": 1, "category_id": category_id,
        "category_name": label, "bbox": bbox, "area": area, "iscrowd": 0,
        "segmentation": {"size": [H, W], "counts": counts},
    })
    seg_vis.append({
        "ann_id": ann_id, "label": label, "category_id": category_id,
        "bbox": bbox, "area": area, "mask": m.astype(bool),
    })
    ann_id += 1

payload = {
    "images": [{"id": 1, "file_name": img_name, "width": W, "height": H}],
    "annotations": annotations,
    "categories": categories,
}
ann_path.parent.mkdir(parents=True, exist_ok=True)
with open(ann_path, "w") as f:
    json.dump(payload, f)

overlay = img_np.copy().astype(np.uint8)
seg_index = []
for idx, item in enumerate(seg_vis):
    mask = item["mask"]
    color = np.array([(37*(idx+3))%256, (97*(idx+5))%256, (17*(idx+7))%256], dtype=np.uint8)
    mask_u8 = (mask.astype(np.uint8) * 255)
    safe_label = ''.join(c if c.isalnum() or c in ('-','_') else '_' for c in item["label"])
    mask_name = f"mask_{idx:02d}_{safe_label}.png"
    cv2.imwrite(str(seg_dir / mask_name), mask_u8)
    overlay[mask] = (0.55 * overlay[mask] + 0.45 * color).astype(np.uint8)
    x, y, w, h = item["bbox"]
    x, y, w, h = int(x), int(y), int(w), int(h)
    cv2.rectangle(overlay, (x, y), (x+w, y+h), tuple(int(v) for v in color.tolist()), 2)
    cv2.putText(overlay, item["label"], (x, max(20, y-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, tuple(int(v) for v in color.tolist()), 2)
    seg_index.append({
        "ann_id": item["ann_id"], "label": item["label"], "category_id": item["category_id"],
        "bbox": item["bbox"], "area": item["area"], "mask_file": mask_name,
    })
Image.fromarray(overlay).save(seg_dir / "overlay.png")
with open(seg_dir / "labels.json", "w") as f:
    json.dump(seg_index, f, indent=2)

if keep_labels:
    print(f"Label filter KEEP_LABELS: {sorted(keep_labels)}")
print(f"Saved: {ann_path}")
print(f"Segmentation preview: {seg_dir / 'overlay.png'}")
print(f"Num instances: {len(annotations)}")
if len(annotations) == 0:
    if keep_labels:
        print("가구가 감지되지 않았습니다. 소파, 침대, 테이블 등 큰 가구가 잘 보이도록 촬영해주세요.")
        import sys
        sys.exit(0)
    else:
        raise RuntimeError("No valid segmentation instances were produced.")
PY
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
