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
DEPTH_GPU_IDX="${DEPTH_GPU_IDX:-$GPU_IDX}"
CROP_GPU_IDX="${CROP_GPU_IDX:-$GPU_IDX}"
COMPLETION_GPU_IDX="${COMPLETION_GPU_IDX:-$GPU_IDX}"
ELEVATION_GPU_IDX="${ELEVATION_GPU_IDX:-$GPU_IDX}"
RECON_GPU_IDX="${RECON_GPU_IDX:-$GPU_IDX}"
ALIGN_GPU_IDX="${ALIGN_GPU_IDX:-$GPU_IDX}"
OBJ_REC="${OBJ_REC:-amodal3r}"
MIN_MASK_AREA="${MIN_MASK_AREA:-6000}"
SKIP_SEG="${SKIP_SEG:-0}"
KEEP_LABELS="${KEEP_LABELS:-}"
USE_YOLO_SEG="${USE_YOLO_SEG:-0}"
YOLO_SEG_MODEL="${YOLO_SEG_MODEL:-yoloe-26l-seg.pt}"
YOLO_CONF="${YOLO_CONF:-0.45}"
YOLO_IOU="${YOLO_IOU:-0.55}"
YOLO_MAX_DET="${YOLO_MAX_DET:-300}"
YOLO_CLASSES="${YOLO_CLASSES:-}"
YOLO_CLASS_PRESET="${YOLO_CLASS_PRESET:-indoor}"
DEPTH_MAX_VALID="${DEPTH_MAX_VALID:-10.0}"
DEPTH_FIT_INTERCEPT="${DEPTH_FIT_INTERCEPT:-1}"
DEPTH_MATCH_PERCENTILE="${DEPTH_MATCH_PERCENTILE:-20}"
DEPTH_MATCH_ERODE="${DEPTH_MATCH_ERODE:-5}"
PLANAR_DEPTH_FLATTEN="${PLANAR_DEPTH_FLATTEN:-1}"
PLANAR_DEPTH_STD_RATIO="${PLANAR_DEPTH_STD_RATIO:-0.05}"
BBOX_EXTENT_PERCENTILE="${BBOX_EXTENT_PERCENTILE:-1.5}"
THIN_OBJECT_DEPTH_M="${THIN_OBJECT_DEPTH_M:-0.05}"
THIN_OBJECT_LABELS="${THIN_OBJECT_LABELS:-tv,television,monitor}"
MASK_MERGE_FRAGMENTS="${MASK_MERGE_FRAGMENTS:-1}"
MASK_MERGE_GAP_RATIO="${MASK_MERGE_GAP_RATIO:-0.08}"
MASK_REMOVE_OVERLAPS="${MASK_REMOVE_OVERLAPS:-1}"
FORCE_DEPTH="${FORCE_DEPTH:-0}"
FORCE_ALIGNMENT="${FORCE_ALIGNMENT:-0}"
FORCE_RECOMPUTE="${FORCE_RECOMPUTE:-0}"
# SEG_BACKEND: yolo | gsam2 | entityv2  (empty = auto from USE_YOLO_SEG)
SEG_BACKEND="${SEG_BACKEND:-}"
GSAM2_PROMPT="${GSAM2_PROMPT:-sofa . couch . chair . dining chair . stool . bench . ottoman . futon . recliner . loveseat . table . dining table . desk . tv stand . kitchen island . cabinet . kitchen cabinet . drawer . drawer unit . shelf . bookcase . bookshelf . wardrobe . closet . dresser . cupboard . locker . shoe rack . storage box . bed . crib . refrigerator . kimchi refrigerator . freezer . washing machine . dryer . dishwasher . microwave . rice cooker . oven . stove . television . tv . monitor . tv on stand . air conditioner . air purifier . vacuum . piano . bicycle}"
GSAM2_BOX_THRESH="${GSAM2_BOX_THRESH:-0.4}"
GSAM2_TEXT_THRESH="${GSAM2_TEXT_THRESH:-0.3}"
GSAM2_NMS_THRESH="${GSAM2_NMS_THRESH:-0.4}"
GSAM2_GDINO_MODEL="${GSAM2_GDINO_MODEL:-auto}"  # auto | swinb | swint
GSAM2_GDINO_CONFIG="${GSAM2_GDINO_CONFIG:-}"
GSAM2_GDINO_CHECKPOINT="${GSAM2_GDINO_CHECKPOINT:-}"
GSAM2_SAM2_CONFIG="${GSAM2_SAM2_CONFIG:-}"
GSAM2_SAM2_CHECKPOINT="${GSAM2_SAM2_CHECKPOINT:-}"
# Optional label overrides for Grounded-SAM/other open-vocab outputs.
# Formats:
#   LABEL_CANONICAL_MAP='{"tv_television_monitor":"television","couch":"sofa"}'
#   LABEL_CANONICAL_MAP='tv_television_monitor:television,couch:sofa'
LABEL_CANONICAL_MAP="${LABEL_CANONICAL_MAP:-}"

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

TORCH_LIB="$(python -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))" 2>/dev/null || echo "")"
if [[ -n "$TORCH_LIB" ]]; then
  export LD_LIBRARY_PATH="${TORCH_LIB}:${LD_LIBRARY_PATH:-}"
fi

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
export DEPTH_MAX_VALID
export DEPTH_FIT_INTERCEPT
export DEPTH_MATCH_PERCENTILE
export DEPTH_MATCH_ERODE
export PLANAR_DEPTH_FLATTEN
export PLANAR_DEPTH_STD_RATIO
export BBOX_EXTENT_PERCENTILE
export THIN_OBJECT_DEPTH_M
export THIN_OBJECT_LABELS
export MASK_MERGE_FRAGMENTS
export MASK_MERGE_GAP_RATIO
export MASK_REMOVE_OVERLAPS
export FORCE_DEPTH
export FORCE_ALIGNMENT
export FORCE_RECOMPUTE
export SEG_BACKEND
export GSAM2_PROMPT
export GSAM2_BOX_THRESH
export GSAM2_TEXT_THRESH
export GSAM2_NMS_THRESH
export GSAM2_GDINO_MODEL
export GSAM2_GDINO_CONFIG
export GSAM2_GDINO_CHECKPOINT
export GSAM2_SAM2_CONFIG
export GSAM2_SAM2_CHECKPOINT
export LABEL_CANONICAL_MAP

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

if [[ "$FORCE_RECOMPUTE" == "1" || "$FORCE_DEPTH" == "1" ]]; then
  echo "[cache] Removing cached depth outputs for recomputation"
  rm -f "$RESULT_SCENE_DIR/depth_map.npy" \
        "$RESULT_SCENE_DIR/cam_params.json" \
        "$RESULT_SCENE_DIR/depth_scene.ply" \
        "$RESULT_SCENE_DIR/depth_scene_no_edge.ply"
fi

if [[ "$FORCE_RECOMPUTE" == "1" || "$FORCE_ALIGNMENT" == "1" ]]; then
  echo "[cache] Removing cached alignment outputs for recomputation"
  rm -f "$RESULT_SCENE_DIR/3dbbox.json" \
        "$RESULT_SCENE_DIR/3dbbox_ground.json" \
        "$RESULT_SCENE_DIR/vis_3dbox.png" \
        "$RESULT_SCENE_DIR/vis_3dbox_no_ground.png" \
        "$RESULT_SCENE_DIR/reconstruction/full_scene.glb" \
        "$RESULT_SCENE_DIR/reconstruction/"*_canonical_upright.npy \
        "$RESULT_SCENE_DIR/reconstruction/"*.glb
fi

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

from model_wrappers import run_entityv2, run_clipseg, run_ovsam, run_yolo_seg, run_grounded_sam2

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
min_mask_area = int(os.environ.get("MIN_MASK_AREA", "6400"))
keep_labels_raw = os.environ.get("KEEP_LABELS", "")
keep_labels = {x.strip().lower() for x in keep_labels_raw.split(",") if x.strip()}
use_yolo_seg = os.environ.get("USE_YOLO_SEG", "0") == "1"
seg_backend = os.environ.get("SEG_BACKEND", "").strip().lower()
if not seg_backend:
    seg_backend = "yolo" if use_yolo_seg else "entityv2"
gsam2_prompt     = os.environ.get("GSAM2_PROMPT", "sofa . couch . chair . dining chair . stool . bench . ottoman . futon . recliner . loveseat . table . dining table . desk . tv stand . kitchen island . cabinet . kitchen cabinet . sideboard . drawer . drawer unit . shelf . bookcase . bookshelf . wardrobe . closet . dresser . cupboard . locker . shoe rack . storage box . bed . crib . refrigerator . kimchi refrigerator . freezer . washing machine . dryer . dishwasher . microwave . rice cooker . oven . stove . television . tv . monitor . tv on stand . air conditioner . air purifier . vacuum . piano . bicycle")
gsam2_box_thresh = float(os.environ.get("GSAM2_BOX_THRESH", "0.35"))
gsam2_txt_thresh = float(os.environ.get("GSAM2_TEXT_THRESH", "0.25"))
gsam2_nms_thresh = float(os.environ.get("GSAM2_NMS_THRESH", "0.5"))
gsam2_gdino_model = os.environ.get("GSAM2_GDINO_MODEL", "auto").strip().lower()
gsam2_gdino_config = os.environ.get("GSAM2_GDINO_CONFIG", "").strip() or None
gsam2_gdino_checkpoint = os.environ.get("GSAM2_GDINO_CHECKPOINT", "").strip() or None
gsam2_sam2_config = os.environ.get("GSAM2_SAM2_CONFIG", "").strip() or None
gsam2_sam2_checkpoint = os.environ.get("GSAM2_SAM2_CHECKPOINT", "").strip() or None
yolo_model = os.environ.get("YOLO_SEG_MODEL", "yoloe-26l-seg.pt")
yolo_conf = float(os.environ.get("YOLO_CONF", "0.45"))
yolo_iou = float(os.environ.get("YOLO_IOU", "0.55"))
yolo_max_det = int(os.environ.get("YOLO_MAX_DET", "300"))
yolo_classes_raw = os.environ.get("YOLO_CLASSES", "")
yolo_classes = [x.strip() for x in yolo_classes_raw.split(",") if x.strip()]
yolo_class_preset = os.environ.get("YOLO_CLASS_PRESET", "indoor").strip().lower()

if not yolo_classes and yolo_class_preset == "indoor":
    yolo_classes = [
        "chair", "table", "sofa", "bed", "desk",
        "cabinet", "shelf", "drawer", "tv", "monitor",
        "refrigerator", "microwave", "washing machine",
        "oven", "bench",
        "blanket", "fan", "storage_box", "box",
        "air conditioner", "cooker",
    ]
furniture_keywords = {
    # 소파/의자류
    "furniture", "sofa", "couch", "chair", "stool", "bench",
    "ottoman", "futon", "recliner", "loveseat",
    # 테이블/책상류
    "table", "desk",
    # 수납류
    "cabinet", "drawer", "shelf", "bookcase", "wardrobe",
    "dresser", "cupboard", "locker", "sideboard",
    # 침실류
    "bed", "crib",
    # 가전류
    "refrigerator", "washer", "dryer", "dishwasher",
    "microwave", "oven", "stove", "television", "monitor",
    "air_conditioner", "vacuum",
    # 기타
    "piano",
}

label_alias_groups = [
    ("sofa", ["sofa", "couch", "loveseat", "futon", "recliner"]),
    ("chair", ["chair", "dining chair"]),
    ("stool", ["stool"]),
    ("bench", ["bench"]),
    ("ottoman", ["ottoman"]),
    ("table", ["table", "dining table", "coffee table", "side table", "end table"]),
    ("desk", ["desk"]),
    ("tv stand", ["tv stand", "television stand", "media console"]),
    ("kitchen island", ["kitchen island"]),
    ("cabinet", ["cabinet", "kitchen cabinet", "cupboard", "locker", "sideboard"]),
    ("drawer", ["drawer", "drawer unit"]),
    ("shelf", ["shelf", "bookcase", "bookshelf"]),
    ("wardrobe", ["wardrobe", "closet"]),
    ("dresser", ["dresser"]),
    ("shoe rack", ["shoe rack"]),
    ("storage box", ["storage box", "box"]),
    ("bed", ["bed", "mattress", "crib"]),
    ("refrigerator", ["refrigerator", "kimchi refrigerator", "freezer"]),
    ("washing machine", ["washing machine", "washer"]),
    ("dryer", ["dryer"]),
    ("dishwasher", ["dishwasher"]),
    ("microwave", ["microwave"]),
    ("rice cooker", ["rice cooker", "cooker"]),
    ("oven", ["oven"]),
    ("stove", ["stove"]),
    ("television", ["television", "tv"]),
    ("monitor", ["monitor"]),
    ("air conditioner", ["air conditioner", "air_conditioner"]),
    ("air purifier", ["air purifier"]),
    ("vacuum", ["vacuum"]),
    ("piano", ["piano"]),
    ("bicycle", ["bicycle"]),
    ("object", ["object", "furniture"]),
]

alias_to_canonical = {}
for canonical, aliases in label_alias_groups:
    alias_to_canonical[canonical] = canonical
    for alias in aliases:
        alias_to_canonical[alias] = canonical

def _label_tokens(label):
    normalized = str(label).strip().lower()
    normalized = normalized.replace("_", " ").replace("-", " ")
    for ch in "()/[]{}:,;|+":
        normalized = normalized.replace(ch, " ")
    return " ".join(normalized.split())

def _load_label_overrides(raw):
    raw = raw.strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return {_label_tokens(k): _label_tokens(v) for k, v in parsed.items()}
    except Exception:
        pass
    overrides = {}
    for item in raw.split(","):
        if ":" not in item:
            continue
        src, dst = item.split(":", 1)
        src = _label_tokens(src)
        dst = _label_tokens(dst)
        if src and dst:
            overrides[src] = dst
    return overrides

label_overrides = _load_label_overrides(os.environ.get("LABEL_CANONICAL_MAP", ""))

def canonicalize_label(raw_label):
    label = _label_tokens(raw_label)
    if not label:
        return "object"
    if label in label_overrides:
        return label_overrides[label]
    if label in alias_to_canonical:
        return alias_to_canonical[label]

    padded = f" {label} "
    matched = []
    for canonical, aliases in label_alias_groups:
        for alias in aliases:
            alias_norm = _label_tokens(alias)
            if f" {alias_norm} " in padded:
                matched.append((canonical, alias_norm))
                break
    if matched:
        # Prefer more specific multi-word matches, then the prompt/order prior above.
        matched.sort(key=lambda item: len(item[1].split()), reverse=True)
        return matched[0][0]

    parts = label.split()
    if len(parts) > 1:
        for part in parts:
            if part in alias_to_canonical:
                return alias_to_canonical[part]
    return label

result_scene_dir = Path(os.environ["RESULT_SCENE_DIR"]).resolve()
seg_dir = result_scene_dir / "segmentation"
seg_dir.mkdir(parents=True, exist_ok=True)

ann_path = repo / "dataset" / "coco" / "annotations" / "coconut_val.json"
img = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
img_np = np.array(img)
H, W = img_np.shape[:2]

masks = []
labels = []

# ── Grounded-SAM-2 backend ─────────────────────────────────────────────────
# Fallback: gsam2 → rembg (masks가 비어있으면 아래 rembg 블록으로 자동 진행)
if seg_backend == "gsam2":
    try:
        masks, labels = run_grounded_sam2(
            img_np,
            text_prompt=gsam2_prompt,
            gdino_model=gsam2_gdino_model,
            gdino_config=gsam2_gdino_config,
            gdino_checkpoint=gsam2_gdino_checkpoint,
            sam2_config=gsam2_sam2_config,
            sam2_checkpoint=gsam2_sam2_checkpoint,
            box_threshold=gsam2_box_thresh,
            text_threshold=gsam2_txt_thresh,
            nms_threshold=gsam2_nms_thresh,
        )
        if len(masks) == 0:
            print("Grounded-SAM-2 found no masks, falling back to rembg.")
            # seg_backend = "entityv2"  # 이전: EntityV2로 fallback
    except Exception as e:
        print(f"Grounded-SAM-2 failed, falling back to rembg: {e}")
        # seg_backend = "entityv2"  # 이전: EntityV2로 fallback
        masks = []
        labels = []

# ── YOLO backend ───────────────────────────────────────────────────────────
if seg_backend == "yolo":
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
        seg_backend = "entityv2"
        masks = []
        labels = []

# ── EntityV2 backend (default / fallback) ─────────────────────────────────
if seg_backend == "entityv2" and len(masks) == 0:
    try:
        seg_masks = run_entityv2(img_np, threshold=0.1, max_size=1500)
        if len(seg_masks) > 0:
            # fg_idx, _ = run_clipseg(img, seg_masks)
            # if len(fg_idx) > 0:
            if True:
                masks = seg_masks
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

def _mask_bbox(m_bool):
    ys, xs = np.where(m_bool)
    if len(xs) == 0:
        return None
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)

def _expanded_intersects(a, b, gap_ratio):
    aw, ah = max(a[2] - a[0] + 1, 1), max(a[3] - a[1] + 1, 1)
    bw, bh = max(b[2] - b[0] + 1, 1), max(b[3] - b[1] + 1, 1)
    gap = gap_ratio * max(aw, ah, bw, bh)
    return not (
        a[2] + gap < b[0] or b[2] + gap < a[0] or
        a[3] + gap < b[1] or b[3] + gap < a[1]
    )

def _merge_group_label(label):
    canonical = canonicalize_label(label)
    if canonical == "drawer":
        return "drawer"
    if canonical == "shelf":   # bookshelf, bookcase, shelf
        return "shelf"
    return None  # no merging for other classes

if os.environ.get("MASK_MERGE_FRAGMENTS", "1") == "1" and len(masks) > 1:
    gap_ratio = float(os.environ.get("MASK_MERGE_GAP_RATIO", "0.08"))
    items = []
    for idx, m in enumerate(masks):
        m_bool = np.asarray(m).astype(bool)
        bbox = _mask_bbox(m_bool)
        if bbox is None:
            continue
        raw_label = labels[idx] if idx < len(labels) else "object"
        items.append({
            "idx": idx,
            "mask": m_bool,
            "bbox": bbox,
            "label": raw_label,
            "area": int(m_bool.sum()),
            "group": _merge_group_label(raw_label),
        })

    print(f"[seg-debug][merge] before={len(masks)} masks, gap_ratio={gap_ratio}")
    for item in items:
        print(f"[seg-debug][merge]   idx={item['idx']} label={item['label']} area={item['area']} group={item['group']}")

    used = [False] * len(items)
    merged_masks = []
    merged_labels = []
    for i, item in enumerate(items):
        if used[i]:
            continue
        used[i] = True
        cluster = [item]
        changed = True
        while changed:
            changed = False
            cluster_bbox = np.array([
                min(x["bbox"][0] for x in cluster),
                min(x["bbox"][1] for x in cluster),
                max(x["bbox"][2] for x in cluster),
                max(x["bbox"][3] for x in cluster),
            ], dtype=np.float32)
            for j, other in enumerate(items):
                if used[j] or item["group"] is None or other["group"] != item["group"]:
                    continue
                if _expanded_intersects(cluster_bbox, other["bbox"], gap_ratio):
                    used[j] = True
                    cluster.append(other)
                    changed = True

        merged_mask = np.zeros((H, W), dtype=bool)
        for x in cluster:
            merged_mask |= x["mask"]
        label = max(cluster, key=lambda x: x["area"])["label"]
        merged_masks.append(merged_mask)
        merged_labels.append(label)
        if len(cluster) > 1:
            members = ",".join(str(x["idx"]) for x in cluster)
            print(f"[mask-merge] group={item['group']} members={members} -> label={label}")
            print(f"[seg-debug][merge]   MERGED idx={[x['idx'] for x in cluster]} -> label={label} merged_area={int(merged_mask.sum())}")
        else:
            print(f"[seg-debug][merge]   KEPT   idx={item['idx']} label={label} area={item['area']}")

    print(f"[mask-merge] {len(masks)} -> {len(merged_masks)} merged masks")
    print(f"[seg-debug][merge] after={len(merged_masks)} masks")
    masks = merged_masks
    labels = merged_labels

if os.environ.get("MASK_REMOVE_OVERLAPS", "1") == "1" and len(masks) > 1:
    items = []
    for idx, m in enumerate(masks):
        m_bool = np.asarray(m).astype(bool)
        area = int(m_bool.sum())
        if area > 0:
            items.append((area, idx, m_bool, labels[idx] if idx < len(labels) else "object"))

    items.sort(key=lambda x: x[0], reverse=True)
    print(f"[seg-debug][overlap] before={len(masks)} masks, min_mask_area={min_mask_area}")
    occupied = np.zeros((H, W), dtype=bool)
    filtered_masks = []
    filtered_labels = []
    for original_area, idx, m_bool, label in items:
        # Keep larger masks intact and remove their pixels from later overlapping masks.
        # Original:
        # masks = masks
        non_overlap = m_bool & ~occupied
        remaining_area = int(non_overlap.sum())
        if remaining_area < min_mask_area:
            print(
                f"[mask-overlap] drop idx={idx}, label={label}, "
                f"remaining={remaining_area}, original={original_area}"
            )
            print(f"[seg-debug][overlap]   DROP idx={idx} label={label} original_area={original_area} remaining_area={remaining_area} (< min_mask_area={min_mask_area})")
            continue
        filtered_masks.append(non_overlap)
        filtered_labels.append(label)
        occupied |= non_overlap
        print(f"[seg-debug][overlap]   KEEP idx={idx} label={label} original_area={original_area} remaining_area={remaining_area}")

    print(f"[mask-overlap] {len(masks)} -> {len(filtered_masks)} non-overlapping masks")
    print(f"[seg-debug][overlap] after={len(filtered_masks)} masks")
    masks = filtered_masks
    labels = filtered_labels

print(f"[seg-debug][annot] annotation loop: {len(masks)} masks to evaluate, min_mask_area={min_mask_area}, keep_labels={sorted(keep_labels) if keep_labels else 'all'}")
for i, m in enumerate(masks):
    m = np.asarray(m).astype(np.uint8)
    area = int(m.sum())
    if area < min_mask_area:
        print(f"[seg-debug][annot]   DROP idx={i} label={labels[i] if i < len(labels) else 'object'} reason=area area={area} < min_mask_area={min_mask_area}")
        continue
    # 경계에 걸친 마스크 제외: 마스크 면적의 0.1% 이상이 이미지 경계에 닿으면 truncated로 판단.
    boundary_pixels = int(m[0, :].sum() + m[-1, :].sum() + m[:, 0].sum() + m[:, -1].sum())
    truncation_ratio = boundary_pixels / max(area, 1)
    is_truncated = truncation_ratio >= 0.001
    if is_truncated:
        print(f"[seg-debug][annot]   DROP idx={i} label={labels[i] if i < len(labels) else 'object'} reason=truncated boundary_pixels={boundary_pixels} truncation_ratio={truncation_ratio:.4f} area={area}")
        continue
    ys, xs = np.where(m > 0)
    if len(xs) == 0:
        print(f"[seg-debug][annot]   DROP idx={i} reason=empty_mask")
        continue
    raw_label = str(labels[i]).strip() if i < len(labels) else "object"
    label = canonicalize_label(raw_label)
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
            print(f"[seg-debug][annot]   DROP idx={i} raw_label={raw_label!r} canonical={label!r} reason=keep_labels keep_labels={sorted(keep_labels)}")
            continue
    print(f"[seg-debug][annot]   KEEP idx={i} raw_label={raw_label!r} canonical={label!r} area={area} boundary_pixels={boundary_pixels} truncation_ratio={truncation_ratio:.4f}")
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
        "raw_category_name": raw_label,
        "segmentation": {"size": [H, W], "counts": counts},
    })
    seg_vis.append({
        "ann_id": ann_id, "label": label, "category_id": category_id,
        "raw_label": raw_label, "bbox": bbox, "area": area, "mask": m.astype(bool),
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
        "raw_label": item.get("raw_label", item["label"]),
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
    --gpu_idx "$DEPTH_GPU_IDX" \
    --start_index 0 --end_index 1 \
    --split val \
    --save_dir ../experimental_results/single \
    --max_valid_depth "$DEPTH_MAX_VALID" \
    --fit_intercept "$DEPTH_FIT_INTERCEPT" &
PID_DEPTH=$!

# --- Branch B: cropping → then spawn completion + elevation in parallel ---
(
  # Step 3: cropping (must finish before 4 and 5)
  run_logged "3_cropping" \
    python batch_scripts/get_crops_enhanced.py \
      --gpu_idx "$CROP_GPU_IDX" \
      --start_index 0 --end_index 1 \
      --split val \
      --save_dir ../experimental_results/single || exit 1

  # Step 4: amodal completion (uses crops)
  run_logged "4_completion" \
    python batch_scripts/completion.py \
      --gpu_idx "$COMPLETION_GPU_IDX" \
      --start_index 0 --end_index 1 \
      --split val \
      --save_dir ../experimental_results/single &
  PID_COMP_INNER=$!

  # Step 5: elevation (uses _reproj.png fallback from crops)
  run_logged "5_elevation" \
    python batch_scripts/elevation.py \
      --gpu_idx "$ELEVATION_GPU_IDX" \
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
CC="${CC:-gcc}"
CXX="${CXX:-g++}"
export CC CXX
python batch_scripts/reconstruction.py \
  --gpu_idx "$RECON_GPU_IDX" \
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
run_logged "7_alignment" \
  python batch_scripts/whole.py \
    --gpu_idx "$ALIGN_GPU_IDX" \
    --start_index 0 --end_index 1 \
    --split val \
    --save_dir ../experimental_results/single

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
