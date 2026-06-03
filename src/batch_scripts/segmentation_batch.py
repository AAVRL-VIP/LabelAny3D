"""
Batch segmentation: produce a single multi-image COCONUT-style annotation JSON.

Segments every input image in ONE process so the segmentation model is loaded
only once (vs. the per-image pipeline that reloads it for each image). Each image
gets a unique image_id; annotations carry that id. Per-scene segmentation/ dirs
and input.png are written so downstream stages (val split) pick them up.

Usage:
    python batch_scripts/segmentation_batch.py \
        --images /abs/a.jpg /abs/b.jpg ... \
        --save_dir ../experimental_results/single --split val

Honors the same env vars as the inline segmentation in
run_single_full_pipeline_parallel.sh (MIN_MASK_AREA, KEEP_LABELS, USE_YOLO_SEG,
YOLO_*, INDOOR_CLASSES_DEFAULT).
"""
import argparse
import json
import os
import random
import sys
from pathlib import Path

sys.path = ['./'] + sys.path

import cv2
import numpy as np
import rembg
import torch
from PIL import Image, ImageOps
from pycocotools import mask as mask_utils

from model_wrappers import run_entityv2, run_clipseg, run_ovsam, run_yolo_seg

seed = 7
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

FURNITURE_KEYWORDS = {
    "furniture", "sofa", "couch", "chair", "stool", "bench",
    "ottoman", "futon", "recliner", "loveseat",
    "table", "desk",
    "cabinet", "drawer", "shelf", "bookcase", "wardrobe",
    "dresser", "cupboard", "locker",
    "bed", "mattress", "crib",
    "refrigerator", "washer", "dryer", "dishwasher",
    "microwave", "oven", "stove", "television", "monitor",
    "air_conditioner", "vacuum",
    "piano", "ironing_board",
}


def segment_image(img_np, cfg):
    """Return (masks, labels) for one image using the configured segmenter chain."""
    masks, labels = [], []

    if cfg["use_yolo_seg"]:
        try:
            masks, labels = run_yolo_seg(
                img_np,
                model_path=cfg["yolo_model"],
                conf=cfg["yolo_conf"],
                iou=cfg["yolo_iou"],
                classes=cfg["yolo_classes"] if cfg["yolo_classes"] else None,
                max_det=cfg["yolo_max_det"],
            )
            print(f"YOLO segmentation produced {len(masks)} masks")
        except Exception as e:
            print(f"YOLO segmentation failed, switching to EntityV2 pipeline: {e}")
            masks, labels = [], []

    if not cfg["use_yolo_seg"] and len(masks) == 0:
        try:
            seg_masks = run_entityv2(img_np, threshold=0.1, max_size=1500)
            if len(seg_masks) > 0:
                fg_idx, _ = run_clipseg(Image.fromarray(img_np), seg_masks)
                if len(fg_idx) > 0:
                    masks = [seg_masks[i] for i in fg_idx]
                    try:
                        labels = run_ovsam(Image.fromarray(img_np), np.array(masks).astype(np.uint8) * 255)
                    except Exception as e:
                        print(f"OVSAM failed, falling back to generic labels: {e}")
                        labels = ["object"] * len(masks)
        except Exception as e:
            print(f"EntityV2/CLIPSeg pipeline failed, switching to rembg fallback: {e}")
            masks, labels = [], []

    if len(masks) == 0:
        rgba = np.array(rembg.remove(Image.fromarray(img_np), rembg.new_session("isnet-general-use")))
        fg = (rgba[..., 3] > 127).astype(np.uint8)
        num_labels, cc = cv2.connectedComponents(fg, connectivity=8)
        comp_masks = []
        for comp_id in range(1, num_labels):
            m = cc == comp_id
            if int(m.sum()) >= cfg["min_mask_area"]:
                comp_masks.append(m)
        if len(comp_masks) > 0:
            masks = comp_masks
            labels = ["object"] * len(masks)
            print(f"rembg fallback produced {len(masks)} connected components")
        elif fg.sum() >= cfg["min_mask_area"]:
            masks = [fg.astype(bool)]
            labels = ["object"]

    if len(labels) != len(masks):
        labels = ["object"] * len(masks)
    return masks, labels


def build_annotations(masks, labels, cfg, H, W, image_id, category_to_id, ann_id_start):
    """Filter masks and emit COCONUT annotations + visualization records."""
    annotations, seg_vis = [], []
    ann_id = ann_id_start
    keep_labels = cfg["keep_labels"]

    for i, m in enumerate(masks):
        m = np.asarray(m).astype(np.uint8)
        if m.sum() < cfg["min_mask_area"]:
            continue
        if m[0, :].any() or m[-1, :].any() or m[:, 0].any() or m[:, -1].any():
            continue  # truncated at image border
        ys, xs = np.where(m > 0)
        if len(xs) == 0:
            continue
        raw_label = str(labels[i]).strip() if i < len(labels) else "object"
        label = raw_label if raw_label else "object"
        if keep_labels:
            label_l = label.lower()
            matched = False
            for k in keep_labels:
                if k == "furniture" and any(w in label_l for w in FURNITURE_KEYWORDS):
                    matched = True
                    break
                if k in label_l:
                    matched = True
                    break
            if not matched:
                continue
        if label not in category_to_id:
            category_to_id[label] = len(category_to_id) + 1
        category_id = category_to_id[label]
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        bbox = [float(x1), float(y1), float(x2 - x1 + 1), float(y2 - y1 + 1)]
        rle = mask_utils.encode(np.asfortranarray(m))
        counts = rle["counts"]
        if isinstance(counts, (bytes, bytearray)):
            counts = counts.decode("utf-8")
        annotations.append({
            "id": ann_id, "image_id": image_id, "category_id": category_id,
            "category_name": label, "bbox": bbox, "area": float(m.sum()), "iscrowd": 0,
            "segmentation": {"size": [H, W], "counts": counts},
        })
        seg_vis.append({
            "ann_id": ann_id, "label": label, "category_id": category_id,
            "bbox": bbox, "area": float(m.sum()), "mask": m.astype(bool),
        })
        ann_id += 1
    return annotations, seg_vis, ann_id


def write_seg_vis(img_np, seg_vis, seg_dir):
    seg_dir.mkdir(parents=True, exist_ok=True)
    overlay = img_np.copy().astype(np.uint8)
    seg_index = []
    for idx, item in enumerate(seg_vis):
        mask = item["mask"]
        color = np.array([(37 * (idx + 3)) % 256, (97 * (idx + 5)) % 256, (17 * (idx + 7)) % 256], dtype=np.uint8)
        safe_label = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in item["label"])
        mask_name = f"mask_{idx:02d}_{safe_label}.png"
        cv2.imwrite(str(seg_dir / mask_name), mask.astype(np.uint8) * 255)
        overlay[mask] = (0.55 * overlay[mask] + 0.45 * color).astype(np.uint8)
        x, y, w, h = (int(v) for v in item["bbox"])
        cv2.rectangle(overlay, (x, y), (x + w, y + h), tuple(int(v) for v in color.tolist()), 2)
        cv2.putText(overlay, item["label"], (x, max(20, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    tuple(int(v) for v in color.tolist()), 2)
        seg_index.append({
            "ann_id": item["ann_id"], "label": item["label"], "category_id": item["category_id"],
            "bbox": item["bbox"], "area": item["area"], "mask_file": mask_name,
        })
    Image.fromarray(overlay).save(seg_dir / "overlay.png")
    with open(seg_dir / "labels.json", "w") as f:
        json.dump(seg_index, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", nargs="+", required=True, help="absolute image paths")
    parser.add_argument("--save_dir", default="../experimental_results/single", type=str)
    parser.add_argument("--split", default="val", type=str)
    args = parser.parse_args()

    repo = Path.cwd().parent
    coco_img_dir = repo / "dataset" / "coco" / "images" / "val2017"
    ann_path = repo / "dataset" / "coco" / "annotations" / "coconut_val.json"
    coco_img_dir.mkdir(parents=True, exist_ok=True)
    ann_path.parent.mkdir(parents=True, exist_ok=True)

    indoor = os.environ.get("INDOOR_CLASSES_DEFAULT", "")
    yolo_classes_raw = os.environ.get("YOLO_CLASSES", "")
    yolo_classes = [x.strip() for x in yolo_classes_raw.split(",") if x.strip()]
    if not yolo_classes and os.environ.get("YOLO_CLASS_PRESET", "indoor").strip().lower() == "indoor":
        yolo_classes = [x.strip() for x in indoor.split(",") if x.strip()]

    cfg = {
        "min_mask_area": int(os.environ.get("MIN_MASK_AREA", "800")),
        "keep_labels": {x.strip().lower() for x in os.environ.get("KEEP_LABELS", "").split(",") if x.strip()},
        "use_yolo_seg": os.environ.get("USE_YOLO_SEG", "0") == "1",
        "yolo_model": os.environ.get("YOLO_SEG_MODEL", "yoloe-26l-seg.pt"),
        "yolo_conf": float(os.environ.get("YOLO_CONF", "0.45")),
        "yolo_iou": float(os.environ.get("YOLO_IOU", "0.55")),
        "yolo_max_det": int(os.environ.get("YOLO_MAX_DET", "300")),
        "yolo_classes": yolo_classes,
    }

    images_payload, annotations_payload = [], []
    category_to_id = {}
    ann_id = 1
    n_processed = 0

    for idx, img_path in enumerate(args.images):
        image_id = idx + 1
        img_path = str(Path(img_path).resolve())
        img_name = os.path.basename(img_path)
        scene_name = img_name.split(".")[0].replace("/", "_").replace("-", "_")
        scene_dir = Path(args.save_dir) / args.split / scene_name
        scene_dir.mkdir(parents=True, exist_ok=True)

        img = ImageOps.exif_transpose(Image.open(img_path)).convert("RGB")
        img_np = np.array(img)
        H, W = img_np.shape[:2]

        # make sure the image is reachable by downstream stages (val2017 + scene input.png)
        coco_dst = coco_img_dir / img_name
        if not coco_dst.exists():
            img.save(coco_dst)
        img.save(scene_dir / "input.png")

        print(f"[{image_id}/{len(args.images)}] segmenting {img_name} ({W}x{H})")
        masks, labels = segment_image(img_np, cfg)
        annos, seg_vis, ann_id = build_annotations(
            masks, labels, cfg, H, W, image_id, category_to_id, ann_id
        )
        write_seg_vis(img_np, seg_vis, scene_dir / "segmentation")

        images_payload.append({"id": image_id, "file_name": img_name, "width": W, "height": H})
        annotations_payload.extend(annos)
        n_processed += 1
        print(f"    -> {len(annos)} instances")

    categories = [{"id": cid, "name": name, "supercategory": "object"}
                  for name, cid in sorted(category_to_id.items(), key=lambda kv: kv[1])]
    payload = {"images": images_payload, "annotations": annotations_payload, "categories": categories}
    with open(ann_path, "w") as f:
        json.dump(payload, f)

    print(f"Saved batch annotation: {ann_path}")
    print(f"BATCH_NUM_IMAGES={len(images_payload)}")
    print(f"BATCH_TOTAL_INSTANCES={len(annotations_payload)}")


if __name__ == "__main__":
    main()
