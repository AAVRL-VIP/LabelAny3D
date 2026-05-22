"""
Drop-in SAM3 segmentation step for LabelAny3D.

Runs in the `sam` conda env (Python 3.12, PyTorch 2.10 cu128, sam3 installed).
Reads an image, runs SAM3 once per text prompt, writes the COCO-style annotation
JSON + per-mask PNGs + overlay + labels.json in the same format the existing
inline pipeline step produces, so downstream stages (depth/crops/...) work
unchanged.

Invoked from run_single_full_pipeline_parallel.sh inside a subshell that
activates the sam env. Parent shell (la3d env) reads the outputs back.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from pycocotools import mask as mask_utils

import sam3
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


# Same indoor preset used by YOLOE path in run_single_full_pipeline_parallel.sh.
# Kept in sync intentionally so swapping USE_YOLO_SEG <-> USE_SAM3 doesn't shift
# the category space the downstream pipeline sees.
INDOOR_PROMPTS = [
    "chair", "table", "sofa", "bed", "desk", "mattress",
    "cabinet", "shelf", "drawer", "tv", "monitor",
    "refrigerator", "microwave", "washing machine",
    "oven", "bench", "furniture", "couch", "bookcase",
    "fan", "storage_box", "box", "closet",
    "air conditioner", "cooker", "wardrobe", "dresser",
    "pantry shelf", "piano", "coffee table", "low table", "television",
]

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


def keep_label(label: str, keep_labels: set) -> bool:
    if not keep_labels:
        return True
    label_l = label.lower()
    for k in keep_labels:
        if k == "furniture":
            if any(w in label_l for w in FURNITURE_KEYWORDS):
                return True
        if k in label_l:
            return True
    return False


def run_sam3_on_image(
    processor: Sam3Processor,
    image: Image.Image,
    prompts: list,
) -> list:
    """Returns list of (mask_bool[H,W], label, score, bbox_xyxy)."""
    state = processor.set_image(image)

    results = []
    for prompt in prompts:
        processor.reset_all_prompts(state)
        state = processor.set_text_prompt(prompt=prompt, state=state)

        masks_t = state.get("masks")
        if masks_t is None or len(masks_t) == 0:
            continue

        masks_np = masks_t.detach().to(torch.bool).cpu().numpy()
        boxes_np = state["boxes"].detach().float().cpu().numpy()
        scores_np = state["scores"].detach().float().cpu().numpy()

        # masks may come as [N,1,H,W] or [N,H,W] — collapse to [N,H,W].
        if masks_np.ndim == 4:
            masks_np = masks_np[:, 0]

        for m, b, s in zip(masks_np, boxes_np, scores_np):
            results.append((m.astype(bool), prompt, float(s), b.tolist()))

    return results


def _interactive_refine_with_clicks(
    model,
    image_np: np.ndarray,
    detections: list,
    enabled: bool,
    device: str,
    use_bf16: bool,
) -> list:
    if not enabled or not detections:
        return detections

    print("[sam3] INTERACTIVE_SAM3=1: launching native click refinement")
    print("[sam3][interactive] Controls: left click=positive, right click=negative")
    print("[sam3][interactive] Keys: n=next, p=prev, c=clear clicks, s=skip, q=finish")

    H, W = image_np.shape[:2]
    window_name = "SAM3 Interactivity (click refine)"
    current_idx = 0
    clicks = {}
    labels = {}
    low_res_logits = {}
    last_render = {}
    autocast_ctx = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if use_bf16 and device.startswith("cuda")
        else torch.autocast("cpu", enabled=False)
    )

    def _predict_with_points(point_coords, point_labels, box_xyxy, mask_input):
        with torch.inference_mode(), autocast_ctx:
            return model.predict_inst(
                {"original_height": H, "original_width": W, "backbone_out": state["backbone_out"]},
                point_coords=point_coords,
                point_labels=point_labels,
                box=np.array(box_xyxy, dtype=np.float32),
                mask_input=mask_input,
                multimask_output=False,
                return_logits=False,
                normalize_coords=False,
            )

    def _render():
        nonlocal last_render
        vis = image_np.copy()
        mask_bool, label, _score, _bbox = detections[current_idx]
        curr_mask = mask_bool.astype(bool)
        if current_idx in low_res_logits:
            points = np.array(clicks[current_idx], dtype=np.float32) if clicks.get(current_idx) else None
            point_labels = np.array(labels[current_idx], dtype=np.int32) if labels.get(current_idx) else None
            pred_masks, pred_ious, pred_low_res = _predict_with_points(
                point_coords=points,
                point_labels=point_labels,
                box_xyxy=detections[current_idx][3],
                mask_input=low_res_logits[current_idx],
            )
            curr_mask = pred_masks[0] > 0.0
            detections[current_idx] = (curr_mask, label, float(pred_ious[0]), detections[current_idx][3])
            low_res_logits[current_idx] = pred_low_res[0]

        overlay = vis.copy()
        color = np.array([50, 205, 50], dtype=np.uint8)
        overlay[curr_mask] = (0.55 * overlay[curr_mask] + 0.45 * color).astype(np.uint8)
        vis = overlay
        x0, y0, x1, y1 = [int(v) for v in detections[current_idx][3]]
        cv2.rectangle(vis, (x0, y0), (x1, y1), (255, 200, 0), 2)
        title = f"[{current_idx + 1}/{len(detections)}] {label}"
        cv2.putText(vis, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)
        for p, l in zip(clicks.get(current_idx, []), labels.get(current_idx, [])):
            px, py = int(p[0]), int(p[1])
            c = (0, 255, 0) if int(l) == 1 else (0, 0, 255)
            cv2.circle(vis, (px, py), 5, c, -1)
            cv2.circle(vis, (px, py), 7, (255, 255, 255), 1)
        last_render = {"mask": curr_mask}
        cv2.imshow(window_name, vis)

    def _on_mouse(event, x, y, _flags, _userdata):
        if event not in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
            return
        lbl = 1 if event == cv2.EVENT_LBUTTONDOWN else 0
        clicks.setdefault(current_idx, []).append([float(x), float(y)])
        labels.setdefault(current_idx, []).append(int(lbl))
        pts = np.array(clicks[current_idx], dtype=np.float32)
        lbs = np.array(labels[current_idx], dtype=np.int32)
        pred_masks, pred_ious, pred_low_res = _predict_with_points(
            point_coords=pts,
            point_labels=lbs,
            box_xyxy=detections[current_idx][3],
            mask_input=low_res_logits.get(current_idx, None),
        )
        new_mask = pred_masks[0] > 0.0
        det = detections[current_idx]
        detections[current_idx] = (new_mask, det[1], float(pred_ious[0]), det[3])
        low_res_logits[current_idx] = pred_low_res[0]
        _render()

    with torch.inference_mode(), autocast_ctx:
        state = Sam3Processor(model).set_image(Image.fromarray(image_np))
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, _on_mouse)
    _render()

    while True:
        key = cv2.waitKey(50) & 0xFF
        if key == 255:
            continue
        if key in (ord("q"), 27):
            break
        if key == ord("n"):
            current_idx = min(current_idx + 1, len(detections) - 1)
            _render()
        elif key == ord("p"):
            current_idx = max(current_idx - 1, 0)
            _render()
        elif key == ord("c"):
            clicks[current_idx] = []
            labels[current_idx] = []
            low_res_logits.pop(current_idx, None)
            _render()
        elif key == ord("s"):
            current_idx = min(current_idx + 1, len(detections) - 1)
            _render()

    cv2.destroyAllWindows()
    return detections


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Input image path")
    ap.add_argument("--out_json", required=True, help="COCO-style annotation JSON path (coconut_val.json)")
    ap.add_argument("--out_seg_dir", required=True, help="Directory for mask PNGs / overlay / labels.json")
    ap.add_argument("--prompts", default="", help="Comma-separated text prompts. Empty → INDOOR_PROMPTS preset.")
    ap.add_argument("--keep_labels", default="", help="Comma-separated KEEP_LABELS filter (e.g. 'furniture').")
    ap.add_argument("--min_mask_area", type=int, default=6400)
    ap.add_argument("--confidence", type=float, default=0.5)
    ap.add_argument("--nms_iou", type=float, default=0.5,
                    help="Mask IoU threshold for deduplicating detections across prompts. "
                         "0 disables NMS.")
    ap.add_argument("--merge_categories",
                    default="cabinet,drawer,shelf,bookcase,wardrobe,dresser,cupboard,"
                            "locker,pantry shelf,closet,storage_box",
                    help="Comma-separated labels eligible for aggressive same-label union "
                         "merging (handles compartmentalized furniture split into sections). "
                         "Other labels (chair, sofa, table, ...) are left untouched.")
    ap.add_argument("--merge_iou", type=float, default=0.05,
                    help="For merge_categories: merge same-label detections whose mask IoU "
                         "exceeds this.")
    ap.add_argument("--merge_bbox_gap", type=int, default=8,
                    help="For merge_categories: also merge same-label detections whose bboxes "
                         "are within this many pixels (touching/near-touching, or overlapping) "
                         "on both axes. Catches sections whose masks don't overlap and "
                         "drawers contained in a parent bbox.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--use_bf16", action="store_true", default=True)
    ap.add_argument("--interactive", action="store_true",
                    help="Enable SAM3 native interactive click refinement "
                         "(left click=positive, right click=negative).")
    args = ap.parse_args()

    image_path = Path(args.image).resolve()
    if not image_path.exists():
        print(f"[sam3] image not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    prompts = [p.strip() for p in args.prompts.split(",") if p.strip()] or INDOOR_PROMPTS
    keep_labels = {x.strip().lower() for x in args.keep_labels.split(",") if x.strip()}

    seg_dir = Path(args.out_seg_dir)
    seg_dir.mkdir(parents=True, exist_ok=True)
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    img = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
    img_np = np.array(img)
    H, W = img_np.shape[:2]

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    autocast_ctx = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if args.use_bf16 and args.device.startswith("cuda")
        else torch.autocast("cpu", enabled=False)
    )

    bpe_path = os.path.join(os.path.dirname(sam3.__file__), "assets", "bpe_simple_vocab_16e6.txt.gz")
    model = build_sam3_image_model(bpe_path=bpe_path).to(args.device).eval()
    processor = Sam3Processor(model, confidence_threshold=args.confidence)

    with torch.inference_mode(), autocast_ctx:
        detections = run_sam3_on_image(
            processor=processor,
            image=img,
            prompts=prompts,
        )
    print(f"[sam3] raw detections: {len(detections)} (prompts={len(prompts)})")

    if args.interactive:
        detections = _interactive_refine_with_clicks(
            model=model,
            image_np=img_np,
            detections=detections,
            enabled=True,
            device=args.device,
            use_bf16=args.use_bf16,
        )

    # Sort highest score first so NMS keeps the most-confident detection per object.
    detections.sort(key=lambda d: -d[2])

    if args.nms_iou > 0 and len(detections) > 1:
        kept = []
        for det in detections:
            m = det[0]
            is_dup = False
            for k in kept:
                inter = np.logical_and(m, k[0]).sum()
                union = np.logical_or(m, k[0]).sum()
                if union == 0:
                    continue
                if inter / union > args.nms_iou:
                    is_dup = True
                    break
            if not is_dup:
                kept.append(det)
        print(f"[sam3] NMS@IoU={args.nms_iou}: {len(detections)} -> {len(kept)}")
        detections = kept

    # Drop bad masks (too small, touching image border) BEFORE category merge so
    # that a union of clean masks can't be poisoned by one border-touching mask
    # that would have been dropped on its own.
    pre_clean_count = len(detections)
    cleaned = []
    for det in detections:
        m_bool = det[0]
        if m_bool.shape != (H, W):
            m_bool = cv2.resize(m_bool.astype(np.uint8), (W, H),
                                interpolation=cv2.INTER_NEAREST).astype(bool)
        m = m_bool.astype(np.uint8)
        if m.sum() < args.min_mask_area:
            continue
        if m[0, :].any() or m[-1, :].any() or m[:, 0].any() or m[:, -1].any():
            continue
        cleaned.append((m_bool, det[1], det[2], det[3]))
    if len(cleaned) != pre_clean_count:
        print(f"[sam3] pre-merge filter (min_area={args.min_mask_area}, no-border): "
              f"{pre_clean_count} -> {len(cleaned)}")
    detections = cleaned

    # Category-restricted same-label union merge.
    # Only applies to "compartmentalized furniture" categories where SAM3 tends
    # to split one physical unit into multiple section detections. Chairs/sofas
    # never get merged here so two adjacent chairs stay separate.
    merge_cats = {c.strip().lower() for c in args.merge_categories.split(",") if c.strip()}
    if merge_cats and len(detections) > 1 and (args.merge_iou > 0 or args.merge_bbox_gap > 0):
        from collections import defaultdict

        def _bbox_gap(b1, b2):
            x_gap = max(b1[0], b2[0]) - min(b1[2], b2[2])
            y_gap = max(b1[1], b2[1]) - min(b1[3], b2[3])
            return x_gap, y_gap

        eligible = [d for d in detections if d[1].lower() in merge_cats]
        untouched = [d for d in detections if d[1].lower() not in merge_cats]

        by_label = defaultdict(list)
        for det in eligible:
            by_label[det[1]].append(det)

        merged_dets = list(untouched)
        for label, dets in by_label.items():
            clusters = []
            for det in dets:
                m, _lab, _score, b = det
                placed = False
                for c in clusters:
                    if args.merge_iou > 0:
                        inter = np.logical_and(m, c["union"]).sum()
                        uni = np.logical_or(m, c["union"]).sum()
                        if uni > 0 and inter / uni > args.merge_iou:
                            placed = True
                    if not placed and args.merge_bbox_gap > 0:
                        for mem in c["members"]:
                            x_gap, y_gap = _bbox_gap(b, mem[3])
                            if x_gap <= args.merge_bbox_gap and y_gap <= args.merge_bbox_gap:
                                placed = True
                                break
                    if placed:
                        c["members"].append(det)
                        c["union"] = np.logical_or(m, c["union"])
                        break
                if not placed:
                    clusters.append({"members": [det], "union": m.copy()})

            for c in clusters:
                if len(c["members"]) == 1:
                    merged_dets.append(c["members"][0])
                else:
                    union_mask = c["union"]
                    max_score = max(d[2] for d in c["members"])
                    ys, xs = np.where(union_mask)
                    if len(xs) == 0:
                        continue
                    bbox_xyxy = [float(xs.min()), float(ys.min()),
                                 float(xs.max()), float(ys.max())]
                    merged_dets.append((union_mask, label, max_score, bbox_xyxy))

        if len(merged_dets) != len(detections):
            print(f"[sam3] category merge ({len(merge_cats)} cats, iou>{args.merge_iou} "
                  f"or bbox_gap<={args.merge_bbox_gap}px): {len(detections)} -> {len(merged_dets)}")
        detections = merged_dets

    annotations = []
    categories = []
    category_to_id = {}
    seg_vis = []
    ann_id = 1
    img_name = image_path.name

    for mask_bool, label, score, bbox_xyxy in detections:
        m = mask_bool.astype(np.uint8)
        # min_area + border filter already applied pre-merge; union of clean
        # masks is also clean, so we only need the label filter here.
        ys, xs = np.where(m > 0)
        if len(xs) == 0:
            continue
        if not keep_label(label, keep_labels):
            continue

        if label not in category_to_id:
            category_to_id[label] = len(category_to_id) + 1
            categories.append({
                "id": category_to_id[label],
                "name": label,
                "supercategory": "object",
            })
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
            "id": ann_id,
            "image_id": 1,
            "category_id": category_id,
            "category_name": label,
            "bbox": bbox,
            "area": area,
            "iscrowd": 0,
            "segmentation": {"size": [H, W], "counts": counts},
            "score": float(score),
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
    with open(out_json, "w") as f:
        json.dump(payload, f)

    overlay = img_np.copy().astype(np.uint8)
    seg_index = []
    for idx, item in enumerate(seg_vis):
        mask = item["mask"]
        color = np.array([(37*(idx+3)) % 256, (97*(idx+5)) % 256, (17*(idx+7)) % 256], dtype=np.uint8)
        mask_u8 = (mask.astype(np.uint8) * 255)
        safe_label = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in item["label"])
        mask_name = f"mask_{idx:02d}_{safe_label}.png"
        cv2.imwrite(str(seg_dir / mask_name), mask_u8)
        overlay[mask] = (0.55 * overlay[mask] + 0.45 * color).astype(np.uint8)
        x, y, w, h = item["bbox"]
        x, y, w, h = int(x), int(y), int(w), int(h)
        col = tuple(int(v) for v in color.tolist())
        cv2.rectangle(overlay, (x, y), (x + w, y + h), col, 2)
        cv2.putText(overlay, item["label"], (x, max(20, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
        seg_index.append({
            "ann_id": item["ann_id"], "label": item["label"], "category_id": item["category_id"],
            "bbox": item["bbox"], "area": item["area"], "mask_file": mask_name,
        })

    Image.fromarray(overlay).save(seg_dir / "overlay.png")
    with open(seg_dir / "labels.json", "w") as f:
        json.dump(seg_index, f, indent=2)

    if keep_labels:
        print(f"[sam3] KEEP_LABELS filter: {sorted(keep_labels)}")
    print(f"[sam3] saved: {out_json}")
    print(f"[sam3] overlay: {seg_dir / 'overlay.png'}")
    print(f"[sam3] num instances: {len(annotations)}")

    if len(annotations) == 0:
        if keep_labels:
            print("가구가 감지되지 않았습니다. 소파, 침대, 테이블 등 큰 가구가 잘 보이도록 촬영해주세요.")
            sys.exit(0)
        raise RuntimeError("No valid SAM3 segmentation instances were produced.")


if __name__ == "__main__":
    main()
