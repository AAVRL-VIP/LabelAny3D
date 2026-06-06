import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


INDOOR_PROMPTS = [
    "chair", "table", "sofa", "bed", "desk", "mattress",
    "cabinet", "shelf", "drawer", "tv", "monitor",
    "refrigerator", "microwave", "washing machine",
    "oven", "bench", "furniture", "couch", "bookcase",
    "fan", "storage_box", "box", "closet",
    "air conditioner", "cooker", "wardrobe", "dresser",
    "pantry shelf", "piano", "coffee table", "low table", "television",
]


def _dilate(mask: np.ndarray, r: int) -> np.ndarray:
    """4-connectivity binary dilation by r pixels (numpy-only, no scipy/cv2)."""
    out = mask
    for _ in range(int(r)):
        s = out.copy()
        s[1:, :] |= out[:-1, :]
        s[:-1, :] |= out[1:, :]
        s[:, 1:] |= out[:, :-1]
        s[:, :-1] |= out[:, 1:]
        out = s
    return out


def _bbox_overlap_ratio(a, b) -> float:
    """Intersection area / smaller-box area for two xyxy boxes (0..1)."""
    ix0 = max(a[0], b[0]); iy0 = max(a[1], b[1])
    ix1 = min(a[2], b[2]); iy1 = min(a[3], b[3])
    iw = max(0.0, ix1 - ix0); ih = max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(0.0, (b[2] - b[0]) * (b[3] - b[1]))
    smaller = min(area_a, area_b)
    return inter / smaller if smaller > 0 else 0.0


def _merge_same_class(items, gap_px: int, bbox_contain: float = 0.5):
    """Merge same-class detections that touch, sit within gap_px, OR whose
    bounding boxes overlap heavily (one largely inside the other).

    items: list of (mask, label, score, bbox_xyxy).
    Transitive (union-find): a chain of related same-class masks all collapse
    into one. Merged mask = OR of members, score = max, bbox recomputed.
    bbox_contain: merge if inter/smaller-bbox-area >= this (set <0 to disable).
    Returns a new list sorted by descending score.
    """
    n = len(items)
    if n <= 1 or (gap_px < 1 and bbox_contain < 0):
        return items

    dilated = [_dilate(it[0], gap_px) if gap_px >= 1 else None for it in items]
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            if items[i][1] != items[j][1]:
                continue  # different class -> never merge
            related = False
            # (a) mask adjacency: dilated mask i intersects (raw) mask j
            if dilated[i] is not None and np.logical_and(dilated[i], items[j][0]).any():
                related = True
            # (b) bbox overlap / containment of same class
            elif bbox_contain >= 0 and _bbox_overlap_ratio(items[i][3], items[j][3]) >= bbox_contain:
                related = True
            if related:
                union(i, j)

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    merged = []
    for idxs in groups.values():
        if len(idxs) == 1:
            merged.append(items[idxs[0]])
            continue
        m = np.zeros_like(items[idxs[0]][0])
        for k in idxs:
            m |= items[k][0]
        label = items[idxs[0]][1]
        score = max(items[k][2] for k in idxs)
        ys, xs = np.where(m)
        bbox = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
        merged.append((m, label, score, bbox))

    merged.sort(key=lambda d: -d[2])
    return merged


def _load_sam3(device: str):
    repo_root = Path(__file__).resolve().parents[1]
    sam3_repo = str((repo_root / "sam3").resolve())
    if sam3_repo not in sys.path:
        sys.path.insert(0, sam3_repo)

    import torch
    import sam3
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    from pycocotools import mask as mask_utils

    bpe_path = os.path.join(os.path.dirname(sam3.__file__), "assets", "bpe_simple_vocab_16e6.txt.gz")
    try:
        model = build_sam3_image_model(
            bpe_path=bpe_path,
            enable_inst_interactivity=True,
        ).to(device).eval()
    except TypeError:
        # Backward compatibility with variants that may not expose this kwarg.
        model = build_sam3_image_model(bpe_path=bpe_path).to(device).eval()
    return torch, model, Sam3Processor, mask_utils


def _session_path(session_dir: Path) -> Path:
    return session_dir / "session.json"


def _load_session(session_dir: Path) -> dict:
    return json.loads(_session_path(session_dir).read_text(encoding="utf-8"))


def _save_session(session_dir: Path, data: dict) -> None:
    _session_path(session_dir).write_text(json.dumps(data), encoding="utf-8")


def _mask_path(session_dir: Path, idx: int) -> Path:
    return session_dir / f"mask_{idx:03d}.npy"


def _logits_path(session_dir: Path, idx: int) -> Path:
    return session_dir / f"logits_{idx:03d}.npy"


def _reindex_files(session_dir: Path, num_dets: int) -> None:
    for old in sorted(session_dir.glob("mask_*.npy")):
        old.unlink(missing_ok=True)
    for old in sorted(session_dir.glob("logits_*.npy")):
        old.unlink(missing_ok=True)
    # no-op placeholder; caller rewrites desired indices after mutation


def _render_overlay(session_dir: Path, data: dict) -> Path:
    image_np = np.array(ImageOps.exif_transpose(Image.open(data["image_path"])).convert("RGB"))
    vis = image_np.copy().astype(np.uint8)
    curr_idx = int(data["current_idx"])
    for idx, det in enumerate(data["detections"]):
        if not bool(det.get("enabled", True)):
            continue
        mask = np.load(_mask_path(session_dir, idx)).astype(bool)
        color = np.array([50, 205, 50], dtype=np.uint8) if idx == curr_idx else np.array([255, 140, 0], dtype=np.uint8)
        vis[mask] = (0.60 * vis[mask] + 0.40 * color).astype(np.uint8)
        x0, y0, x1, y1 = [int(v) for v in det["bbox_xyxy"]]
        w = 3 if idx == curr_idx else 1
        vis[max(0, y0):min(vis.shape[0], y0 + w), max(0, x0):min(vis.shape[1], x1 + 1)] = color
        vis[max(0, y1 - w + 1):min(vis.shape[0], y1 + 1), max(0, x0):min(vis.shape[1], x1 + 1)] = color
        vis[max(0, y0):min(vis.shape[0], y1 + 1), max(0, x0):min(vis.shape[1], x0 + w)] = color
        vis[max(0, y0):min(vis.shape[0], y1 + 1), max(0, x1 - w + 1):min(vis.shape[1], x1 + 1)] = color
        if idx == curr_idx:
            for p, lbl in zip(det["clicks"], det["click_labels"]):
                px, py = int(p[0]), int(p[1])
                if 0 <= px < vis.shape[1] and 0 <= py < vis.shape[0]:
                    c = np.array([0, 255, 0], dtype=np.uint8) if int(lbl) == 1 else np.array([255, 0, 0], dtype=np.uint8)
                    vis[max(0, py - 3):min(vis.shape[0], py + 4), max(0, px - 3):min(vis.shape[1], px + 4)] = c
    out = session_dir / "overlay.png"
    Image.fromarray(vis).save(out)
    return out


def _instances_snapshot(data: dict):
    return [
        {
            "index": i,
            "label": d["label"],
            "score": float(d["score"]),
            "bbox_xyxy": d["bbox_xyxy"],
            "enabled": bool(d.get("enabled", True)),
        }
        for i, d in enumerate(data["detections"])
    ]


def _start(args):
    session_dir = Path(args.session_dir)
    session_dir.mkdir(parents=True, exist_ok=True)

    torch, model, Sam3Processor, _ = _load_sam3(args.device)
    img = ImageOps.exif_transpose(Image.open(args.image)).convert("RGB")
    prompts = [p.strip() for p in args.prompts.split(",") if p.strip()] or list(INDOOR_PROMPTS)
    processor = Sam3Processor(model, confidence_threshold=args.confidence)
    autocast_ctx = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if args.use_bf16 and args.device.startswith("cuda")
        else torch.autocast("cpu", enabled=False)
    )

    H, W = img.height, img.width
    raw_dets = []
    with torch.inference_mode(), autocast_ctx:
        state = processor.set_image(img)
        for prompt in prompts:
            processor.reset_all_prompts(state)
            state = processor.set_text_prompt(prompt=prompt, state=state)
            masks_t = state.get("masks")
            if masks_t is None or len(masks_t) == 0:
                continue
            masks_np = masks_t.detach().to(torch.bool).cpu().numpy()
            boxes_np = state["boxes"].detach().float().cpu().numpy()
            scores_np = state["scores"].detach().float().cpu().numpy()
            if masks_np.ndim == 4:
                masks_np = masks_np[:, 0]
            for m, b, s in zip(masks_np, boxes_np, scores_np):
                raw_dets.append((m.astype(bool), prompt, float(s), [float(v) for v in b.tolist()]))

    if not raw_dets:
        print(json.dumps({"ok": False, "error": "no_detections"}))
        return

    # Match main SAM3 pipeline behavior: sort by score, NMS, area/border filtering.
    raw_dets.sort(key=lambda d: -d[2])
    nms_iou = 0.5
    kept = []
    for det in raw_dets:
        m = det[0]
        is_dup = False
        for k in kept:
            inter = np.logical_and(m, k[0]).sum()
            union = np.logical_or(m, k[0]).sum()
            if union > 0 and (inter / union) > nms_iou:
                is_dup = True
                break
        if not is_dup:
            kept.append(det)

    min_mask_area = 6400
    filtered = []
    for m, label, score, bbox in kept:
        if m.shape != (H, W):
            continue
        mu = m.astype(np.uint8)
        if mu.sum() < min_mask_area:
            continue
        if mu[0, :].any() or mu[-1, :].any() or mu[:, 0].any() or mu[:, -1].any():
            continue
        filtered.append((m, label, score, bbox))

    if not filtered:
        print(json.dumps({"ok": False, "error": "all_filtered_out"}))
        return

    # Merge same-class detections that overlap or sit right next to each other
    # (e.g. a stack of drawers -> one drawer). NMS above only removes high-IoU
    # duplicates; this handles adjacent-but-distinct instances of one class.
    if getattr(args, "merge_same_class", True):
        gap_px = args.merge_gap_px
        if gap_px is None or gap_px < 0:
            gap_px = max(4, int(round(0.012 * max(H, W))))  # auto: ~1.2% of long side
        filtered = _merge_same_class(filtered, gap_px, bbox_contain=args.merge_bbox_contain)

    # keep top-k for usability
    max_instances = 12
    filtered = filtered[:max_instances]

    detections = []
    for i, (m, label, score, bbox) in enumerate(filtered):
        np.save(_mask_path(session_dir, i), m.astype(bool))
        detections.append(
            {
                "label": label,
                "score": float(score),
                "bbox_xyxy": bbox,
                "clicks": [],
                "click_labels": [],
                "enabled": True,
            }
        )

    data = {
        "image_path": str(Path(args.image).resolve()),
        "image_name": Path(args.image).name,
        "device": args.device,
        "use_bf16": bool(args.use_bf16),
        "current_idx": 0,
        "detections": detections,
    }
    _save_session(session_dir, data)
    _render_overlay(session_dir, data)
    print(json.dumps({"ok": True, "num_instances": len(detections), "current_idx": 0, "instances": _instances_snapshot(data)}))


def _select(args):
    session_dir = Path(args.session_dir)
    data = _load_session(session_dir)
    idx = int(args.instance_index)
    if idx < 0 or idx >= len(data["detections"]):
        print(json.dumps({"ok": False, "error": "instance_index_out_of_range"}))
        return
    data["current_idx"] = idx
    _save_session(session_dir, data)
    _render_overlay(session_dir, data)
    det = data["detections"][idx]
    print(json.dumps({"ok": True, "current_idx": idx, "label": det["label"], "score": det["score"], "num_clicks": len(det["clicks"])}))


def _click(args):
    session_dir = Path(args.session_dir)
    data = _load_session(session_dir)
    idx = int(args.instance_index) if args.instance_index is not None else int(data["current_idx"])
    if idx < 0 or idx >= len(data["detections"]):
        print(json.dumps({"ok": False, "error": "instance_index_out_of_range"}))
        return

    torch, model, Sam3Processor, _ = _load_sam3(data["device"])
    img = ImageOps.exif_transpose(Image.open(data["image_path"])).convert("RGB")
    det = data["detections"][idx]
    det["clicks"].append([float(args.x), float(args.y)])
    det["click_labels"].append(int(args.label))

    processor = Sam3Processor(model, confidence_threshold=0.5)
    autocast_ctx = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if data["use_bf16"] and data["device"].startswith("cuda")
        else torch.autocast("cpu", enabled=False)
    )
    with torch.inference_mode(), autocast_ctx:
        state = processor.set_image(img)
        mask_input = None
        lpath = _logits_path(session_dir, idx)
        if lpath.exists():
            mask_input = np.load(lpath)
        pred_masks, pred_ious, pred_low_res = model.predict_inst(
            {"original_height": img.height, "original_width": img.width, "backbone_out": state["backbone_out"]},
            point_coords=np.array(det["clicks"], dtype=np.float32),
            point_labels=np.array(det["click_labels"], dtype=np.int32),
            box=np.array(det["bbox_xyxy"], dtype=np.float32),
            mask_input=mask_input,
            multimask_output=False,
            return_logits=False,
            normalize_coords=False,
        )

    np.save(_mask_path(session_dir, idx), (pred_masks[0] > 0.0).astype(bool))
    np.save(_logits_path(session_dir, idx), pred_low_res[0])
    det["score"] = float(pred_ious[0])
    ys, xs = np.where(pred_masks[0] > 0.0)
    if len(xs) > 0:
        det["bbox_xyxy"] = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
    data["current_idx"] = idx
    _save_session(session_dir, data)
    _render_overlay(session_dir, data)
    print(json.dumps({"ok": True, "instance_index": idx, "label": det["label"], "score": det["score"], "num_clicks": len(det["clicks"])}))


def _tap(args):
    session_dir = Path(args.session_dir)
    data = _load_session(session_dir)
    x = int(round(float(args.x)))
    y = int(round(float(args.y)))
    pref_idx = int(args.instance_index) if args.instance_index is not None else int(data["current_idx"])

    masks = [np.load(_mask_path(session_dir, i)).astype(bool) for i in range(len(data["detections"]))]
    h, w = masks[0].shape if masks else (0, 0)
    if not (0 <= x < w and 0 <= y < h):
        print(json.dumps({"ok": False, "error": "point_out_of_bounds"}))
        return

    # 1) If clicking on currently selected object, toggle-off from active set.
    if 0 <= pref_idx < len(masks) and masks[pref_idx][y, x]:
        data["detections"][pref_idx]["enabled"] = False
        enabled_indices = [i for i, d in enumerate(data["detections"]) if bool(d.get("enabled", True))]
        if enabled_indices:
            data["current_idx"] = enabled_indices[0]
        _save_session(session_dir, data)
        _render_overlay(session_dir, data)
        print(json.dumps({"ok": True, "action": "toggled_off", "toggled_index": pref_idx, "num_instances": len(data["detections"]), "current_idx": data["current_idx"], "instances": _instances_snapshot(data)}))
        return

    # 2) If clicking on another existing object:
    #    disabled -> enable + select, enabled -> select only.
    for i, m in enumerate(masks):
        if m[y, x]:
            was_enabled = bool(data["detections"][i].get("enabled", True))
            data["detections"][i]["enabled"] = True
            data["current_idx"] = i
            _save_session(session_dir, data)
            _render_overlay(session_dir, data)
            action = "selected" if was_enabled else "toggled_on"
            print(json.dumps({"ok": True, "action": action, "selected_index": i, "toggled_index": i, "num_instances": len(data["detections"]), "current_idx": i, "instances": _instances_snapshot(data)}))
            return

    # 3) Background click: create a new object from a positive point prompt.
    torch, model, Sam3Processor, _ = _load_sam3(data["device"])
    img = ImageOps.exif_transpose(Image.open(data["image_path"])).convert("RGB")
    processor = Sam3Processor(model, confidence_threshold=0.5)
    autocast_ctx = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if data["use_bf16"] and data["device"].startswith("cuda")
        else torch.autocast("cpu", enabled=False)
    )
    with torch.inference_mode(), autocast_ctx:
        state = processor.set_image(img)
        pred_masks, pred_ious, pred_low_res = model.predict_inst(
            {"original_height": img.height, "original_width": img.width, "backbone_out": state["backbone_out"]},
            point_coords=np.array([[float(x), float(y)]], dtype=np.float32),
            point_labels=np.array([1], dtype=np.int32),
            box=None,
            mask_input=None,
            multimask_output=False,
            return_logits=False,
            normalize_coords=False,
        )

    new_mask = (pred_masks[0] > 0.0).astype(bool)
    if new_mask.sum() < 1200:
        print(json.dumps({"ok": True, "action": "ignored", "reason": "too_small", "num_instances": len(data["detections"]), "current_idx": data["current_idx"], "instances": _instances_snapshot(data)}))
        return
    ys, xs = np.where(new_mask)
    bbox = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
    label = args.new_label or "furniture"
    new_idx = len(data["detections"])
    data["detections"].append(
        {
            "label": label,
            "score": float(pred_ious[0]),
            "bbox_xyxy": bbox,
            "clicks": [[float(x), float(y)]],
            "click_labels": [1],
            "enabled": True,
        }
    )
    np.save(_mask_path(session_dir, new_idx), new_mask.astype(bool))
    np.save(_logits_path(session_dir, new_idx), pred_low_res[0])
    data["current_idx"] = new_idx
    _save_session(session_dir, data)
    _render_overlay(session_dir, data)
    print(json.dumps({"ok": True, "action": "added", "added_index": new_idx, "num_instances": len(data["detections"]), "current_idx": new_idx, "label": label, "instances": _instances_snapshot(data)}))


def _commit(args):
    from pycocotools import mask as mask_utils

    session_dir = Path(args.session_dir)
    data = _load_session(session_dir)
    out_json = Path(args.out_json)
    out_seg_dir = Path(args.out_seg_dir)
    image_np = np.array(ImageOps.exif_transpose(Image.open(data["image_path"])).convert("RGB"))
    H, W = image_np.shape[:2]
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_seg_dir.mkdir(parents=True, exist_ok=True)

    selected_set = None
    if args.selected_indices:
        try:
            selected_set = {int(x) for x in args.selected_indices.split(",") if str(x).strip() != ""}
        except Exception:
            selected_set = None

    annotations = []
    categories = []
    category_to_id = {}
    seg_vis = []
    ann_id = 1
    for idx, det in enumerate(data["detections"]):
        if not bool(det.get("enabled", True)):
            continue
        if selected_set is not None and idx not in selected_set:
            continue
        mask_bool = np.load(_mask_path(session_dir, idx)).astype(bool)
        m = mask_bool.astype(np.uint8)
        ys, xs = np.where(m > 0)
        if len(xs) == 0:
            continue
        label = det["label"]
        if label not in category_to_id:
            category_to_id[label] = len(category_to_id) + 1
            categories.append({"id": category_to_id[label], "name": label, "supercategory": "object"})
        cid = category_to_id[label]
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        bbox = [float(x1), float(y1), float(x2 - x1 + 1), float(y2 - y1 + 1)]
        rle = mask_utils.encode(np.asfortranarray(m))
        counts = rle["counts"].decode("utf-8") if isinstance(rle["counts"], (bytes, bytearray)) else rle["counts"]
        annotations.append(
            {
                "id": ann_id,
                "image_id": 1,
                "category_id": cid,
                "category_name": label,
                "bbox": bbox,
                "area": float(m.sum()),
                "iscrowd": 0,
                "segmentation": {"size": [H, W], "counts": counts},
                "score": float(det["score"]),
            }
        )
        seg_vis.append({"ann_id": ann_id, "label": label, "category_id": cid, "bbox": bbox, "area": float(m.sum()), "mask": mask_bool})
        ann_id += 1

    payload = {"images": [{"id": 1, "file_name": data["image_name"], "width": W, "height": H}], "annotations": annotations, "categories": categories}
    out_json.write_text(json.dumps(payload), encoding="utf-8")

    overlay = image_np.copy().astype(np.uint8)
    seg_index = []
    for idx, item in enumerate(seg_vis):
        mask = item["mask"]
        color = np.array([(37 * (idx + 3)) % 256, (97 * (idx + 5)) % 256, (17 * (idx + 7)) % 256], dtype=np.uint8)
        mask_u8 = (mask.astype(np.uint8) * 255)
        safe_label = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in item["label"])
        mask_name = f"mask_{idx:02d}_{safe_label}.png"
        Image.fromarray(mask_u8).save(out_seg_dir / mask_name)
        overlay[mask] = (0.55 * overlay[mask] + 0.45 * color).astype(np.uint8)
        seg_index.append({"ann_id": item["ann_id"], "label": item["label"], "category_id": item["category_id"], "bbox": item["bbox"], "area": item["area"], "mask_file": mask_name})
    Image.fromarray(overlay).save(out_seg_dir / "overlay.png")
    (out_seg_dir / "labels.json").write_text(json.dumps(seg_index, indent=2), encoding="utf-8")
    print(json.dumps({"ok": True, "num_instances": len(annotations)}))


def _close(args):
    session_dir = Path(args.session_dir)
    if session_dir.exists():
        for p in session_dir.glob("*"):
            try:
                p.unlink()
            except Exception:
                pass
        try:
            session_dir.rmdir()
        except Exception:
            pass
    print(json.dumps({"ok": True}))


def _set_enabled(args):
    session_dir = Path(args.session_dir)
    data = _load_session(session_dir)
    idx = int(args.instance_index)
    if idx < 0 or idx >= len(data["detections"]):
        print(json.dumps({"ok": False, "error": "instance_index_out_of_range"}))
        return
    enabled = bool(int(args.enabled))
    data["detections"][idx]["enabled"] = enabled
    if enabled:
        data["current_idx"] = idx
    _save_session(session_dir, data)
    _render_overlay(session_dir, data)
    print(json.dumps({"ok": True, "instance_index": idx, "enabled": enabled, "current_idx": data["current_idx"], "instances": _instances_snapshot(data)}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--action", required=True, choices=["start", "select", "click", "tap", "set_enabled", "commit", "close"])
    ap.add_argument("--session_dir", required=True)
    ap.add_argument("--image")
    ap.add_argument("--prompts", default=",".join(INDOOR_PROMPTS))
    ap.add_argument("--confidence", type=float, default=0.5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--use_bf16", action="store_true", default=True)
    ap.add_argument("--instance_index", type=int)
    ap.add_argument("--x", type=float)
    ap.add_argument("--y", type=float)
    ap.add_argument("--label", type=int)
    ap.add_argument("--out_json")
    ap.add_argument("--out_seg_dir")
    ap.add_argument("--new_label", default="furniture")
    ap.add_argument("--selected_indices", default="")
    ap.add_argument("--enabled", type=int)
    ap.add_argument("--no_merge_same_class", dest="merge_same_class", action="store_false",
                    default=True, help="Disable auto-merging of adjacent same-class masks.")
    ap.add_argument("--merge_gap_px", type=int, default=-1,
                    help="Max pixel gap to treat two same-class masks as one. -1 = auto (~1.2%% of long side).")
    ap.add_argument("--merge_bbox_contain", type=float, default=0.5,
                    help="Merge same-class if (bbox intersection / smaller bbox) >= this. <0 disables.")
    args = ap.parse_args()

    if args.action == "start":
        _start(args)
    elif args.action == "select":
        _select(args)
    elif args.action == "click":
        _click(args)
    elif args.action == "tap":
        _tap(args)
    elif args.action == "set_enabled":
        _set_enabled(args)
    elif args.action == "commit":
        _commit(args)
    elif args.action == "close":
        _close(args)


if __name__ == "__main__":
    main()
