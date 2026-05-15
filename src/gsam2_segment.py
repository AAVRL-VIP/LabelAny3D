#!/usr/bin/env python3
"""
Simple Grounding SAM2 segmentation script.
Usage:
    python gsam2_segment.py <image_path> [--out-dir <dir>]
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

GSAM2_PROMPT = (
    "sofa . couch . chair . dining chair . stool . bench . ottoman . futon . recliner . loveseat . "
    "table . dining table . desk . tv stand . kitchen island . "
    "cabinet . kitchen cabinet . sideboard . drawer . drawer unit . "
    "shelf . bookcase . bookshelf . wardrobe . closet . dresser . cupboard . locker . shoe rack . storage box . "
    "bed . crib . "
    "refrigerator . kimchi refrigerator . freezer . washing machine . dryer . dishwasher . "
    "microwave . rice cooker . oven . stove . "
    "television . tv . monitor . tv on stand . air conditioner . air purifier . vacuum . "
    "piano . bicycle"
)

BOX_THRESH = 0.35
TEXT_THRESH = 0.25
NMS_THRESH  = 0.5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", type=Path)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--box-thresh", type=float, default=BOX_THRESH)
    parser.add_argument("--text-thresh", type=float, default=TEXT_THRESH)
    parser.add_argument("--nms-thresh", type=float, default=NMS_THRESH)
    parser.add_argument("--gdino-model", default="auto", choices=["auto", "swinb", "swint"])
    args = parser.parse_args()

    image_path = args.image_path.resolve()
    out_dir = args.out_dir or image_path.parent / f"{image_path.stem}_gsam2"
    out_dir.mkdir(parents=True, exist_ok=True)

    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from model_wrappers import run_grounded_sam2

    img = np.array(Image.open(image_path).convert("RGB"))

    masks, labels = run_grounded_sam2(
        image=img,
        text_prompt=GSAM2_PROMPT,
        gdino_model=args.gdino_model,
        box_threshold=args.box_thresh,
        text_threshold=args.text_thresh,
        nms_threshold=args.nms_thresh,
    )

    print(f"Detected {len(masks)} instances")

    overlay = img.copy()
    index = []
    for i, (mask, label) in enumerate(zip(masks, labels)):
        color = np.array([
            (37 * (i + 3)) % 256,
            (97 * (i + 5)) % 256,
            (17 * (i + 7)) % 256,
        ], dtype=np.uint8)

        mask_u8 = (mask.astype(np.uint8) * 255)
        safe_label = ''.join(c if c.isalnum() or c in ('-', '_') else '_' for c in label)
        mask_name = f"mask_{i:02d}_{safe_label}.png"
        cv2.imwrite(str(out_dir / mask_name), mask_u8)

        overlay[mask] = (0.55 * overlay[mask] + 0.45 * color).astype(np.uint8)
        ys, xs = np.where(mask)
        if len(xs):
            x1, x2, y1, y2 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
            cv2.rectangle(overlay, (x1, y1), (x2, y2), tuple(int(v) for v in color.tolist()), 2)
            cv2.putText(overlay, label, (x1, max(20, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, tuple(int(v) for v in color.tolist()), 2)
            index.append({"id": i, "label": label, "bbox": [x1, y1, x2 - x1, y2 - y1], "mask_file": mask_name})
            print(f"  [{i:02d}] {label}  bbox=({x1},{y1},{x2},{y2})")

    Image.fromarray(overlay).save(out_dir / "overlay.png")
    with open(out_dir / "labels.json", "w") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)

    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
