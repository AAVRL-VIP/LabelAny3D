import argparse
from omegaconf import OmegaConf
import sys
import os
import json
from tqdm import tqdm
import torch
import cv2

sys.path = ['./'] + sys.path

from dataset_model import get_scene
from pathlib import Path
import numpy as np
from PIL import Image
from util import read_bounding_boxes_segmentations, crop_object
from scipy.ndimage import binary_opening
from detectron2.structures import BoxMode
from batch_scripts.coconut_loader import CoconutLoader, get_dataset_paths


class SingleImageLoader:
    def __init__(self, annotation_json):
        with open(annotation_json, "r") as f:
            data = json.load(f)

        self.images = data.get("images", [])
        self.annotations = data.get("annotations", [])
        self.categories = data.get("categories", [])

        self.cat_id_to_name = {}
        for c in self.categories:
            self.cat_id_to_name[c["id"]] = c["name"]

        self.annotations_by_image = {}
        for ann in self.annotations:
            image_id = ann["image_id"]
            ann = ann.copy()

            # category_name 추가
            cat_id = ann.get("category_id", -1)
            ann["category_name"] = self.cat_id_to_name.get(cat_id, f"category_{cat_id}")

            # bbox를 float로 강제 변환
            if "bbox" in ann:
                ann["bbox"] = [float(x) for x in ann["bbox"]]

            self.annotations_by_image.setdefault(image_id, []).append(ann)

    def get_image_by_index(self, index):
        return self.images[index]

    def get_annotations(self, image_id):
        return self.annotations_by_image.get(image_id, [])


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="path to the yaml config file", default='configs/image.yaml', type=str)
    parser.add_argument('--gpu_idx', type=int, default=0, help='GPU index')
    parser.add_argument('--start_index', type=int, default=0, help='Object index to start processing')
    parser.add_argument('--end_index', type=int, default=1, help='Object index to end processing')
    parser.add_argument("--split", help="split", default="val", type=str)
    parser.add_argument("--save_dir", help="save directory", default="../experimental_results/COCO/", type=str)

    parser.add_argument("--image_path", type=str, default="", help="single image path for single-image mode")
    parser.add_argument("--annotation_json", type=str, default="", help="single image annotation json for single-image mode")

    args, extras = parser.parse_known_args()
    opt = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras))

    if args.split == "single":
        if not args.image_path:
            raise ValueError("--split single requires --image_path")
        if not args.annotation_json:
            raise ValueError("--split single requires --annotation_json")
        if not os.path.exists(args.image_path):
            raise FileNotFoundError(f"Image not found: {args.image_path}")
        if not os.path.exists(args.annotation_json):
            raise FileNotFoundError(f"Annotation json not found: {args.annotation_json}")

        loader = SingleImageLoader(args.annotation_json)
        dataset_root = None
        image_infos = [
            loader.get_image_by_index(i)
            for i in range(args.start_index, min(args.end_index, len(loader.images)))
        ]
    else:
        dataset_root, annotations_dir = get_dataset_paths(args.split)
        loader = CoconutLoader(split=args.split, annotations_dir=annotations_dir)
        image_infos = [
            loader.get_image_by_index(i)
            for i in range(args.start_index, args.end_index)
        ]

    crop_size = 512

    for image_info in tqdm(image_infos):
        if args.split == "single":
            img_name = os.path.basename(args.image_path)
            image_id = image_info["id"]
            image_path = args.image_path
            scene_name = Path(img_name).stem
        else:
            img_name = image_info["file_name"]
            image_id = image_info["id"]
            image_path = os.path.join(dataset_root, img_name)
            scene_name = img_name.split(".")[0].replace("/", "_").replace("-", "_")

        output_dir = os.path.join(args.save_dir, args.split, scene_name)

        opt.scene.attributes.img_path = image_path
        scene = get_scene(opt.scene.type, opt.scene.attributes)

        out_dir = Path(output_dir)
        print(f"Saving to {out_dir}")
        out_dir.mkdir(exist_ok=True, parents=True)
        (out_dir / "crops").mkdir(exist_ok=True)
        (out_dir / "object_space").mkdir(exist_ok=True)
        (out_dir / "reconstruction").mkdir(exist_ok=True)

        # input.png 없으면 저장
        if not os.path.exists(out_dir / 'input.png'):
            scene.image_pil.save(out_dir / 'input.png')

        annotations = loader.get_annotations(image_id)
        if annotations:
            bboxes, masks, object_ids, instance_labels = read_bounding_boxes_segmentations(
                annotations, scene.image_pil.size
            )
            if len(masks[object_ids]) == 0:
                print(f"No valid objects found in {img_name}")
                continue
        else:
            print(f"No annotations found for {img_name}")
            continue

        bboxes = BoxMode.convert(np.array(bboxes), BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)

        # 기존 코드에서는 enhanced 이미지 기준으로 mask를 x4 스케일업했음
        # 우리는 enhance를 생략하므로 input 이미지 크기 기준 그대로 사용
        masks = np.array(masks)

        # enhanced/input.png 대신 그냥 input.png 사용
        base_image = Image.open(out_dir / 'input.png')
        scene.image_pil = base_image.convert('RGB')
        scene.image_np = np.array(base_image)

        selected_bboxes = []
        valid_obj_count = 0

        for obj_idx in range(len(masks[object_ids]) - 1, -1, -1):
            label = instance_labels[object_ids[obj_idx]]
            label = label.replace(' (', ', ').replace(')', '')
            obj_id = f"{obj_idx}_{label.replace(' ', '_')}"

            mask = binary_opening(masks[object_ids][obj_idx], np.ones((7, 7)))
            if mask.sum() < 6400:
                print(f"Skipped too small object: {obj_id}")
                continue

            selected_bboxes.append(bboxes[object_ids[obj_idx]])

            crop_path = out_dir / "crops" / f"{obj_id}_reproj.png"
            crop_params_path = out_dir / "crops" / f"{obj_id}_crop_params.npy"

            if not crop_path.exists() or not crop_params_path.exists():
                crop, crop_params = crop_object(scene.image_np, mask, crop_size)
                crop.save(crop_path)

                # enhance x4가 없으므로 원래 좌표 그대로 저장
                crop_params = np.array([crop_params[0], crop_params[1], crop_params[2]])
                np.save(crop_params_path, crop_params)

            valid_obj_count += 1

        with open(out_dir / "bboxes.json", "w") as f:
            json.dump(np.array(selected_bboxes).tolist(), f)

        print(f"Saved {valid_obj_count} valid crops for {scene_name}")