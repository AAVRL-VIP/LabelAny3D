import argparse
from omegaconf import OmegaConf
import sys
import os
from tqdm import tqdm
import torch

sys.path = [
    './',
    '../external/dreamgaussian',
    '../external/One-2-3-45',
] + sys.path

from dataset_model import get_scene
from pathlib import Path
import numpy as np
from PIL import Image
from util import initialize_zero123, estimate_elevation
from batch_scripts.coconut_loader import CoconutLoader, get_dataset_paths


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="path to the yaml config file", default='configs/image.yaml', type=str)
    parser.add_argument('--gpu_idx', type=int, default=0, help='GPU index')
    parser.add_argument('--start_index', type=int, default=0, help='Object index to start processing')
    parser.add_argument('--end_index', type=int, default=1, help='Object index to end processing')
    parser.add_argument("--split", help="split", default="val", type=str)
    parser.add_argument("--save_dir", help="save directory", default="../experimental_results/COCO/", type=str)
    parser.add_argument("--image_path", type=str, default="", help="single image path for single-image mode")

    args, extras = parser.parse_known_args()
    opt = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras))

    if args.split == "single":
        if not args.image_path:
            raise ValueError("--split single requires --image_path")
        if not os.path.exists(args.image_path):
            raise FileNotFoundError(f"Image not found: {args.image_path}")

        image_infos = [{
            "id": 0,
            "file_name": os.path.basename(args.image_path),
            "full_path": args.image_path,
        }]
        dataset_root = None
        loader = None
    else:
        dataset_root, annotations_dir = get_dataset_paths(args.split)
        loader = CoconutLoader(split=args.split, annotations_dir=annotations_dir)
        image_infos = [
            loader.get_image_by_index(i)
            for i in range(args.start_index, args.end_index)
        ]

    assert torch.cuda.is_available()
    device = f"cuda:{args.gpu_idx}"

    zero123_p = initialize_zero123(device)

    for image_info in tqdm(image_infos):
        if args.split == "single":
            img_name = image_info["file_name"]
            image_path = image_info["full_path"]
            scene_name = Path(img_name).stem
        else:
            img_name = image_info["file_name"]
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

        crop_root = out_dir / "crops"
        crop_paths = sorted(list(crop_root.glob("*_reproj.png")))

        if len(crop_paths) == 0:
            print(f"No crop images found in {crop_root}")
            continue

        for crop_path in crop_paths[::-1]:
            obj_id = crop_path.stem.replace("_reproj", "")
            label = obj_id.split("_", 1)[-1]

            full_crop_path = out_dir / "crops" / f"{obj_id}_rgba.png"
            if not full_crop_path.exists():
                full_crop_path = out_dir / "crops" / f"{obj_id}_reproj.png"

            obj_space_dir = out_dir / "object_space" / f"{obj_id}"
            obj_space_dir.mkdir(exist_ok=True, parents=True)

            elevation_path = obj_space_dir / "estimated_elevation.npy"

            if elevation_path.exists():
                print(f"Skipping existing elevation: {elevation_path}")
                continue

            print(f"Estimating elevation for {obj_id} using {full_crop_path.name}")
            full_crop = Image.open(full_crop_path).convert("RGBA")
            estimate_elevation(
                np.array(full_crop.resize((256, 256))),
                obj_space_dir,
                zero123_p
            )
            print(f"Saved: {elevation_path}")