import argparse
from omegaconf import OmegaConf
import sys
import os
import json
from tqdm import tqdm
import torch
import trimesh
import cv2

sys.path = ['./'] + sys.path

from dataset_model import get_scene
from pathlib import Path
import numpy as np
from PIL import Image
from util import restore_mask_from_crop, align_to_depth_match, draw_cube
from util_3dbox import save_3d_with_ground_alignment_bbox
from matching.process_image_space import load_model
from batch_scripts.coconut_loader import CoconutLoader, get_dataset_paths


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default='configs/image.yaml', type=str)
    parser.add_argument('--gpu_idx', type=int, default=0)
    parser.add_argument('--start_index', type=int, default=0)
    parser.add_argument('--end_index', type=int, default=1)
    parser.add_argument("--split", default="val", type=str)
    parser.add_argument("--save_dir", default="../experimental_results/COCO/", type=str)
    parser.add_argument("--image_path", type=str, default="", help="single image path")

    args, extras = parser.parse_known_args()
    opt = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras))

    if args.split == "single":
        if not args.image_path:
            raise ValueError("--split single requires --image_path")
        if not os.path.exists(args.image_path):
            raise FileNotFoundError(args.image_path)

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

    mast3r_model = load_model(device)

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

        if os.path.exists(out_dir / '3dbbox.json'):
            print("Already done")
            continue

        cam_params_path = out_dir / 'cam_params.json'
        depth_map_path = out_dir / 'depth_map.npy'

        if not cam_params_path.exists():
            print(f"Missing cam_params.json: {cam_params_path}")
            continue
        if not depth_map_path.exists():
            print(f"Missing depth_map.npy: {depth_map_path}")
            continue

        with open(cam_params_path, 'r') as fp:
            cam_params = json.load(fp)

        K_img = np.array(cam_params['K'])
        pose = np.array(cam_params['c2w'])
        depth_map = np.load(depth_map_path)

        scene_mesh = trimesh.Scene([None])
        crop_root = out_dir / "crops"
        crop_paths = sorted(list(crop_root.glob("*_reproj.png")))

        if len(crop_paths) == 0:
            print("No crops found")
            continue

        placed_count = 0

        for crop_path in crop_paths[::-1]:
            obj_id = crop_path.stem.replace("_reproj", "")

            crop = Image.open(crop_path)
            crop_params_path = out_dir / "crops" / f"{obj_id}_crop_params.npy"

            if not crop_params_path.exists():
                print(f"Missing crop params: {obj_id}")
                continue

            crop_params = np.load(crop_params_path)

            crop_np = np.array(crop)
            if crop_np.ndim == 3 and crop_np.shape[2] == 4:
                resized_mask = crop_np[:, :, 3] > 127
            else:
                # alpha가 없는 경우, 거의 흰색이 아닌 영역을 foreground로 간주
                resized_mask = np.any(crop_np[:, :, :3] < 250, axis=2)

            mask = restore_mask_from_crop(
                resized_mask,
                crop_params[0],
                crop_params[1],
                crop_params[2],
                scene.image_np.shape[:2]
            )

            full_crop_path = out_dir / "crops" / f"{obj_id}_rgba.png"
            if not full_crop_path.exists():
                full_crop_path = out_dir / "crops" / f"{obj_id}_reproj.png"

            object_space_path = out_dir / "object_space" / f"{obj_id}.glb"

            if not os.path.exists(object_space_path):
                print(f"Missing reconstruction: {obj_id}")
                continue

            try:
                obj_mesh = trimesh.load(object_space_path)
                if isinstance(obj_mesh, trimesh.Scene):
                    dumped = obj_mesh.dump()
                    if len(dumped) == 0:
                        print(f"Empty scene mesh: {obj_id}")
                        continue
                    obj_mesh = dumped[0]
            except Exception as e:
                print(f"Failed loading mesh {obj_id}: {e}")
                continue

            try:
                transform = align_to_depth_match(mask, depth_map, obj_id, out_dir, mast3r_model)
            except Exception as e:
                print(f"Error aligning {obj_id}: {e}")
                continue

            try:
                obj_mesh.apply_transform(transform)
                obj_mesh.apply_transform(pose)

                convention_transform = np.array(
                    [[-1, 0, 0, 0],
                     [0, -1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]]
                )

                obj_mesh.apply_transform(convention_transform)

                (out_dir / 'reconstruction').mkdir(exist_ok=True, parents=True)
                obj_mesh.export(out_dir / 'reconstruction' / f"{obj_id}.glb")
                scene_mesh.add_geometry([obj_mesh])

                canonical_upright = (convention_transform @ transform)[:, 1]
                np.save(out_dir / 'reconstruction' / f'{obj_id}_canonical_upright.npy', canonical_upright)

                placed_count += 1
                print(f"Placed object: {obj_id}")

            except Exception as e:
                print(f"Failed placing/exporting {obj_id}: {e}")
                continue

        if placed_count == 0 or len(scene_mesh.geometry) == 0:
            print("No objects were successfully placed.")
            continue

        scene_mesh.export(out_dir / 'reconstruction' / 'full_scene.glb')

        print("Saving 3D bbox")
        save_3d_with_ground_alignment_bbox(out_dir)
        draw_cube(out_dir, is_ground=True)

        if os.path.exists(out_dir / '3dbbox_ground.json'):
            os.rename(out_dir / '3dbbox_ground.json', out_dir / '3dbbox.json')