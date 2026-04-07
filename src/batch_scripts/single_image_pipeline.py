import argparse
import json
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import trimesh
from omegaconf import OmegaConf
from PIL import Image
from scipy.ndimage import binary_opening
from sklearn.linear_model import LinearRegression, RANSACRegressor

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
# Keep behavior consistent with existing scripts that assume execution from src/.
os.chdir(SRC_DIR)

from dataset_model import get_scene
from matching.process_image_space import load_model
from model_wrappers import (
    enhance_with_invsr,
    infer_with_depthpro,
    infer_with_moge,
    run_clipseg,
    run_entityv2,
    run_ovsam,
)
from util import (
    align_to_depth_match,
    complete_crop,
    crop_object,
    depth_to_points,
    draw_cube,
    estimate_elevation,
    initialize_acompletion,
    initialize_zero123,
    restore_mask_from_crop,
)
from util_3dbox import save_3d_with_ground_alignment_bbox
from batch_scripts.reconstruction import reconstruct_object


def ensure_cropformer_path():
    """
    Ensure CropFormer module path is discoverable for run_entityv2().
    """
    candidates = [
        SRC_DIR.parent / "external" / "detectron2" / "projects" / "CropFormer",
        SRC_DIR.parent / "external" / "CropFormer",
    ]
    for path in candidates:
        if path.exists():
            path_str = str(path)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)
            return


def align_depth(relative_depth, metric_depth, mask=None, min_samples=0.2, max_valid_depth=400.0):
    regressor = RANSACRegressor(estimator=LinearRegression(fit_intercept=False), min_samples=min_samples)
    valid = (~np.isinf(relative_depth)) & (metric_depth < max_valid_depth)
    if mask is not None:
        valid &= mask
    if valid.sum() == 0:
        return metric_depth

    try:
        regressor.fit(relative_depth[valid].reshape(-1, 1), metric_depth[valid].reshape(-1, 1))
    except Exception:
        return metric_depth

    depth = np.full_like(relative_depth, 10000.0)
    if mask is not None:
        depth[mask] = regressor.predict(relative_depth[mask].reshape(-1, 1)).flatten()
    else:
        valid_mask = ~np.isinf(relative_depth)
        depth[valid_mask] = regressor.predict(relative_depth[valid_mask].reshape(-1, 1)).flatten()
    return depth


def slugify(text):
    cleaned = re.sub(r"\s+", "_", str(text).strip())
    cleaned = re.sub(r"[^0-9a-zA-Z_]+", "", cleaned)
    return cleaned or "object"


def ensure_dirs(out_dir: Path):
    out_dir.mkdir(exist_ok=True, parents=True)
    (out_dir / "crops").mkdir(exist_ok=True)
    (out_dir / "object_space").mkdir(exist_ok=True)
    (out_dir / "reconstruction").mkdir(exist_ok=True)
    (out_dir / "enhanced").mkdir(exist_ok=True)


def run_depth(scene, out_dir: Path, device: str):
    if (out_dir / "depth_map.npy").exists() and (out_dir / "cam_params.json").exists():
        return

    _, moge_depth_map, moge_mask, K_img = infer_with_moge(str(out_dir / "input.png"), out_dir)
    pro_depth_map = infer_with_depthpro(scene.image_pil, K_img[0, 0], device=device)
    depth_map = align_depth(moge_depth_map, pro_depth_map, mask=moge_mask)
    np.save(out_dir / "depth_map.npy", depth_map)

    pts3d = depth_to_points(depth_map[None], K_img)
    trimesh.PointCloud(pts3d.reshape(-1, 3), scene.image_np.reshape(-1, 3)).export(out_dir / "depth_scene.ply")

    cam_params = {
        "K": K_img.tolist(),
        "c2w": np.eye(4).tolist(),
        "W": scene.image_pil.width,
        "H": scene.image_pil.height,
    }
    with open(out_dir / "cam_params.json", "w") as fp:
        json.dump(cam_params, fp)


def run_enhance(out_dir: Path):
    enhanced_path = out_dir / "enhanced" / "input.png"
    if not enhanced_path.exists():
        enhance_with_invsr(out_dir / "input.png", out_dir / "enhanced")
    return enhanced_path


def run_auto_crops(enhanced_path: Path, original_size, out_dir: Path, crop_size: int, min_mask_area: int):
    if list((out_dir / "crops").glob("*_reproj.png")):
        return

    enhanced_image = Image.open(enhanced_path).convert("RGB")
    enhanced_np = np.array(enhanced_image)
    orig_w, orig_h = original_size
    enh_w, enh_h = enhanced_image.size
    scale_x = enh_w / float(orig_w)
    scale_y = enh_h / float(orig_h)

    ensure_cropformer_path()
    try:
        masks = run_entityv2(enhanced_np, threshold=0.1, max_size=1500)
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "Auto segmentation dependency is missing. "
            "Install/clone CropFormer under "
            "'external/detectron2/projects/CropFormer' (or 'external/CropFormer'), "
            "then retry."
        ) from e
    if len(masks) == 0:
        raise RuntimeError("EntityV2 did not return any instance masks.")

    fg_indices, _ = run_clipseg(enhanced_image, masks)
    if len(fg_indices) == 0:
        raise RuntimeError("CLIPSeg filtered out all masks.")

    selected_masks = masks[fg_indices]
    try:
        labels = run_ovsam(enhanced_image, selected_masks)
    except Exception as e:
        print(f"OVSAM skipped due to error: {e}")
        labels = None
    if labels is None:
        labels = ["object"] * len(selected_masks)

    selected_bboxes = []
    for idx in range(len(selected_masks) - 1, -1, -1):
        mask = binary_opening(selected_masks[idx], np.ones((7, 7)))
        if mask.sum() < min_mask_area:
            continue

        ys, xs = np.where(mask)
        if len(xs) == 0:
            continue
        x1, x2 = float(xs.min()), float(xs.max())
        y1, y2 = float(ys.min()), float(ys.max())
        selected_bboxes.append([x1 / scale_x, y1 / scale_y, x2 / scale_x, y2 / scale_y])

        label = labels[idx] if idx < len(labels) else "object"
        obj_id = f"{idx}_{slugify(label)}"
        crop_path = out_dir / "crops" / f"{obj_id}_reproj.png"
        crop_params_path = out_dir / "crops" / f"{obj_id}_crop_params.npy"

        crop, crop_params = crop_object(enhanced_np, mask, crop_size)
        crop.save(crop_path)
        crop_params = np.array([crop_params[0] / scale_x, crop_params[1] / scale_y, crop_params[2] * scale_x])
        np.save(crop_params_path, crop_params)

    with open(out_dir / "bboxes.json", "w") as f:
        json.dump(selected_bboxes, f)

    if len(selected_bboxes) == 0:
        raise RuntimeError("No valid objects remained after filtering.")


def run_completion(out_dir: Path, acompletion_model, run_opt):
    crop_paths = list((out_dir / "crops").glob("*_reproj.png"))
    for crop_path in crop_paths:
        obj_id = crop_path.stem.replace("_reproj", "")
        label = obj_id.split("_", 1)[-1]
        full_crop_path = out_dir / "crops" / f"{obj_id}_rgba.png"
        if full_crop_path.exists():
            continue
        crop = Image.open(crop_path)
        full_crop = complete_crop(crop, label, acompletion_model, run_opt)
        full_crop.save(full_crop_path)


def run_elevation(out_dir: Path, zero123_model):
    crop_paths = list((out_dir / "crops").glob("*_reproj.png"))
    for crop_path in crop_paths:
        obj_id = crop_path.stem.replace("_reproj", "")
        full_crop_path = out_dir / "crops" / f"{obj_id}_rgba.png"
        if not full_crop_path.exists():
            full_crop_path = out_dir / "crops" / f"{obj_id}_reproj.png"
        elevation_dir = out_dir / "object_space" / obj_id
        elevation_path = elevation_dir / "estimated_elevation.npy"
        if elevation_path.exists():
            continue
        full_crop = Image.open(full_crop_path)
        estimate_elevation(np.array(full_crop.resize((256, 256))), elevation_dir, zero123_model)


def run_reconstruction(out_dir: Path, run_opt):
    crop_paths = list((out_dir / "crops").glob("*_reproj.png"))
    for crop_path in crop_paths:
        obj_id = crop_path.stem.replace("_reproj", "")
        object_space_path = out_dir / "object_space" / f"{obj_id}.glb"
        if not object_space_path.exists():
            reconstruct_object(run_opt, out_dir, obj_id)


def run_whole(scene, out_dir: Path, mast3r_model):
    if (out_dir / "3dbbox.json").exists():
        return

    with open(out_dir / "cam_params.json", "r") as fp:
        cam_params = json.load(fp)
    pose = np.array(cam_params["c2w"])
    depth_map = np.load(out_dir / "depth_map.npy")

    scene_mesh = trimesh.Scene([None])
    crop_paths = list((out_dir / "crops").glob("*_reproj.png"))
    for crop_path in crop_paths:
        obj_id = crop_path.stem.replace("_reproj", "")
        crop_params_path = out_dir / "crops" / f"{obj_id}_crop_params.npy"
        if not crop_params_path.exists():
            continue
        crop_params = np.load(crop_params_path)
        crop = Image.open(crop_path)
        resized_mask = np.array(crop)[:, :, 3] > 127
        mask = restore_mask_from_crop(
            resized_mask, crop_params[0], crop_params[1], crop_params[2], scene.image_np.shape[:2]
        )
        object_space_path = out_dir / "object_space" / f"{obj_id}.glb"
        if not object_space_path.exists():
            continue

        obj_mesh = trimesh.load(object_space_path)
        if isinstance(obj_mesh, trimesh.Scene):
            dumped = obj_mesh.dump()
            if len(dumped) == 0:
                continue
            obj_mesh = dumped[0]

        try:
            transform = align_to_depth_match(mask, depth_map, obj_id, out_dir, mast3r_model)
        except Exception as e:
            print(f"Skipping {obj_id} due to alignment error: {e}")
            continue

        obj_mesh.apply_transform(transform)
        obj_mesh.apply_transform(pose)

        convention_transform = np.array(
            [[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        )
        obj_mesh.apply_transform(convention_transform)

        obj_mesh.export(out_dir / "reconstruction" / f"{obj_id}.glb")
        scene_mesh.add_geometry([obj_mesh])

        canonical_upright = (convention_transform @ transform)[:, 1]
        np.save(out_dir / "reconstruction" / f"{obj_id}_canonical_upright.npy", canonical_upright)

    if len(scene_mesh.geometry) == 0:
        raise RuntimeError("No reconstructed objects were aligned into the scene.")

    scene_mesh.export(out_dir / "reconstruction" / "full_scene.glb")
    save_3d_with_ground_alignment_bbox(out_dir)
    draw_cube(out_dir, is_ground=True)
    if os.path.exists(out_dir / "3dbbox_ground.json"):
        os.rename(out_dir / "3dbbox_ground.json", out_dir / "3dbbox.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True, help="Path to a single RGB image")
    parser.add_argument("--config", type=str, default="configs/image.yaml")
    parser.add_argument("--gpu_idx", type=int, default=0)
    parser.add_argument("--save_dir", type=str, default="../experimental_results/single/")
    parser.add_argument("--obj_rec", type=str, default="trellis", choices=["trellis", "hunyuan3d", "amodal3r"])
    parser.add_argument("--crop_size", type=int, default=512)
    parser.add_argument("--min_mask_area", type=int, default=6400)
    parser.add_argument(
        "--prepare_only",
        action="store_true",
        help="Run only until auto-crop generation (depth + enhance + masks/crops), then stop.",
    )

    args, extras = parser.parse_known_args()
    assert torch.cuda.is_available(), "CUDA is required for this pipeline."
    device = f"cuda:{args.gpu_idx}"

    image_path = Path(args.image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    opt = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras))
    opt.scene.attributes.img_path = str(image_path)
    opt.run.amodal_completion = "our"
    opt.run.obj_rec = args.obj_rec
    scene = get_scene(opt.scene.type, opt.scene.attributes)

    scene_name = slugify(image_path.stem)
    out_dir = Path(args.save_dir) / scene_name
    ensure_dirs(out_dir)
    scene.image_pil.save(out_dir / "input.png")
    print(f"Saving results to: {out_dir}")

    run_depth(scene, out_dir, device=device)
    enhanced_path = run_enhance(out_dir)
    run_auto_crops(
        enhanced_path=enhanced_path,
        original_size=scene.image_pil.size,
        out_dir=out_dir,
        crop_size=args.crop_size,
        min_mask_area=args.min_mask_area,
    )

    if args.prepare_only:
        print("Prepare-only mode complete.")
        print(f"Prepared directory: {out_dir}")
        print(f"Crops directory: {out_dir / 'crops'}")
        print(f"Depth map: {out_dir / 'depth_map.npy'}")
        print(f"Camera params: {out_dir / 'cam_params.json'}")
        sys.exit(0)

    acompletion_model = initialize_acompletion(device)
    run_completion(out_dir, acompletion_model, opt.run.amodal_completion)

    zero123_model = initialize_zero123(device)
    run_elevation(out_dir, zero123_model)

    run_reconstruction(out_dir, opt.run)

    mast3r_model = load_model(device)
    run_whole(scene, out_dir, mast3r_model)

    print(f"Done. Final bbox file: {out_dir / '3dbbox.json'}")
