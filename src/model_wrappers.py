"""
Unified model wrappers for all external models used in the pipeline.

Models included:
- Reconstruction: TRELLIS, Hunyuan3D
- Depth: MoGe, DepthPro
- Enhancement: InvSR
- Completion: Amodal completion

Segmentation is handled exclusively by SAM3 (see src/sam3_seg_for_la3d.py and
src/sam3_web_interactive.py); no in-pipeline segmentation wrappers live here.
"""

import os
import sys
import warnings
import numpy as np

warnings.simplefilter('ignore', category=UserWarning)
warnings.simplefilter('ignore', category=FutureWarning)
warnings.simplefilter('ignore', category=DeprecationWarning)


# =============================================================================
# Lazy loading state for all models
# =============================================================================
_loaded_models = {}


def _ensure_path(external_path):
    """Ensure external path is in sys.path"""
    if external_path not in sys.path:
        sys.path.insert(0, external_path)


# =============================================================================
# TRELLIS - Image to 3D Reconstruction
# =============================================================================
def load_trellis():
    """Load TRELLIS model (lazy loading)"""
    if 'trellis' not in _loaded_models:
        _ensure_path('../external/TRELLIS')
        os.environ['ATTN_BACKEND'] = 'xformers'

        from trellis.pipelines import TrellisImageTo3DPipeline

        pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
        pipeline.cuda()
        _loaded_models['trellis'] = pipeline
        print("TRELLIS model loaded.")

    return _loaded_models['trellis']


def infer_with_trellis(out_dir, obj_id):
    """
    Run TRELLIS inference on a single object.

    Args:
        out_dir: Output directory (Path object)
        obj_id: Object identifier string

    Returns:
        Mesh object, or None if failed
    """
    from pathlib import Path
    from PIL import Image

    _ensure_path('../external/TRELLIS')
    from trellis.utils import postprocessing_utils

    print("Starting TRELLIS inference...")

    try:
        pipeline = load_trellis()

        img_path = (Path(out_dir) / "crops" / f"{obj_id}_rgba.png").absolute()
        image = Image.open(img_path)

        outputs = pipeline.run(image, seed=1)

        glb = postprocessing_utils.to_glb(
            outputs['gaussian'][0],
            outputs['mesh'][0],
            texture_size=1024,
        )
        glb.export(f"{out_dir}/object_space/{obj_id}.glb")

        print(f"TRELLIS inference complete: {out_dir}/object_space/{obj_id}.glb")
        return outputs['mesh'][0]

    except Exception as e:
        print(f"TRELLIS inference failed: {e}")
        return None


# =============================================================================
# Hunyuan3D - Image to 3D Reconstruction
# =============================================================================
def load_hunyuan3d():
    """Load Hunyuan3D models (lazy loading)"""
    if 'hunyuan3d' not in _loaded_models:
        _ensure_path('../external/Hunyuan3D-1')

        from infer import Removebg, Image2Views, Views2Mesh

        rembg_model = Removebg()
        image_to_views = Image2Views(
            device='cuda:0',
            use_lite=False,
            save_memory=False,
            std_pretrain='../external/Hunyuan3D-1/weights/mvd_std',
        )
        views_to_mesh = Views2Mesh(
            '../external/Hunyuan3D-1/svrm/configs/svrm.yaml',
            '../external/Hunyuan3D-1/weights/svrm/svrm.safetensors',
            'cuda:0',
            use_lite=False,
            save_memory=False
        )

        _loaded_models['hunyuan3d'] = {
            'rembg': rembg_model,
            'image_to_views': image_to_views,
            'views_to_mesh': views_to_mesh,
        }
        print("Hunyuan3D models loaded.")

    return _loaded_models['hunyuan3d']


def infer_with_hunyuan(out_dir, obj_id, gen_seed=0, gen_steps=50, max_faces_num=90000, do_texture_mapping=True):
    """
    Run Hunyuan3D inference on a single object.

    Args:
        out_dir: Output directory (Path object)
        obj_id: Object identifier string
        gen_seed: Random seed for generation
        gen_steps: Number of generation steps
        max_faces_num: Maximum number of faces in the output mesh
        do_texture_mapping: Whether to apply texture mapping

    Returns:
        Path to the generated GLB file, or None if failed
    """
    from pathlib import Path
    from PIL import Image
    import shutil

    print("Starting Hunyuan3D inference...")

    try:
        models = load_hunyuan3d()

        save_path = (Path(out_dir) / "object_space" / f"{obj_id}").absolute()
        img_path = (Path(out_dir) / "crops" / f"{obj_id}_rgba.png").absolute()

        os.makedirs(save_path, exist_ok=True)

        # Load input image
        res_rgb_pil = Image.open(img_path)
        res_rgb_pil.save(os.path.join(save_path, "img_nobg.png"))

        # Stage 1: Image to multi-views
        (views_grid_pil, cond_img), view_pil_list = models['image_to_views'](
            res_rgb_pil,
            seed=gen_seed,
            steps=gen_steps
        )
        views_grid_pil.save(os.path.join(save_path, "views.jpg"))

        # Stage 2: Views to mesh
        models['views_to_mesh'](
            views_grid_pil,
            cond_img,
            seed=gen_seed,
            target_face_count=max_faces_num,
            save_folder=str(save_path),
            do_texture_mapping=do_texture_mapping
        )

        # Move the output mesh to the expected location
        source_mesh = save_path / "mesh.glb"
        target_mesh = Path(out_dir) / "object_space" / f"{obj_id}.glb"

        if source_mesh.exists():
            shutil.copy(str(source_mesh), str(target_mesh))
            print(f"Hunyuan3D inference complete: {target_mesh}")
            return target_mesh
        else:
            print(f"Warning: Expected mesh not found at {source_mesh}")
            return None

    except Exception as e:
        print(f"Hunyuan3D inference failed: {e}")
        return None



# =============================================================================
# Amodal3R - Amodal Image to 3D Reconstruction
# =============================================================================
def load_amodal3r():
    """Load Amodal3R model (lazy loading)."""
    if 'amodal3r' not in _loaded_models:
        _ensure_path('../external/Amodal3R')
        # Avoid hard dependency on flash-attn wheels; xformers backend is fine.
        os.environ.setdefault('ATTN_BACKEND', 'xformers')
        os.environ.setdefault('SPCONV_ALGO', 'native')

        from amodal3r.pipelines import Amodal3RImageTo3DPipeline

        pipeline = Amodal3RImageTo3DPipeline.from_pretrained("Sm0kyWu/Amodal3R")
        pipeline.cuda()
        _loaded_models['amodal3r'] = pipeline
        print("Amodal3R model loaded.")

    return _loaded_models['amodal3r']


def infer_with_amodal3r(out_dir, obj_id, seed=1, mesh_simplify=0.95, texture_size=512):
    """
    Run Amodal3R inference on a single object.

    Notes:
    - Amodal3R expects a grayscale mask where:
      background=255, visible=188, occluded=0.
    - If *_rgba.png exists, visible region is taken from alpha.
    - Otherwise fallback to *_reproj.png and build a visible mask from non-background pixels.

    Args:
        out_dir: Output directory (Path object)
        obj_id: Object identifier string
        seed: Random seed for generation
        mesh_simplify: Mesh simplification factor for GLB export
        texture_size: Texture resolution for GLB export

    Returns:
        Path to generated GLB file, or None if failed
    """
    from pathlib import Path
    from PIL import Image
    import numpy as np

    print("Starting Amodal3R inference...")

    try:
        pipeline = load_amodal3r()

        crop_dir = Path(out_dir) / "crops"
        rgba_path = (crop_dir / f"{obj_id}_rgba.png").absolute()
        reproj_path = (crop_dir / f"{obj_id}_reproj.png").absolute()

        if rgba_path.exists():
            print(f"Using RGBA crop: {rgba_path.name}")
            image_rgba = Image.open(rgba_path).convert("RGBA")
            image_rgb = image_rgba.convert("RGB")

            alpha = np.array(image_rgba)[:, :, 3]
            visible_mask = alpha > 127

        elif reproj_path.exists():
            print(f"RGBA crop not found. Using reproj crop: {reproj_path.name}")
            reproj_img = Image.open(reproj_path).convert("RGB")
            image_rgb = reproj_img

            reproj_np = np.array(reproj_img)

            # 흰 배경(또는 거의 흰색 배경)을 background로 간주
            # 필요하면 threshold 조절 가능
            visible_mask = np.any(reproj_np < 250, axis=2)

        else:
            print(f"No crop found for {obj_id}: neither RGBA nor reproj exists.")
            return None

        # Amodal3R mask format:
        # background = 255, visible = 188, occluded = 0
        mask_np = np.full(visible_mask.shape, 255, dtype=np.uint8)
        mask_np[visible_mask] = 188
        mask_img = Image.fromarray(mask_np, mode="L")

        outputs = pipeline.run_multi_image(
            [image_rgb],
            [mask_img],
            seed=seed,
        )

        from amodal3r.utils import postprocessing_utils

        glb = postprocessing_utils.to_glb(
            outputs['gaussian'][0],
            outputs['mesh'][0],
            simplify=mesh_simplify,
            texture_size=texture_size,
            verbose=False,
        )

        target_mesh = Path(out_dir) / "object_space" / f"{obj_id}.glb"
        glb.export(str(target_mesh))
        print(f"Amodal3R inference complete: {target_mesh}")
        return target_mesh

    except Exception as e:
        print(f"Amodal3R inference failed for {obj_id}: {e}")
        return None


# =============================================================================
# MoGe - Monocular Geometry Estimation
# =============================================================================
def load_moge():
    """Load MoGe model (lazy loading)"""
    if 'moge' not in _loaded_models:
        _ensure_path('../external/MoGe')
        from infer_moge import infer_geometry_on_image as _infer_moge
        _loaded_models['moge'] = _infer_moge
        print("MoGe model loaded.")

    return _loaded_models['moge']


def infer_with_moge(image_path, out_dir):
    """
    Run MoGe inference to get depth and camera intrinsics.

    Args:
        image_path: Path to input image
        out_dir: Output directory

    Returns:
        Tuple of (points, depth_map, mask, K)
    """
    infer_fn = load_moge()
    return infer_fn(image_path, out_dir)


# =============================================================================
# DepthPro - Metric Depth Estimation
# =============================================================================
def load_depthpro(device='cuda:0'):
    """Load DepthPro model (lazy loading)"""
    import torch

    if 'depthpro' not in _loaded_models:
        import depth_pro

        model, transform = depth_pro.create_model_and_transforms(
            device=device,
            precision=torch.float16
        )
        model.eval()

        _loaded_models['depthpro'] = {
            'model': model,
            'transform': transform,
        }
        print("DepthPro model loaded.")

    return _loaded_models['depthpro']


def infer_with_depthpro(image_pil, focal_length, device='cuda:0'):
    """
    Run DepthPro inference to get metric depth.

    Args:
        image_pil: PIL Image
        focal_length: Focal length in pixels
        device: CUDA device

    Returns:
        Depth map as numpy array
    """
    models = load_depthpro(device)

    img = models['transform'](image_pil)
    prediction = models['model'].infer(img, f_px=focal_length)
    depth = prediction["depth"]

    return depth.cpu().numpy()


# =============================================================================
# InvSR - Image Super-Resolution
# =============================================================================
def load_invsr():
    """Load InvSR model (lazy loading)"""
    if 'invsr' not in _loaded_models:
        _ensure_path('../external/InvSR')
        from inference_invsr_us import get_parser, get_configs, InvSamplerSR

        args = get_parser(description="InvSR")
        configs = get_configs(args)
        sampler = InvSamplerSR(configs)

        _loaded_models['invsr'] = {
            'sampler': sampler,
            'args': args,
        }
        print("InvSR model loaded.")

    return _loaded_models['invsr']


def enhance_with_invsr(input_path, output_dir):
    """
    Run InvSR super-resolution.

    Args:
        input_path: Path to input image
        output_dir: Output directory

    Returns:
        Path to enhanced image
    """
    models = load_invsr()
    models['sampler'].inference(
        str(input_path),
        out_path=output_dir,
        bs=models['args'].bs
    )
    return output_dir / 'input.png'



# =============================================================================
# Amodal Completion (in-the-wild mode)
# =============================================================================
def complete_object(crop, label, model):
    """
    Complete occluded object regions using diffusion model.

    Args:
        crop: Cropped RGBA image
        label: Object category label
        model: Diffusion model (InstructPix2Pix)

    Returns:
        Completed RGB image
    """
    import numpy as np

    image, mask = np.split(np.array(crop) / 255, (3,), axis=-1)
    image[mask[:, :, 0] < 0.5] = 0.5
    completed = model(
        prompt=label,
        image=image,
        num_inference_steps=40,
        image_guidance_scale=1.5,
        guidance_scale=8.5,
        num_images_per_prompt=1
    ).images[0]
    return completed
