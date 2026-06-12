"""
Unified model wrappers for all external models used in the pipeline.

Models included:
- Reconstruction: Amodal3R (the only supported backend)
- Completion: Amodal completion

Segmentation is handled exclusively by SAM3 (see src/sam3_seg_for_la3d.py and
src/sam3_web_interactive.py); no in-pipeline segmentation wrappers live here.
Depth (MoGe) is invoked directly by batch_scripts/depth.py, not through here.
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
