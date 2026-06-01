import json
import math
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
import sys
import importlib
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image, ImageOps
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel


APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent
RUNTIME_DIR = APP_DIR / "runtime"
UPLOAD_DIR = RUNTIME_DIR / "uploads"
JOBS_DIR = RUNTIME_DIR / "jobs"
LOGS_DIR = RUNTIME_DIR / "logs"
INTERACTIVE_DIR = RUNTIME_DIR / "interactive"
PIPELINE_SCRIPT = REPO_ROOT / "run_single_full_pipeline_parallel.sh"
# Python used to run the SAM3 interactive subprocess. SAM3 now lives in the
# same env as this server, so default to the current interpreter. Override with
# SAM_PYTHON=/path/to/python if SAM3 is installed in a separate env.
SAM_PYTHON = os.environ.get("SAM_PYTHON", sys.executable)
SAM3_WEB_SCRIPT = REPO_ROOT / "src" / "sam3_web_interactive.py"
DEFAULT_GPU_IDX = 0
DEFAULT_OBJ_REC = "amodal3r"
DEFAULT_USE_YOLO_SEG = 1
DEFAULT_ONE_TON_CAPACITY_M3 = 6.0
DEFAULT_FILL_RATE = 0.8
DEFAULT_SAM3_PROMPTS = [
    "chair", "table", "sofa", "bed", "desk", "mattress",
    "cabinet", "shelf", "drawer", "tv", "monitor",
    "refrigerator", "microwave", "washing machine",
    "oven", "bench", "furniture", "couch", "bookcase",
    "fan", "storage_box", "box", "closet",
    "air conditioner", "cooker", "wardrobe", "dresser",
    "pantry shelf", "piano", "coffee table", "low table", "television",
]

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
JOBS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
INTERACTIVE_DIR.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD_MB = 20
ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".webp"}

_jobs_lock = threading.Lock()
_jobs: Dict[str, dict] = {}
_pipeline_lock = threading.Lock()
_sam3_lock = threading.Lock()
_sam3_sessions: Dict[str, dict] = {}
_sam3_runtime: Dict[str, object] = {"model": None, "processor_cls": None, "mask_utils": None}


class BatchAggregateRequest(BaseModel):
    job_ids: List[str]


class RunCommittedRequest(BaseModel):
    """Run the downstream pipeline for a set of already-committed interactive
    SAM3 sessions. Used by the multi-image flow: interact + commit every image
    first (auto_run_pipeline=False), then process them all at once."""
    sessions: List[str]
    from_elevator: bool = True
    from_floor: int = 1
    to_elevator: bool = True
    to_floor: int = 1
    distance_km: float = 10.0


class EstimateRequest(BaseModel):
    from_elevator: bool = True
    from_floor: int = 1
    to_elevator: bool = True
    to_floor: int = 1
    distance_km: float = 10.0
    total_volume_m3: float = 0.0


class Sam3ClickRequest(BaseModel):
    x: float
    y: float
    label: int
    instance_index: Optional[int] = None
    new_label: Optional[str] = "furniture"


class Sam3SelectRequest(BaseModel):
    instance_index: int


class Sam3SetEnabledRequest(BaseModel):
    instance_index: int
    enabled: bool


def _safe_stem(name: str) -> str:
    stem = Path(name).stem
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", stem).strip("_")
    return cleaned or "image"


def _job_path(job_id: str) -> Path:
    return JOBS_DIR / f"{job_id}.json"


def _save_job(job: dict) -> None:
    with _jobs_lock:
        _jobs[job["job_id"]] = job
    _job_path(job["job_id"]).write_text(json.dumps(job, indent=2), encoding="utf-8")


def _load_job(job_id: str) -> Optional[dict]:
    with _jobs_lock:
        cached = _jobs.get(job_id)
    if cached is not None:
        return cached
    path = _job_path(job_id)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    with _jobs_lock:
        _jobs[job_id] = data
    return data


def _update_job(job_id: str, **updates) -> None:
    job = _load_job(job_id)
    if job is None:
        return
    job.update(updates)
    job["updated_at"] = time.time()
    _save_job(job)


def _extract_stage(line: str) -> Optional[str]:
    stage_map = {
        "[1/7]": "1/7 segmentation",
        "[6/7]": "6/7 reconstruction",
        "[7/7]": "7/7 scene alignment",
        "[2/7]": "2/7 depth",
        "[3/7]": "3/7 cropping",
        "[4/7]": "4/7 completion",
        "[5/7]": "5/7 elevation",
    }
    for key, value in stage_map.items():
        if key in line:
            return value
    if "=== Unified Parallel" in line:
        return "2-5/7 depth+crop+completion+elevation"
    if "TIMING SUMMARY" in line:
        return "finishing"
    return None


def _compute_summary_from_total_volume(total_volume: float, one_ton_capacity_m3: float, fill_rate: float,
                                       from_elevator: bool = True, from_floor: int = 1,
                                       to_elevator: bool = True, to_floor: int = 1,
                                       distance_km: float = 10.0) -> dict:
    recommended_ton = None
    for ton in range(1, 11):
        effective = one_ton_capacity_m3 * ton * fill_rate
        if total_volume <= effective:
            recommended_ton = ton
            break

    fallback = None
    if recommended_ton is None:
        effective_10t = one_ton_capacity_m3 * 10 * fill_rate
        fallback = math.ceil(total_volume / effective_10t) if effective_10t > 0 else None

    # 인력 추정 (부피 기준)
    if total_volume <= 10:
        workers = 2
    elif total_volume <= 20:
        workers = 3
    elif total_volume <= 35:
        workers = 4
    else:
        workers = 5

    # 추가비용 계산
    extra_costs = []
    # 톤수별 기본 비용 (포장이사 기준)
    base_price_map = {
        1: 350000, 2: 500000, 3: 700000, 4: 900000,
        5: 1300000, 6: 1500000, 7: 1700000, 8: 1900000,
        9: 2100000, 10: 2300000
    }
    ton = recommended_ton or 10
    base_price = base_price_map.get(ton, ton * 200000)  # 톤당 10만원 기준

    if not from_elevator and from_floor > 1:
        surcharge = (from_floor - 1) * 30000 * workers
        extra_costs.append({"item": f"출발지 계단 할증 ({from_floor}층, 엘리베이터 없음)", "amount": surcharge})
    if not to_elevator and to_floor > 1:
        surcharge = (to_floor - 1) * 30000 * workers
        extra_costs.append({"item": f"도착지 계단 할증 ({to_floor}층, 엘리베이터 없음)", "amount": surcharge})
    needs_ladder_truck = (not from_elevator and from_floor >= 4) or (not to_elevator and to_floor >= 4)
    if needs_ladder_truck:
        extra_costs.append({"item": "사다리차 비용", "amount": 300000})

    # 이사 거리 할증
    if distance_km > 30:
        dist_surcharge = int((distance_km - 30) / 10) * 50000
        extra_costs.append({"item": f"장거리 할증 ({int(distance_km)}km)", "amount": dist_surcharge})

    total_extra = sum(c["amount"] for c in extra_costs)
    total_price = base_price + total_extra

    return {
        "total_volume_m3": round(total_volume, 4),
        "recommended_truck_ton": recommended_ton,
        "requires_over_10t_or_multi_trucks": recommended_ton is None,
        "min_10t_trucks_needed": fallback,
        "estimated_workers": workers,
        "base_price": base_price,
        "extra_costs": extra_costs,
        "total_extra_cost": total_extra,
        "total_estimated_price": total_price,
        "needs_ladder_truck": needs_ladder_truck,
    }


def _run_sam3_subprocess(args: List[str]) -> dict:
    cmd = [SAM_PYTHON, str(SAM3_WEB_SCRIPT)] + args
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.returncode != 0:
        raise HTTPException(status_code=500, detail=f"SAM3 subprocess failed: {proc.stderr.strip() or proc.stdout.strip()}")
    out = (proc.stdout or "").strip().splitlines()
    if not out:
        raise HTTPException(status_code=500, detail="SAM3 subprocess returned empty output.")
    try:
        data = json.loads(out[-1])
    except Exception:
        raise HTTPException(status_code=500, detail=f"SAM3 subprocess invalid output: {out[-1]}")
    if not data.get("ok", False):
        raise HTTPException(status_code=409, detail=f"SAM3 interactive error: {data.get('error', 'unknown')}")
    return data


def _sam3_load_runtime(device: str = "cuda"):
    if _sam3_runtime["model"] is not None:
        return _sam3_runtime

    try:
        # Force local repo sam3 package precedence to avoid importing an unrelated
        # namespace/package named "sam3" from the environment.
        sam3_repo = str((REPO_ROOT / "sam3").resolve())
        if sam3_repo not in sys.path:
            sys.path.insert(0, sam3_repo)
        if "sam3" in sys.modules:
            del sys.modules["sam3"]

        import torch
        sam3 = importlib.import_module("sam3")
        build_sam3_image_model = getattr(sam3, "build_sam3_image_model", None)
        if build_sam3_image_model is None:
            mb = importlib.import_module("sam3.model_builder")
            build_sam3_image_model = getattr(mb, "build_sam3_image_model")
        Sam3Processor = importlib.import_module("sam3.model.sam3_image_processor").Sam3Processor
        from pycocotools import mask as mask_utils
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=(
                f"SAM3 runtime import failed in current env: {e}. "
                "Run web API in sam-capable environment."
            ),
        )

    bpe_path = os.path.join(os.path.dirname(sam3.__file__), "assets", "bpe_simple_vocab_16e6.txt.gz")
    model = build_sam3_image_model(bpe_path=bpe_path).to(device).eval()
    _sam3_runtime["model"] = model
    _sam3_runtime["processor_cls"] = Sam3Processor
    _sam3_runtime["mask_utils"] = mask_utils
    _sam3_runtime["torch"] = torch
    return _sam3_runtime


def _render_session_overlay(session: dict, save_path: Path) -> None:
    image_np = session["image_np"]
    vis = image_np.copy().astype(np.uint8)
    curr_idx = session["current_idx"]
    for idx, det in enumerate(session["detections"]):
        mask = det["mask"].astype(bool)
        if idx == curr_idx:
            color = np.array([50, 205, 50], dtype=np.uint8)
        else:
            color = np.array([255, 140, 0], dtype=np.uint8)
        vis[mask] = (0.60 * vis[mask] + 0.40 * color).astype(np.uint8)

        x0, y0, x1, y1 = [int(v) for v in det["bbox_xyxy"]]
        width = 3 if idx == curr_idx else 1
        # draw rectangle using numpy slicing to avoid cv2 dependency
        vis[max(0, y0):min(vis.shape[0], y0 + width), max(0, x0):min(vis.shape[1], x1 + 1)] = color
        vis[max(0, y1 - width + 1):min(vis.shape[0], y1 + 1), max(0, x0):min(vis.shape[1], x1 + 1)] = color
        vis[max(0, y0):min(vis.shape[0], y1 + 1), max(0, x0):min(vis.shape[1], x0 + width)] = color
        vis[max(0, y0):min(vis.shape[0], y1 + 1), max(0, x1 - width + 1):min(vis.shape[1], x1 + 1)] = color

        if idx == curr_idx:
            for p, lbl in zip(det["clicks"], det["click_labels"]):
                px, py = int(p[0]), int(p[1])
                if 0 <= py < vis.shape[0] and 0 <= px < vis.shape[1]:
                    vis[max(0, py - 3):min(vis.shape[0], py + 4), max(0, px - 3):min(vis.shape[1], px + 4)] = (
                        np.array([0, 255, 0], dtype=np.uint8) if int(lbl) == 1 else np.array([255, 0, 0], dtype=np.uint8)
                    )

    Image.fromarray(vis).save(save_path)


def _serialize_outputs(session: dict, out_json: Path, out_seg_dir: Path) -> None:
    mask_utils = _sam3_runtime["mask_utils"]
    image_np = session["image_np"]
    H, W = image_np.shape[:2]
    out_seg_dir.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    annotations = []
    categories = []
    category_to_id = {}
    seg_vis = []
    ann_id = 1
    for det in session["detections"]:
        mask_bool = det["mask"].astype(bool)
        label = det["label"]
        score = float(det["score"])
        m = mask_bool.astype(np.uint8)
        ys, xs = np.where(m > 0)
        if len(xs) == 0:
            continue
        if label not in category_to_id:
            category_to_id[label] = len(category_to_id) + 1
            categories.append({"id": category_to_id[label], "name": label, "supercategory": "object"})
        category_id = category_to_id[label]
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        bbox = [float(x1), float(y1), float(x2 - x1 + 1), float(y2 - y1 + 1)]
        area = float(m.sum())
        rle = mask_utils.encode(np.asfortranarray(m))
        counts = rle["counts"].decode("utf-8") if isinstance(rle["counts"], (bytes, bytearray)) else rle["counts"]
        annotations.append(
            {
                "id": ann_id,
                "image_id": 1,
                "category_id": category_id,
                "category_name": label,
                "bbox": bbox,
                "area": area,
                "iscrowd": 0,
                "segmentation": {"size": [H, W], "counts": counts},
                "score": score,
            }
        )
        seg_vis.append({"ann_id": ann_id, "label": label, "category_id": category_id, "bbox": bbox, "area": area, "mask": mask_bool})
        ann_id += 1

    payload = {
        "images": [{"id": 1, "file_name": session["image_name"], "width": W, "height": H}],
        "annotations": annotations,
        "categories": categories,
    }
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
        seg_index.append(
            {
                "ann_id": item["ann_id"],
                "label": item["label"],
                "category_id": item["category_id"],
                "bbox": item["bbox"],
                "area": item["area"],
                "mask_file": mask_name,
            }
        )

    Image.fromarray(overlay).save(out_seg_dir / "overlay.png")
    (out_seg_dir / "labels.json").write_text(json.dumps(seg_index, indent=2), encoding="utf-8")


def _compute_summary(bbox_path: Path, one_ton_capacity_m3: float, fill_rate: float,
                     from_elevator: bool = True, from_floor: int = 1,
                     to_elevator: bool = True, to_floor: int = 1,
                     distance_km: float = 10.0) -> dict:
    bboxes = json.loads(bbox_path.read_text(encoding="utf-8"))
    total_volume = 0.0
    for obj in bboxes:
        dims = obj.get("dimensions", [0.0, 0.0, 0.0])
        if len(dims) != 3:
            continue
        dx, dy, dz = float(dims[0]), float(dims[1]), float(dims[2])
        total_volume += dx * dy * dz

    return _compute_summary_from_total_volume(
        total_volume=total_volume,
        one_ton_capacity_m3=one_ton_capacity_m3,
        fill_rate=fill_rate,
        from_elevator=from_elevator,
        from_floor=from_floor,
        to_elevator=to_elevator,
        to_floor=to_floor,
        distance_km=distance_km,
    )


def _run_pipeline(job_id: str) -> None:
    job = _load_job(job_id)
    if job is None:
        return

    image_path = Path(job["image_path"])
    scene_name = image_path.stem
    if job.get("force_clean_scene", False):
        scene_dir = REPO_ROOT / "experimental_results" / "single" / "val" / scene_name
        if scene_dir.exists():
            for child in scene_dir.iterdir():
                # Preserve committed interactive segmentation artifacts.
                if child.name == "segmentation":
                    continue
                if child.is_dir():
                    shutil.rmtree(child, ignore_errors=True)
                else:
                    try:
                        child.unlink()
                    except Exception:
                        pass
    log_path = LOGS_DIR / f"{job_id}.log"
    cmd = ["bash", str(PIPELINE_SCRIPT), str(image_path)]

    env = os.environ.copy()
    env["GPU_IDX"] = str(job["gpu_idx"])
    env["OBJ_REC"] = str(job["obj_rec"])
    env["USE_YOLO_SEG"] = str(job["use_yolo_seg"])
    if "skip_seg" in job:
        env["SKIP_SEG"] = str(job["skip_seg"])
    if "seg_json" in job and job["seg_json"]:
        env["SEG_JSON"] = str(job["seg_json"])

    _update_job(job_id, status="queued", stage="queued (waiting turn)", log_path=str(log_path))
    with _pipeline_lock:
        _update_job(job_id, status="running", stage="starting pipeline", log_path=str(log_path))
        start = time.time()

        with log_path.open("w", encoding="utf-8") as log_fp:
            proc = subprocess.Popen(
                cmd,
                cwd=str(REPO_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )

            assert proc.stdout is not None
            for raw_line in proc.stdout:
                line = raw_line.rstrip("\n")
                log_fp.write(raw_line)
                log_fp.flush()

                stage = _extract_stage(line)
                if stage:
                    _update_job(job_id, stage=stage)

            return_code = proc.wait()

        elapsed = round(time.time() - start, 2)
    if return_code != 0:
        _update_job(
            job_id,
            status="failed",
            stage="failed",
            error=f"Pipeline exited with code {return_code}",
            elapsed_sec=elapsed,
        )
        return

    bbox_path = REPO_ROOT / "experimental_results" / "single" / "val" / scene_name / "3dbbox.json"
    if not bbox_path.exists():
        _update_job(
            job_id,
            status="failed",
            stage="failed",
            error="Pipeline completed but 3dbbox.json was not generated (no detected objects).",
            elapsed_sec=elapsed,
        )
        return

    try:
        summary = _compute_summary(
            bbox_path=bbox_path,
            one_ton_capacity_m3=float(job["one_ton_capacity_m3"]),
            fill_rate=float(job["fill_rate"]),
            from_elevator=job.get("from_elevator", True) in [True, "true", "True", 1],
            from_floor=int(job.get("from_floor", 1)),
            to_elevator=job.get("to_elevator", True) in [True, "true", "True", 1],
            to_floor=int(job.get("to_floor", 1)),
            distance_km=float(job.get("distance_km", 10.0)),
        )
    except Exception as e:
        import traceback
        _update_job(job_id, status="failed", stage="failed",
                    error=f"Summary computation failed: {e}\n{traceback.format_exc()}",
                    elapsed_sec=elapsed)
        return

    import time as _time
    job["status"] = "done"
    job["stage"] = "done"
    job["elapsed_sec"] = elapsed
    job["result"] = summary
    job["result_bbox_path"] = str(bbox_path)
    job["updated_at"] = _time.time()
    _job_path(job_id).write_text(json.dumps(job, indent=2), encoding="utf-8")
    with _jobs_lock:
        _jobs[job_id] = job


async def _create_job_from_upload(
    image: UploadFile,
    from_elevator: str,
    from_floor: int,
    to_elevator: str,
    to_floor: int,
    distance_km: float,
) -> dict:
    ext = Path(image.filename or "").suffix.lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    data = await image.read()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=400, detail=f"Image exceeds {MAX_UPLOAD_MB}MB limit.")

    job_id = time.strftime("%Y%m%d%H%M%S") + "_" + uuid.uuid4().hex[:8]
    safe_stem = _safe_stem(image.filename or "image")
    final_name = f"{job_id}_{safe_stem}{ext}"
    save_path = UPLOAD_DIR / final_name
    save_path.write_bytes(data)

    job = {
        "job_id": job_id,
        "status": "queued",
        "stage": "queued",
        "error": None,
        "result": None,
        "created_at": time.time(),
        "updated_at": time.time(),
        "image_path": str(save_path),
        "image_name": final_name,
        "gpu_idx": DEFAULT_GPU_IDX,
        "obj_rec": DEFAULT_OBJ_REC,
        "use_yolo_seg": DEFAULT_USE_YOLO_SEG,
        "one_ton_capacity_m3": DEFAULT_ONE_TON_CAPACITY_M3,
        "fill_rate": DEFAULT_FILL_RATE,
        "from_elevator": from_elevator.lower() == "true",
        "from_floor": from_floor,
        "to_elevator": to_elevator.lower() == "true",
        "to_floor": to_floor,
        "distance_km": distance_km,
    }
    _save_job(job)

    worker = threading.Thread(target=_run_pipeline, args=(job_id,), daemon=True)
    worker.start()
    return job


def _create_job_from_existing_image(
    image_path: Path,
    image_name: str,
    from_elevator: bool = True,
    from_floor: int = 1,
    to_elevator: bool = True,
    to_floor: int = 1,
    distance_km: float = 10.0,
    skip_seg: int = 0,
    seg_json: Optional[str] = None,
    force_clean_scene: bool = False,
) -> dict:
    job_id = time.strftime("%Y%m%d%H%M%S") + "_" + uuid.uuid4().hex[:8]
    job = {
        "job_id": job_id,
        "status": "queued",
        "stage": "queued",
        "error": None,
        "result": None,
        "created_at": time.time(),
        "updated_at": time.time(),
        "image_path": str(image_path),
        "image_name": image_name,
        "gpu_idx": DEFAULT_GPU_IDX,
        "obj_rec": DEFAULT_OBJ_REC,
        "use_yolo_seg": DEFAULT_USE_YOLO_SEG,
        "one_ton_capacity_m3": DEFAULT_ONE_TON_CAPACITY_M3,
        "fill_rate": DEFAULT_FILL_RATE,
        "from_elevator": from_elevator,
        "from_floor": from_floor,
        "to_elevator": to_elevator,
        "to_floor": to_floor,
        "distance_km": distance_km,
        "skip_seg": int(skip_seg),
        "seg_json": seg_json,
        "force_clean_scene": bool(force_clean_scene),
    }
    _save_job(job)
    worker = threading.Thread(target=_run_pipeline, args=(job_id,), daemon=True)
    worker.start()
    return job


app = FastAPI(title="LabelAny3D Demo API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/")
def root():
    return FileResponse(APP_DIR / "noble_logistics_web.html")


@app.post("/api/sam3/interactive/start")
async def sam3_interactive_start(
    image: UploadFile = File(...),
    prompts: str = Form(",".join(DEFAULT_SAM3_PROMPTS)),
    confidence: float = Form(0.5),
    device: str = Form("cuda"),
    use_bf16: bool = Form(True),
):
    ext = Path(image.filename or "").suffix.lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")
    data = await image.read()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=400, detail=f"Image exceeds {MAX_UPLOAD_MB}MB limit.")

    session_id = time.strftime("%Y%m%d%H%M%S") + "_" + uuid.uuid4().hex[:8]
    session_dir = INTERACTIVE_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    image_name = f"{_safe_stem(image.filename or 'image')}{ext}"
    image_path = session_dir / image_name
    image_path.write_bytes(data)

    data = _run_sam3_subprocess(
        [
            "--action", "start",
            "--session_dir", str(session_dir),
            "--image", str(image_path),
            "--prompts", prompts,
            "--confidence", str(confidence),
            "--device", device,
        ] + (["--use_bf16"] if use_bf16 else [])
    )
    return {
        "session_id": session_id,
        "num_instances": data["num_instances"],
        "current_idx": data["current_idx"],
        "overlay_url": f"/api/sam3/interactive/{session_id}/overlay",
        "instances": data["instances"],
    }


@app.get("/api/sam3/interactive/{session_id}/overlay")
def sam3_interactive_overlay(session_id: str):
    session_dir = INTERACTIVE_DIR / session_id
    overlay_path = session_dir / "overlay.png"
    if not overlay_path.exists():
        raise HTTPException(status_code=404, detail="Overlay not found.")
    return FileResponse(path=overlay_path)


@app.post("/api/sam3/interactive/{session_id}/select")
def sam3_interactive_select(session_id: str, req: Sam3SelectRequest):
    session_dir = INTERACTIVE_DIR / session_id
    data = _run_sam3_subprocess(
        [
            "--action", "select",
            "--session_dir", str(session_dir),
            "--instance_index", str(req.instance_index),
        ]
    )
    return {
        "session_id": session_id,
        "current_idx": data["current_idx"],
        "label": data["label"],
        "score": data["score"],
        "num_clicks": data["num_clicks"],
        "overlay_url": f"/api/sam3/interactive/{session_id}/overlay",
    }


@app.post("/api/sam3/interactive/{session_id}/click")
def sam3_interactive_click(session_id: str, req: Sam3ClickRequest):
    if req.label not in (0, 1):
        raise HTTPException(status_code=400, detail="label must be 0 (negative) or 1 (positive)")
    session_dir = INTERACTIVE_DIR / session_id
    cmd = [
        "--action", "click",
        "--session_dir", str(session_dir),
        "--x", str(req.x),
        "--y", str(req.y),
        "--label", str(req.label),
    ]
    if req.instance_index is not None:
        cmd += ["--instance_index", str(req.instance_index)]
    data = _run_sam3_subprocess(cmd)
    return {
        "session_id": session_id,
        "instance_index": data["instance_index"],
        "label": data["label"],
        "score": data["score"],
        "num_clicks": data["num_clicks"],
        "overlay_url": f"/api/sam3/interactive/{session_id}/overlay",
    }


@app.post("/api/sam3/interactive/{session_id}/tap")
def sam3_interactive_tap(session_id: str, req: Sam3ClickRequest):
    session_dir = INTERACTIVE_DIR / session_id
    cmd = [
        "--action", "tap",
        "--session_dir", str(session_dir),
        "--x", str(req.x),
        "--y", str(req.y),
        "--new_label", str(req.new_label or "furniture"),
    ]
    if req.instance_index is not None:
        cmd += ["--instance_index", str(req.instance_index)]
    data = _run_sam3_subprocess(cmd)
    return {
        "session_id": session_id,
        "action": data.get("action"),
        "current_idx": data.get("current_idx"),
        "num_instances": data.get("num_instances"),
        "removed_index": data.get("removed_index"),
        "toggled_index": data.get("toggled_index"),
        "added_index": data.get("added_index"),
        "instances": data.get("instances"),
        "overlay_url": f"/api/sam3/interactive/{session_id}/overlay",
    }


@app.post("/api/sam3/interactive/{session_id}/set_enabled")
def sam3_interactive_set_enabled(session_id: str, req: Sam3SetEnabledRequest):
    session_dir = INTERACTIVE_DIR / session_id
    data = _run_sam3_subprocess(
        [
            "--action", "set_enabled",
            "--session_dir", str(session_dir),
            "--instance_index", str(req.instance_index),
            "--enabled", "1" if req.enabled else "0",
        ]
    )
    return {
        "session_id": session_id,
        "instance_index": data.get("instance_index"),
        "enabled": data.get("enabled"),
        "current_idx": data.get("current_idx"),
        "instances": data.get("instances"),
        "overlay_url": f"/api/sam3/interactive/{session_id}/overlay",
    }


@app.post("/api/sam3/interactive/{session_id}/commit")
def sam3_interactive_commit(
    session_id: str,
    scene_name: str = Form(""),
    selected_indices: str = Form(""),
    auto_run_pipeline: bool = Form(True),
    from_elevator: bool = Form(True),
    from_floor: int = Form(1),
    to_elevator: bool = Form(True),
    to_floor: int = Form(1),
    distance_km: float = Form(10.0),
):
    session_dir = INTERACTIVE_DIR / session_id
    if not session_dir.exists():
        raise HTTPException(status_code=404, detail="Interactive session not found.")
    # Resolve original uploaded image path from session metadata first.
    # Do not infer from arbitrary image files in session_dir, because overlay.png
    # is also an image and can incorrectly set scene_name to "overlay".
    session_meta_path = session_dir / "session.json"
    session_image_path: Optional[Path] = None
    if session_meta_path.exists():
        try:
            meta = json.loads(session_meta_path.read_text(encoding="utf-8"))
            meta_img = Path(str(meta.get("image_path", "")))
            if meta_img.exists():
                session_image_path = meta_img
        except Exception:
            session_image_path = None
    if session_image_path is None:
        image_candidates = [
            p for p in session_dir.iterdir()
            if p.is_file() and p.suffix.lower() in ALLOWED_EXT and p.name.lower() != "overlay.png"
        ]
        if not image_candidates:
            raise HTTPException(status_code=500, detail="Committed session image not found.")
        session_image_path = sorted(image_candidates)[0]
    scene_name_final = session_image_path.stem

    out_json = REPO_ROOT / "dataset" / "coco" / "annotations" / "coconut_val.json"
    out_seg_dir = REPO_ROOT / "experimental_results" / "single" / "val" / scene_name_final / "segmentation"
    data = _run_sam3_subprocess(
        [
            "--action", "commit",
            "--session_dir", str(session_dir),
            "--out_json", str(out_json),
            "--out_seg_dir", str(out_seg_dir),
            "--selected_indices", selected_indices,
        ]
    )
    # Persist a per-session copy of the committed annotation. The shared
    # coconut_val.json gets overwritten by the next commit, so the multi-image
    # "interact all first, then process all" flow relies on this per-session
    # copy (kept in the session dir, which force_clean_scene never touches).
    committed_json = session_dir / "committed_coconut.json"
    try:
        shutil.copyfile(out_json, committed_json)
    except Exception:
        committed_json = out_json
    resp = {
        "session_id": session_id,
        "scene_name": scene_name_final,
        "saved": True,
        "out_json": str(out_json),
        "committed_json": str(committed_json),
        "out_seg_dir": str(out_seg_dir),
        "overlay": str(out_seg_dir / "overlay.png"),
        "num_instances": data["num_instances"],
    }
    if auto_run_pipeline:
        image_path = session_image_path
        job = _create_job_from_existing_image(
            image_path=image_path,
            image_name=image_path.name,
            from_elevator=from_elevator,
            from_floor=from_floor,
            to_elevator=to_elevator,
            to_floor=to_floor,
            distance_km=distance_km,
            skip_seg=1,
            seg_json=str(out_json),
            force_clean_scene=True,
        )
        resp["job_id"] = job["job_id"]
        resp["job_status"] = "queued"
    return resp


@app.post("/api/sam3/interactive/{session_id}/close")
def sam3_interactive_close(session_id: str):
    session_dir = INTERACTIVE_DIR / session_id
    existed = session_dir.exists()
    if existed:
        _run_sam3_subprocess(["--action", "close", "--session_dir", str(session_dir)])
    return {"closed": existed, "session_id": session_id}


@app.post("/api/jobs")
async def create_job(
    image: UploadFile = File(...),
    from_elevator: str = Form("true"),
    from_floor: int = Form(1),
    to_elevator: str = Form("true"),
    to_floor: int = Form(1),
    distance_km: float = Form(10.0),
):
    if not PIPELINE_SCRIPT.exists():
        raise HTTPException(status_code=500, detail="Pipeline script not found.")

    print(f"[DEBUG] from_elevator={from_elevator}, from_floor={from_floor}, to_elevator={to_elevator}, to_floor={to_floor}, distance_km={distance_km}")
    job = await _create_job_from_upload(
        image=image,
        from_elevator=from_elevator,
        from_floor=from_floor,
        to_elevator=to_elevator,
        to_floor=to_floor,
        distance_km=distance_km,
    )
    return {"job_id": job["job_id"], "status": "queued"}


@app.post("/api/jobs/batch")
async def create_jobs_batch(
    images: List[UploadFile] = File(...),
    from_elevator: str = Form("true"),
    from_floor: int = Form(1),
    to_elevator: str = Form("true"),
    to_floor: int = Form(1),
    distance_km: float = Form(10.0),
):
    if not PIPELINE_SCRIPT.exists():
        raise HTTPException(status_code=500, detail="Pipeline script not found.")
    if not images:
        raise HTTPException(status_code=400, detail="No images provided.")

    jobs = []
    for image in images:
        job = await _create_job_from_upload(
            image=image,
            from_elevator=from_elevator,
            from_floor=from_floor,
            to_elevator=to_elevator,
            to_floor=to_floor,
            distance_km=distance_km,
        )
        jobs.append({"job_id": job["job_id"], "status": "queued", "image_name": job["image_name"]})

    return {"count": len(jobs), "jobs": jobs}


@app.post("/api/jobs/run_committed")
def run_committed_jobs(req: RunCommittedRequest):
    """Queue the downstream pipeline for already-committed interactive sessions.

    The multi-image flow commits every image's segmentation first
    (auto_run_pipeline=False), collecting session ids, then calls this to run
    them all. Each job reuses the session's saved segmentation (skip_seg=1) and
    runs serialized via _pipeline_lock — safe on a single GPU."""
    if not req.sessions:
        raise HTTPException(status_code=400, detail="sessions is empty.")
    if not PIPELINE_SCRIPT.exists():
        raise HTTPException(status_code=500, detail="Pipeline script not found.")

    jobs = []
    for session_id in req.sessions:
        session_dir = INTERACTIVE_DIR / session_id
        meta_path = session_dir / "session.json"
        if not meta_path.exists():
            raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            raise HTTPException(status_code=500, detail=f"Unreadable session meta: {session_id}")
        image_path = Path(str(meta.get("image_path", "")))
        if not image_path.exists():
            raise HTTPException(status_code=500, detail=f"Image for session {session_id} not found.")
        committed_json = session_dir / "committed_coconut.json"
        if not committed_json.exists():
            raise HTTPException(
                status_code=409,
                detail=f"Session {session_id} has no committed segmentation. Commit it first.",
            )
        job = _create_job_from_existing_image(
            image_path=image_path,
            image_name=image_path.name,
            from_elevator=req.from_elevator,
            from_floor=req.from_floor,
            to_elevator=req.to_elevator,
            to_floor=req.to_floor,
            distance_km=req.distance_km,
            skip_seg=1,
            seg_json=str(committed_json),
            force_clean_scene=True,
        )
        jobs.append(
            {
                "job_id": job["job_id"],
                "status": "queued",
                "image_name": job["image_name"],
                "session_id": session_id,
            }
        )

    return {"count": len(jobs), "jobs": jobs}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = _load_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return {
        "job_id": job["job_id"],
        "status": job["status"],
        "stage": job.get("stage"),
        "error": job.get("error"),
        "elapsed_sec": job.get("elapsed_sec"),
        "created_at": job.get("created_at"),
        "updated_at": job.get("updated_at"),
    }


@app.get("/api/jobs/{job_id}/result")
def get_job_result(job_id: str):
    job = _load_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job["status"] != "done":
        raise HTTPException(status_code=409, detail=f"Job is not done. current_status={job['status']}")
    return {
        "job_id": job["job_id"],
        "status": "done",
        "total_volume_m3": job["result"]["total_volume_m3"],
        "recommended_truck_ton": job["result"]["recommended_truck_ton"],
        "requires_over_10t_or_multi_trucks": job["result"]["requires_over_10t_or_multi_trucks"],
        "min_10t_trucks_needed": job["result"]["min_10t_trucks_needed"],
        "estimated_workers": job["result"]["estimated_workers"],
        "base_price": job["result"]["base_price"],
        "extra_costs": job["result"]["extra_costs"],
        "total_extra_cost": job["result"]["total_extra_cost"],
        "total_estimated_price": job["result"]["total_estimated_price"],
        "needs_ladder_truck": job["result"]["needs_ladder_truck"],
    }


@app.post("/api/jobs/batch/aggregate")
def get_batch_aggregate(req: BatchAggregateRequest):
    if not req.job_ids:
        raise HTTPException(status_code=400, detail="job_ids is empty.")

    jobs = []
    for job_id in req.job_ids:
        job = _load_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
        jobs.append(job)

    done_jobs = [j for j in jobs if j.get("status") == "done" and j.get("result")]
    failed_jobs = [j for j in jobs if j.get("status") == "failed"]
    pending_jobs = [j for j in jobs if j.get("status") not in ["done", "failed"]]

    if pending_jobs:
        raise HTTPException(status_code=409, detail="Some jobs are not finished yet.")
    if not done_jobs:
        raise HTTPException(status_code=409, detail="No successful jobs to aggregate.")

    volume_breakdown = []
    total_volume = 0.0
    total_elapsed_sec = 0.0
    for j in done_jobs:
        vol = float(j["result"]["total_volume_m3"])
        elapsed = float(j.get("elapsed_sec") or 0.0)
        total_volume += vol
        total_elapsed_sec += elapsed
        volume_breakdown.append(
            {
                "job_id": j["job_id"],
                "image_name": j.get("image_name"),
                "volume_m3": round(vol, 4),
                "elapsed_sec": round(elapsed, 2),
            }
        )

    volume_breakdown.sort(key=lambda x: x["image_name"] or "")
    ref_job = done_jobs[0]
    summary = _compute_summary_from_total_volume(
        total_volume=total_volume,
        one_ton_capacity_m3=float(ref_job["one_ton_capacity_m3"]),
        fill_rate=float(ref_job["fill_rate"]),
        from_elevator=bool(ref_job.get("from_elevator", True)),
        from_floor=int(ref_job.get("from_floor", 1)),
        to_elevator=bool(ref_job.get("to_elevator", True)),
        to_floor=int(ref_job.get("to_floor", 1)),
        distance_km=float(ref_job.get("distance_km", 10.0)),
    )

    return {
        "total_jobs": len(jobs),
        "done_jobs": len(done_jobs),
        "failed_jobs": len(failed_jobs),
        "volume_breakdown": volume_breakdown,
        "volume_total_m3": round(total_volume, 4),
        "elapsed_total_sec": round(total_elapsed_sec, 2),
        "volume_log": " + ".join(
            [f'{v["image_name"]}: {v["volume_m3"]:.4f}m³' for v in volume_breakdown]
        ) + f" = 총 {round(total_volume, 4):.4f}m³",
        "summary": summary,
    }


@app.post("/api/estimate")
def estimate_cost(req: EstimateRequest):
    return _compute_summary_from_total_volume(
        total_volume=req.total_volume_m3,
        one_ton_capacity_m3=DEFAULT_ONE_TON_CAPACITY_M3,
        fill_rate=DEFAULT_FILL_RATE,
        from_elevator=req.from_elevator,
        from_floor=req.from_floor,
        to_elevator=req.to_elevator,
        to_floor=req.to_floor,
        distance_km=req.distance_km,
    )


@app.get("/api/jobs/{job_id}/log")
def get_job_log(job_id: str):
    job = _load_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    log_path = job.get("log_path")
    if not log_path or not Path(log_path).exists():
        raise HTTPException(status_code=404, detail="Log not ready.")
    return FileResponse(path=log_path, media_type="text/plain")


@app.post("/api/admin/cleanup")
def cleanup_runtime(max_age_hours: float = 24.0):
    cutoff = time.time() - max_age_hours * 3600.0
    removed = 0
    for path in list(JOBS_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            created_at = float(data.get("created_at", 0))
            if created_at < cutoff:
                image_path = data.get("image_path")
                log_path = data.get("log_path")
                if image_path and Path(image_path).exists():
                    Path(image_path).unlink()
                if log_path and Path(log_path).exists():
                    Path(log_path).unlink()
                path.unlink()
                removed += 1
        except Exception:
            continue

    # Best-effort cleanup for empty directories
    for p in [UPLOAD_DIR, LOGS_DIR]:
        if p.exists() and not any(p.iterdir()):
            shutil.rmtree(p)
            p.mkdir(parents=True, exist_ok=True)

    return {"removed_jobs": removed}
