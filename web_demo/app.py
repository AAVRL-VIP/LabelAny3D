import json
import math
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

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
PIPELINE_SCRIPT = REPO_ROOT / "run_single_full_pipeline_parallel.sh"
DEFAULT_GPU_IDX = 0
DEFAULT_OBJ_REC = "amodal3r"
DEFAULT_USE_YOLO_SEG = 1
DEFAULT_ONE_TON_CAPACITY_M3 = 6.0
DEFAULT_FILL_RATE = 0.6

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
JOBS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD_MB = 20
ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".webp"}

_jobs_lock = threading.Lock()
_jobs: Dict[str, dict] = {}
_pipeline_lock = threading.Lock()


class BatchAggregateRequest(BaseModel):
    job_ids: List[str]


class EstimateRequest(BaseModel):
    from_elevator: bool = True
    from_floor: int = 1
    to_elevator: bool = True
    to_floor: int = 1
    distance_km: float = 10.0
    total_volume_m3: float = 0.0


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
    log_path = LOGS_DIR / f"{job_id}.log"
    cmd = ["bash", str(PIPELINE_SCRIPT), str(image_path)]

    env = os.environ.copy()
    env["GPU_IDX"] = str(job["gpu_idx"])
    env["OBJ_REC"] = str(job["obj_rec"])
    env["USE_YOLO_SEG"] = str(job["use_yolo_seg"])

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

    scene_name = Path(job["image_path"]).stem
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
