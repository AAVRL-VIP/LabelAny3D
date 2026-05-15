#!/usr/bin/env bash
set -u

ROOT="/workspace/Hayeon/LabelAny3D/LabelAny3D"
INPUT_DIR="$ROOT/anon_0008"
LOG_DIR="$ROOT/batch_logs"
CONDA_ENV_DIR="${CONDA_ENV_DIR:-/opt/conda/envs/la3d}"
export PATH="$CONDA_ENV_DIR/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "$LOG_DIR"

echo "순차적 파이프라인 처리를 시작합니다 (anon_0008 photo_001 ~ photo_010, 005/006 제외, Grounded-SAM-2)"

FAILED=0
for i in 1 2 3 4 7 8 9 10; do
  NUM=$(printf "%03d" "$i")
  IMAGE_PATH="$INPUT_DIR/photo_${NUM}.jpeg"
  LOG_PATH="$LOG_DIR/anon_0008_photo_${NUM}_gsam2.log"

  echo "================================================="
  echo "[$(date +%H:%M:%S)] 작업 시작: photo_${NUM}.jpeg"
  echo "로그: $LOG_PATH"
  echo "================================================="

  if [[ ! -f "$IMAGE_PATH" ]]; then
    echo "이미지가 없습니다: $IMAGE_PATH" | tee "$LOG_PATH"
    FAILED=1
    continue
  fi

  env \
    USE_YOLO_SEG=0 \
    SEG_BACKEND=gsam2 \
    MASK_REMOVE_OVERLAPS=1 \
    DEPTH_GPU_IDX="${DEPTH_GPU_IDX:-1}" \
    COMPLETION_GPU_IDX="${COMPLETION_GPU_IDX:-2}" \
    ELEVATION_GPU_IDX="${ELEVATION_GPU_IDX:-2}" \
    RECON_GPU_IDX="${RECON_GPU_IDX:-2}" \
    ALIGN_GPU_IDX="${ALIGN_GPU_IDX:-2}" \
    bash "$ROOT/run_single_full_pipeline_parallel.sh" "$IMAGE_PATH" > "$LOG_PATH" 2>&1
  RC=$?

  if [[ $RC -ne 0 ]]; then
    echo "[$(date +%H:%M:%S)] 작업 실패: photo_${NUM}.jpeg (exit $RC)"
    FAILED=1
  else
    echo "[$(date +%H:%M:%S)] 작업 완료: photo_${NUM}.jpeg"
  fi
done

echo "================================================="
if [[ $FAILED -ne 0 ]]; then
  echo "일부 이미지 처리가 실패했습니다. $LOG_DIR 로그를 확인하세요."
  exit 1
fi
echo "모든 이미지 처리가 완료되었습니다."
