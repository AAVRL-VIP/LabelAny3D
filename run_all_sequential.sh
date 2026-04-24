#!/usr/bin/env bash

echo "순차적 파이프라인 처리를 시작합니다 (photo_003 ~ photo_014)"

# 1부터 14까지 반복
for i in {1..14}; do
  # 숫자를 001, 014 형태의 3자리 문자열로 변환
  NUM=$(printf "%03d" $i)
  
  echo "================================================="
  echo "[$(date +%H:%M:%S)] 작업 시작: photo_${NUM}.jpg"
  echo "================================================="

  # & 없이 실행해야 이 명령어가 완전히 끝난 후 다음 루프가 시작됩니다.
  env USE_YOLO_SEG=1 YOLO_SEG_MODEL=yoloe-26l-seg.pt bash run_single_full_pipeline_parallel.sh /workspace/Hayeon/LabelAny3D/LabelAny3D/anon_0005/photo_${NUM}.jpg > 5_${NUM}_yolo.log 2>&1
  
  echo "[$(date +%H:%M:%S)] 작업 완료: photo_${NUM}.jpg"
done

echo "================================================="
echo "모든 이미지 처리가 완료되었습니다."