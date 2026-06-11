<div align="center">

# 노블로지스 (Noble Logistics) — AI 이사 견적 데모

방 사진 한 장으로 가구의 3D 부피를 자동 측정하고 이사 견적(추천 트럭 톤수·인원·비용)을 산출하는 웹 데모.
[LabelAny3D](https://uva-computer-vision-lab.github.io/LabelAny3D/) 단일 이미지 3D 복원 파이프라인 위에 구축되었습니다.

</div>

## 1. 모델 환경

```bash
pip install -r requirements.txt
```

GPU 환경 정의: [envs/labelany3d.yml](envs/labelany3d.yml), [envs/sam.yml](envs/sam.yml)
설치 및 외부 의존성: [docs/INSTALL.md](docs/INSTALL.md)

## 2. 파이프라인 구조

<p align="center">
  <img src="images/pipeline_overview.jpg" alt="파이프라인 구조도" width="100%"/>
</p>

[run_single_full_pipeline_parallel.sh](run_single_full_pipeline_parallel.sh)는 7단계를 일부 병렬화해 실행합니다:

```
[1] 분할 (SAM3)                    ← 순차 (이후 모든 단계가 의존)
      │
      ├─ [2] 깊이 추정 ────────────────────────┐
      └─ [3] 객체 크롭 → [4] 아모달 완성        │  병렬 실행
                       → [5] 고도 추정          │
      ─────────────────────────────────────────┘  전부 대기
[6] 3D 복원 (Amodal3R)             ← 순차
[7] 장면 레이아웃 정렬             ← 순차
```

- **[1] 분할**: SAM3가 실내 가구 키워드(의자·책상·소파·냉장고 등) 프롬프트로 인스턴스 마스크 생성
- **[2] 깊이 / [3] 크롭**: 깊이 추정과 객체 크롭을 병렬 실행
- **[4] 아모달 완성 / [5] 고도 추정**: 크롭 결과를 받아 병렬 실행
- **[6] 3D 복원**: `_rgba.png`로부터 3D 메쉬 복원
- **[7] 정렬**: 개별 객체를 하나의 장면 좌표계로 정렬해 `3dbbox.json` 생성

마지막에 [src/calc_volume.py](src/calc_volume.py)가 `3dbbox.json`의 각 3D 바운딩 박스 부피를 합산합니다.

## 3. 모델 성능

방 이미지로부터 예측한 트럭 톤수와 실제 정답(GT) 비교:

| House | GT Truck Size (ton) | Model Predicted (ton) |
|-------|:-------------------:|:---------------------:|
| house_005 | 1 | 1 |
| house_010 | 3 ~ 4 | 4 |
| house_016 | 2 | 2 |

3D 바운딩 박스 복원 결과:

<p align="center">
  <img src="images/result1.jpg" alt="3D bbox 결과 1" width="48%"/>
  <img src="images/result2.jpg" alt="3D bbox 결과 2" width="48%"/>
</p>

## 4. 사용 방법

### 웹 데모

```bash
cd web_demo
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

브라우저에서 `http://localhost:8000` 접속 → 방 사진 업로드 → 이사 정보(층수·엘리베이터·거리) 입력 → 견적 확인.

웹 데모는 **대화형 분할**을 지원해, 자동 분할이 부정확할 때 클릭으로 인스턴스를 추가·제외·수정한 뒤 파이프라인을 실행할 수 있습니다(`start → click/tap/select → commit`).

### 서버에서 단독 실행

```bash
bash run_single_full_pipeline_parallel.sh /abs/path/image.jpg
```

> ⚠️ 서버 단독 실행에는 **인스턴스 제외(대화형 선택) 기능이 없습니다.** SAM3가 검출한 모든 가구 인스턴스가 그대로 3D 복원·부피 계산에 포함됩니다. 특정 객체를 빼고 계산하려면 웹 데모의 대화형 분할을 사용하세요.

주요 환경 변수:

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `GPU_IDX` | `0` | 사용할 GPU 인덱스 |
| `OBJ_REC` | `amodal3r` | 3D 복원 백엔드 |
| `MIN_MASK_AREA` | `800` | 최소 마스크 면적(px) 필터 |
| `SAM3_PROMPTS` | 실내 가구 기본 목록 | 분할 프롬프트(쉼표 구분) |
| `SAM3_CONF` | `0.5` | SAM3 신뢰도 임계값 |
| `SKIP_SEG` | `0` | `1`이면 기존 분할 JSON(`SEG_JSON`) 재사용 |
| `SAM_PYTHON` | `/opt/conda/envs/sam/bin/python` | SAM3용 파이썬 인터프리터 |

결과는 `experimental_results/single/val/<scene>/`에 저장되며, 단계별 소요 시간은 `timing.txt`, 최종 박스는 `3dbbox.json`입니다.

## 5. 인용 (LabelAny3D)

본 데모는 [LabelAny3D](https://uva-computer-vision-lab.github.io/LabelAny3D/)의 단일 이미지 3D 복원 파이프라인을 기반으로 합니다. 연구에 활용 시 아래를 인용해 주세요:

```BibTeX
@inproceedings{yao2025labelany3d,
  title={LabelAny3D: Label Any Object 3D in the Wild},
  author={Jin Yao and Radowan Mahmud Redoy and Sebastian Elbaum and Matthew B. Dwyer and Zezhou Cheng},
  booktitle={Neural Information Processing Systems (NeurIPS)},
  year={2025}
}
```

- 프로젝트 페이지: https://uva-computer-vision-lab.github.io/LabelAny3D/
- 논문: https://openreview.net/pdf?id=Q2fU0JDHuW
- 코드: https://github.com/UVA-Computer-Vision-Lab/LabelAny3D
