# LG Aimers 7th — 식음업장 메뉴 수요 예측 AI

> **LG Aimers 7기 식음업장 메뉴 수요 예측 AI 오프라인 해커톤**  
> Resort F&B outlet-menu demand forecasting with modular time-series tabular ML

## Competition Result

| 항목 | 내용 |
|---|---|
| 대회 | LG Aimers 7기 식음업장 메뉴 수요 예측 AI 오프라인 해커톤 |
| 주최 | **LG AI Research** |
| 주관 | DACON |
| 참여/시상 | Hankyung.com |
| 최종 성과 | **최종 3등 — 한경닷컴 사장상** |
| 문제 유형 | 정형 시계열 예측, Multi-step Forecasting, Menu-level Demand Forecasting |
| 예측 목표 | 최근 28~35일 메뉴별 판매 이력과 리조트 운영 메타데이터를 활용한 향후 7일 매출수량 예측 |
| 사용 모델 | LightGBM Tweedie Regressor, MultiOutputRegressor, Median Ensemble |

## 1. 프로젝트 개요

본 레포지토리는 LG Aimers 7기 식음업장 메뉴 수요 예측 해커톤에서 사용한 제출 파이프라인을 포트폴리오용으로 재구성한 버전입니다.  
원본 노트북 기반의 세 가지 제출 코드, 즉 `Pipeline_1`, `Pipeline_2`, `Median Ensemble`을 하나의 재현 가능한 모듈형 구조로 정리했습니다.

리조트 F&B 수요는 단순한 일별 판매 추세만으로 설명하기 어렵습니다. 실제 수요는 요일, 주말/공휴일, 객실 판매 실적, 스키장 내장객, 화담숲 방문객, 단체 결제, 날씨, 메뉴별 판매 희소성에 의해 크게 달라집니다. 따라서 본 솔루션은 단일 모델 성능보다 **feature engineering, abnormal demand smoothing, sparse demand handling, robust ensemble**에 중점을 두었습니다.

## 2. 핵심 접근 방식

### 2.1 Pipeline 1 — Tweedie 기반 기본 시계열 모델

`Pipeline_1`은 메뉴별 판매 이력과 weather/group/ski/room 계열 메타데이터를 결합한 LightGBM Tweedie 기반 direct multi-output forecasting 파이프라인입니다.

주요 특징은 다음과 같습니다.

- 35일 input window 기반 lag feature 구성
- `lag_1~7`, `lag_14`, `lag_21`, `lag_28`, `lag_35`
- rolling mean/median/std/max 및 zero ratio
- slope, EWM 기반 단기 추세 반영
- 요일, 주말, 공휴일, 월말/분기말 calendar feature
- Tweedie variance power ensemble
- sparse menu를 고려한 non-negative post-processing

### 2.2 Pipeline 2 — ROOM/Hwadam 보정 강화 모델

`Pipeline_2`는 기본 수요 예측 구조에 room zero-day correction과 화담숲 관련 업장 feature를 추가한 보정형 파이프라인입니다.

주요 특징은 다음과 같습니다.

- 객실 판매 실적이 전부 0으로 기록된 날짜에 대한 room zero-day 보정
- 주중/주말 분리 기준의 최근 정상 판매 중앙값 대체
- 화담숲주막, 화담숲카페에 한해 화담숲/화담채/모노레일 방문객 feature 반영
- 주중/주말별 extreme max value를 둘째/셋째 큰 값으로 clamp
- holiday spike 및 intermittent demand 안정화

### 2.3 Median Ensemble

마지막 제출 단계에서는 두 파이프라인의 제출 파일을 ID 기준으로 정렬한 뒤 median ensemble을 적용했습니다.

```text
submission_final = median(submission_pipeline_1, submission_pipeline_2)
```

median ensemble은 특정 파이프라인의 과대 예측 또는 특정 이벤트성 예측 오류에 덜 민감하다는 장점이 있습니다.

## 3. 데이터 구조

원본 대회 데이터는 GitHub에 포함하지 않습니다. 실행 시 아래 구조로 `data/` 폴더를 구성하면 됩니다.

```text
data/
├── train/
│   ├── train.csv
│   ├── price.csv
│   ├── room_type.csv
│   ├── Map.jpg
│   └── meta/
│       ├── TRAIN_group.csv
│       ├── TRAIN_hwadam.csv
│       ├── TRAIN_room.csv
│       ├── TRAIN_ski.csv
│       └── TRAIN_weather.csv
├── test/
│   ├── TEST_00.csv ... TEST_09.csv
│   └── meta/
│       ├── TEST_group_00.csv ... TEST_group_09.csv
│       ├── TEST_hwadam_00.csv ... TEST_hwadam_09.csv
│       ├── TEST_room_00.csv ... TEST_room_09.csv
│       ├── TEST_ski_00.csv ... TEST_ski_09.csv
│       └── TEST_weather_00.csv ... TEST_weather_09.csv
└── sample_submission.csv
```

## 4. Repository Structure

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── docs/
│   └── pipeline_summary.md
├── scripts/
│   ├── run_pipeline.py
│   └── median_ensemble.py
└── src/
    └── lg_aimers_7th/
        ├── __init__.py
        ├── config.py
        ├── io.py
        ├── preprocess.py
        ├── features.py
        ├── model.py
        ├── postprocess.py
        └── pipeline.py
```

## 5. 실행 방법

### 5.1 패키지 설치

```bash
pip install -r requirements.txt
```

### 5.2 전체 파이프라인 실행

```bash
python scripts/run_pipeline.py \
  --data-root data \
  --output-dir outputs \
  --n-estimators 1200
```

최종 제출 파일은 아래 경로에 저장됩니다.

```text
outputs/submission_median_ensemble.csv
```

### 5.3 이미 생성된 두 제출 파일 앙상블

```bash
python scripts/median_ensemble.py \
  --path-a outputs/submission_pipeline_1_tweedie_weather_group_ski.csv \
  --path-b outputs/submission_pipeline_2_room_hwadam_clamp.csv \
  --out-path outputs/submission_median_ensemble.csv
```

## 6. 재현성 관련 메모

현재 레포지토리는 대회 데이터 없이도 코드 구조와 모듈 구성이 확인 가능하도록 정리되어 있습니다.  
실제 학습/추론 재현을 위해서는 DACON에서 제공된 원본 데이터 파일을 `data/` 경로에 배치해야 합니다.

경로는 모두 상대경로 기반으로 작성되어 있어 Colab, 로컬, GitHub Codespaces 환경에서 동일한 구조로 실행할 수 있습니다.

## 7. Technical Highlights

- Direct 7-day multi-output forecasting
- LightGBM Tweedie objective for non-negative skewed demand
- Weather, room, ski, group, Hwadam metadata integration
- Weekday holiday imputation and spike smoothing
- Room zero-day correction
- Sparse demand guard
- Median ensemble for robust final submission

## 8. Award

본 프로젝트는 **LG AI Research 주최 LG Aimers 7기 오프라인 해커톤에서 최종 3등을 기록하여 한경닷컴 사장상을 수상**했습니다.
