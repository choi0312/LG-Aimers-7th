# LG Aimers 7th — Menu Demand Forecasting for F&B Outlets at Gonjiam Resort

> **LG Aimers 7기 식음업장 메뉴 수요 예측 AI 오프라인 해커톤**  
> Resort F&B Outlet Menu-level Demand Forecasting with Time-series Tabular Modeling

## Competition Result

| 항목 | 내용 |
|---|---|
| 대회명 | 식음업장 메뉴 수요 예측 AI 오프라인 해커톤 |
| 주최 | **LG AI Research** |
| 주관 | DACON |
| 참여 | Hankyung.com |
| 최종 성과 | **최종 3등, 한경닷컴 사장상** |
| 문제 유형 | 정형 데이터 기반 시계열 수요 예측, Multi-step Forecasting, Menu-level Demand Forecasting |
| 예측 목표 | 최근 최대 28일의 메뉴별 판매 이력과 외부 메타 정보를 활용하여 향후 7일간 메뉴별 매출 수량 예측 |
| 주요 지표 | SMAPE, NMAE, NRMSE, Pearson R² 기반 종합 평가 |

## 1. Project Overview

본 프로젝트는 리조트 내 식음업장의 메뉴별 판매량을 예측하기 위한 실전형 수요 예측 파이프라인입니다. 리조트 F&B 수요는 계절성, 요일 효과, 휴일, 객실 투숙 규모, 화담숲/스키장 방문객, 단체 결제, 기상 조건 등 다양한 외생 변수에 의해 급격히 변동합니다. 본 솔루션은 단순 시계열 예측이 아니라 **메뉴 단위 판매 이력 + 리조트 운영 메타데이터 + 캘린더 이벤트 + 강건한 후처리**를 결합한 정형 시계열 예측 시스템으로 설계되었습니다.

최종 제출 전략은 두 개의 LightGBM Tweedie 기반 파이프라인을 독립적으로 구성한 뒤, 제출 파일 단위의 median ensemble을 적용하는 방식입니다. 이를 통해 특정 모델의 과대/과소 예측 리스크를 완화하고, sparse menu 및 intermittent demand가 존재하는 메뉴 수요 예측 문제에서 안정적인 제출값을 생성했습니다.

## 2. Dataset Summary

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

| 데이터 | 설명 | 주요 컬럼 |
|---|---|---|
| `train.csv` | 2023.01.01 ~ 2024.06.15 메뉴별 매출 수량 | 영업일자, 영업장명_메뉴명, 매출수량 |
| `price.csv` | 메뉴별 평균 판매금액 | 영업장명_메뉴명, 평균판매금액 |
| `room_type.csv` | 객실 타입별 기준 인원 | 객실타입, 객실타입명, 기준인원 |
| `TRAIN_group.csv` | 영업장별 단체 결제 수 | 영업일자, 영업장별 단체 결제 컬럼 |
| `TRAIN_hwadam.csv` | 화담숲/화담채/모노레일 방문객 수 | 영업일자, 화담숲, 화담채, 모노레일 |
| `TRAIN_room.csv` | 객실 판매 실적 | 영업일자, 객실 타입별 판매량 |
| `TRAIN_ski.csv` | 스키장 시간대별 내장객 수 | 영업일자, 시간대별 태깅 수, 1일 내장객 |
| `TRAIN_weather.csv` | 이천 지역 기상 정보 | 일시, 평균기온, 최고기온, 최저기온, 강수량 |
| `TEST_00.csv` ~ `TEST_09.csv` | 2025년 특정 시점의 최근 28일 메뉴별 판매량 | 영업일자, 영업장명_메뉴명, 매출수량 |
| `sample_submission.csv` | 제출 양식 | TEST_xx+1일 ~ TEST_xx+7일, 메뉴별 예측값 |

## 3. Modeling Strategy

### 3.1 Multi-step Forecasting Formulation

각 `영업장명_메뉴명`을 독립적인 item time-series로 보고, 기준일 `t`에서의 최근 판매 패턴과 외부 메타 피처를 이용해 `t+1`부터 `t+7`까지의 매출 수량을 직접 예측하는 **direct multi-output forecasting** 구조를 사용했습니다.

- 입력: 메뉴별 최근 판매 이력, lag/rolling/EWM/trend 피처, 캘린더 피처, 외부 운영 메타데이터
- 출력: `horizon=7`의 다중 타깃 벡터
- 모델: LightGBM Tweedie Regressor + MultiOutputRegressor
- 앙상블: Tweedie variance power 및 seed variation 기반 예측 평균, 최종 제출 파일 median ensemble

### 3.2 Feature Engineering

| 구분 | 피처 예시 | 목적 |
|---|---|---|
| 판매 이력 | lag 0/1/2/3/6/7/14/21/27 | 최근 수요, 주간 반복성, 장기 기억 반영 |
| 이동 통계 | rolling mean/std/min/max, EWM | 변동성 및 추세 안정화 |
| 간헐 수요 | zero ratio, nonzero mean, long-zero-block flag | sparse menu의 과대 예측 방지 |
| 추세 | 최근 7/14/28일 slope | 단기 상승/하락 패턴 반영 |
| 캘린더 | 요일, 주말, 월말, 공휴일, horizon별 요일 | 리조트 방문 패턴 및 운영 주기 반영 |
| 가격 | 평균 판매금액 | 메뉴 단가에 따른 수요 규모 차이 보정 |
| 객실 | 객실 타입별 판매량 × 기준인원 기반 총 방문객 proxy | 투숙 규모와 F&B 수요의 연계 반영 |
| 화담숲 | 화담숲/화담채/모노레일 방문객 | 특정 업장 수요 변동 요인 반영 |
| 스키장 | 시간대별 내장객, 1일 내장객 | 계절성/레저 방문객 수요 반영 |
| 날씨 | 평균/최고/최저 기온, 강수량 | 야외 활동 및 방문객 수요 변동 반영 |
| 단체 결제 | 영업장별 단체 결제 수 | 대량 주문 및 단체 수요 이벤트 반영 |

### 3.3 Robust Preprocessing and Post-processing

실제 판매 데이터에는 휴일/이벤트성 급등, 미판매 구간, sparse menu, test block별 스케일 차이가 존재합니다. 따라서 다음과 같은 안전장치를 적용했습니다.

1. **평일 공휴일 보정**: 평일 공휴일의 비정상 급등값을 동일 요일 rolling median 기반으로 완화  
2. **주중/주말 분리 통계**: test block 내부에서 주중/주말 수요 분포를 분리하여 극단값 영향을 완화  
3. **음수 예측 차단**: Tweedie 예측 후 모든 값을 0 이상으로 clip  
4. **정수 제출 보정**: 대회 제출 형식에 맞게 threshold rounding 적용  
5. **최소 수요 보정**: 평가 산식과 sparse item 특성을 고려하여 제출값의 0 과소 예측을 제한  
6. **Median Ensemble**: 독립 파이프라인의 제출값을 ID/컬럼 스키마 기준으로 정렬 후 median 결합

## 4. Repository Structure

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   └── lg_aimers_demand.py
├── scripts/
│   ├── run_lgbm_ensemble.py
│   └── median_ensemble.py
└── outputs/
    ├── submission_pipeline_a.csv
    ├── submission_pipeline_b.csv
    └── submission_median_ensemble.csv
```

## 5. How to Run

### 5.1 Install Dependencies

```bash
pip install -r requirements.txt
```

### 5.2 Run Full Pipeline

```bash
python scripts/run_lgbm_ensemble.py \
  --data-root /path/to/dacon/data \
  --output-dir outputs \
  --n-estimators 1200
```

### 5.3 Run Median Ensemble Only

```bash
python scripts/median_ensemble.py \
  --path-a outputs/submission_pipeline_a.csv \
  --path-b outputs/submission_pipeline_b.csv \
  --out-path outputs/submission_median_ensemble.csv
```

## 6. Technical Highlights

- **Competition-specific feature design**: 리조트 운영 메타데이터를 단순 부가정보가 아니라 메뉴 수요 변동을 설명하는 외생 변수로 통합
- **Direct 7-day multi-output forecasting**: recursive forecasting에서 발생할 수 있는 오차 누적을 방지
- **Tweedie objective**: 0이 많고 양의 정수값이 long-tail 형태를 보이는 매출 수량 데이터에 적합
- **Block-wise inference**: `TEST_00` ~ `TEST_09` 각각의 28일 history를 독립 block으로 처리
- **Schema-safe ensemble**: 제출 파일의 ID와 메뉴 컬럼을 기준으로 정렬 후 결합하여 제출 형식 오류 방지
- **Reproducible Colab workflow**: 데이터 준비, 학습, 추론, README 갱신, GitHub push까지 단일 Colab 노트북에서 수행 가능

## 7. Award

본 프로젝트는 **LG AI Research가 주최한 LG Aimers 7기 식음업장 메뉴 수요 예측 AI 오프라인 해커톤에서 최종 3등을 수상**했으며, **한경닷컴 사장상**을 수상했습니다.
