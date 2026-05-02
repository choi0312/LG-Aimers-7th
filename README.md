# LG Aimers 7th — F&B Menu Demand Forecasting

> **LG Aimers 7기 식음업장 메뉴 수요 예측 AI 오프라인 해커톤**  
> Menu-level demand forecasting for resort F&B outlets using structured time-series modeling

## Competition Result

| 항목 | 내용 |
|---|---|
| Competition | LG Aimers 7기 식음업장 메뉴 수요 예측 AI 오프라인 해커톤 |
| Host | **LG AI Research** |
| Organizer | DACON |
| Participating Organization | Hankyung.com |
| Final Result | **3rd Place — 한경닷컴 사장상** |
| Task Type | Tabular Time-series Forecasting, Multi-step Demand Forecasting |
| Objective | Forecast 7-day menu-level sales quantity for each F&B outlet-menu pair |
| Evaluation | Composite metric based on SMAPE, NMAE, NRMSE, and Pearson R² |

## 1. Overview

This repository contains a modular forecasting pipeline developed for the **LG Aimers 7th F&B menu demand forecasting competition**. The objective is to predict the next 7 days of sales quantity for each `영업장명_메뉴명` pair using recent menu-level sales history and resort operation metadata.

The solution is designed around the characteristics of real F&B demand data: intermittent sales, sparse menu demand, weekday/weekend seasonality, holiday spikes, weather sensitivity, visitor traffic, room occupancy, ski resort inflow, and group purchase effects. Instead of relying on a single black-box model, the pipeline combines robust feature engineering, LightGBM Tweedie regression, direct multi-output forecasting, and median ensembling.

## 2. Data Schema

The competition dataset is expected to follow the structure below. Raw data is intentionally excluded from this repository.

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

| File | Description | Main Columns |
|---|---|---|
| `train.csv` | Historical menu-level sales quantity | `영업일자`, `영업장명_메뉴명`, `매출수량` |
| `price.csv` | Average selling price by outlet-menu item | `영업장명_메뉴명`, `평균판매금액` |
| `room_type.csv` | Resort room type metadata | `객실타입`, `객실타입명`, `기준인원` |
| `TRAIN_group.csv` | Group purchase counts by outlet | `영업일자`, outlet-level group payment columns |
| `TRAIN_hwadam.csv` | Hwadam Forest, Hwadamchae, monorail visitors | `영업일자`, visitor columns |
| `TRAIN_room.csv` | Daily room sales records | `영업일자`, room type columns |
| `TRAIN_ski.csv` | Hourly ski resort visitor tagging records | `영업일자`, hourly visitor columns, `1일 내장객` |
| `TRAIN_weather.csv` | Icheon area weather records | `일시`, temperature and precipitation columns |
| `TEST_00.csv` ~ `TEST_09.csv` | 28-day sales blocks for 2025 test periods | `영업일자`, `영업장명_메뉴명`, `매출수량` |
| `sample_submission.csv` | Submission template | `TEST_xx+1일` ~ `TEST_xx+7일` rows and menu columns |

## 3. Modeling Strategy

### 3.1 Forecasting Formulation

The task is formulated as **direct 7-day multi-output forecasting**. For each outlet-menu item, the model observes the latest 28-day sales window and predicts sales quantity for horizons `t+1` to `t+7`.

```text
Input  : recent item-level sales window + calendar features + operation metadata
Output : [sales(t+1), sales(t+2), ..., sales(t+7)]
Model  : LightGBM Tweedie Regressor wrapped by MultiOutputRegressor
Final  : Median ensemble of two independently configured pipelines
```

### 3.2 Feature Engineering

| Feature Group | Examples | Purpose |
|---|---|---|
| Lag features | `lag_0`, `lag_1`, `lag_7`, `lag_14`, `lag_21`, `lag_27` | Recent demand and weekly recurrence |
| Rolling statistics | rolling mean, median, std, min, max, sum | Local level and volatility |
| Intermittent demand | zero ratio, nonzero mean, last nonzero gap | Sparse demand stabilization |
| Trend features | 7/14/28-day slope | Short-term increasing/decreasing trend |
| Calendar features | weekday, weekend, holiday, month-end, cyclic encoding | Calendar seasonality |
| Price features | average selling price | Scale correction by menu price level |
| Room metadata | room sales and capacity proxy | Accommodation-driven F&B demand |
| Visitor metadata | Hwadam, monorail, ski resort inflow | External traffic-driven demand |
| Weather metadata | temperature, precipitation | Weather-sensitive resort demand |
| Group metadata | outlet-level group purchase counts | Group-order and event demand |

### 3.3 Robust Post-processing

The pipeline includes several post-processing rules for practical demand forecasting.

1. Non-negative clipping for all predictions
2. Sparse-menu protection for items with long zero-sales intervals
3. Holiday spike smoothing for abnormal weekday holiday outliers
4. Weekpart-aware clipping to reduce unrealistic weekday/weekend predictions
5. Median ensemble to reduce model-specific prediction variance

## 4. Repository Structure

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── docs/
│   └── data_schema.md
├── scripts/
│   ├── run_training.py
│   └── ensemble_submissions.py
└── src/
    └── lg_aimers_7th/
        ├── __init__.py
        ├── config.py
        ├── data_io.py
        ├── features.py
        ├── modeling.py
        ├── postprocess.py
        └── pipeline.py
```

## 5. Installation

```bash
pip install -r requirements.txt
```

## 6. Usage

After preparing the competition data under `data/`, run:

```bash
python scripts/run_training.py \
  --data-root data \
  --output-dir outputs \
  --n-estimators 1200
```

The final submission file will be saved as:

```text
outputs/submission_median_ensemble.csv
```

To ensemble two existing submission files:

```bash
python scripts/ensemble_submissions.py \
  --path-a outputs/submission_pipeline_a.csv \
  --path-b outputs/submission_pipeline_b.csv \
  --out-path outputs/submission_median_ensemble.csv
```

## 7. Key Contributions

- Designed a production-style modular pipeline for menu-level time-series demand forecasting
- Integrated item-level sales history with external resort operation metadata
- Applied Tweedie regression to handle non-negative, count-like, right-skewed demand
- Stabilized intermittent demand through sparse-item-aware post-processing
- Reduced submission variance through independent pipeline configuration and median ensembling

## 8. Award

This project achieved **3rd place** in the LG Aimers 7th offline hackathon and received the **Hankyung.com President Award**.
