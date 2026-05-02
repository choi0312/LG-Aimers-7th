# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATE_COL, KEY_COL, PREDICT_DAYS, ModelConfig
from .io import find_file, list_test_files, read_csv_robust, standardize_sales_frame, load_price, build_daily_meta
from .preprocess import weekday_holiday_impute, spike_clamp, apply_room_zero_fix, clamp_max_to_second_by_weekpart
from .features import merge_meta, build_feature_frame, make_train_matrix, make_test_matrix
from .model import fit_tweedie_ensemble, predict_ensemble
from .postprocess import postprocess_prediction, build_submission, median_ensemble

def apply_pipeline_preprocess(sales: pd.DataFrame, meta: pd.DataFrame, config: ModelConfig) -> pd.DataFrame:
    out = sales.copy()
    if config.apply_weekday_holiday_impute:
        out = weekday_holiday_impute(out)
    if config.apply_spike_clamp:
        out = spike_clamp(out)
    if config.apply_room_zero_fix:
        out = apply_room_zero_fix(out, meta)
    if config.apply_weekpart_second_clamp:
        out = clamp_max_to_second_by_weekpart(out)
    return out

def run_single_pipeline(data_root: str | Path, output_dir: str | Path, config: ModelConfig) -> Path:
    data_root = Path(data_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = find_file(data_root, ["train/train.csv", "train.csv"])
    sample_path = find_file(data_root, ["sample_submission.csv"])

    train_raw = standardize_sales_frame(read_csv_robust(train_path))
    train_meta = build_daily_meta(data_root)
    train = apply_pipeline_preprocess(train_raw, train_meta, config)
    price = load_price(data_root)

    keys = sorted(train[KEY_COL].unique())
    key_to_id = {key: i for i, key in enumerate(keys)}

    train_with_meta = merge_meta(train, train_meta, use_hwadam=config.use_hwadam_features)
    if not price.empty:
        train_with_meta = train_with_meta.merge(price, on=KEY_COL, how="left")

    feature_frame = build_feature_frame(train_with_meta, key_to_id, input_window=config.input_window)
    X, y, feature_cols = make_train_matrix(feature_frame, input_window=config.input_window)

    print(f"[{config.name}] train={train.shape}, X={X.shape}, y={y.shape}, features={len(feature_cols)}")
    models = fit_tweedie_ensemble(X, y, config)

    prediction_dict = {}
    for test_path in list_test_files(data_root):
        test_id = test_path.stem
        test_raw = standardize_sales_frame(read_csv_robust(test_path))
        test_meta = build_daily_meta(data_root, test_id=test_id)
        merged_meta = pd.concat([train_meta, test_meta], ignore_index=True).drop_duplicates(DATE_COL).sort_values(DATE_COL)
        test = apply_pipeline_preprocess(test_raw, merged_meta, config)

        history = pd.concat([train, test], ignore_index=True)
        history_with_meta = merge_meta(history, merged_meta, use_hwadam=config.use_hwadam_features)
        if not price.empty:
            history_with_meta = history_with_meta.merge(price, on=KEY_COL, how="left")

        test_feature_frame = build_feature_frame(history_with_meta, key_to_id, input_window=config.input_window)
        X_test, pred_keys = make_test_matrix(test_feature_frame, test, feature_cols)
        pred = predict_ensemble(models, X_test)
        pred = postprocess_prediction(pred, pred_keys, test, config)

        prediction_dict[test_id] = pd.DataFrame(
            pred,
            index=pred_keys,
            columns=[f"h{h}" for h in range(1, PREDICT_DAYS + 1)],
        )
        print(f"[{config.name}] {test_id}: {prediction_dict[test_id].shape}")

    sample = read_csv_robust(sample_path)
    submission = build_submission(sample, prediction_dict)
    out_path = output_dir / f"submission_{config.name}.csv"
    submission.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[{config.name}] saved: {out_path}")
    return out_path

def default_configs(n_estimators: int = 1200) -> tuple[ModelConfig, ModelConfig]:
    pipeline_1 = ModelConfig(
        name="pipeline_1_tweedie_weather_group_ski",
        input_window=35,
        seed=42,
        n_estimators=n_estimators,
        tweedie_powers=(1.10, 1.30, 1.50),
        apply_weekday_holiday_impute=True,
        apply_spike_clamp=True,
        apply_room_zero_fix=False,
        apply_weekpart_second_clamp=False,
        use_hwadam_features=False,
        round_threshold=0.130,
    )
    pipeline_2 = ModelConfig(
        name="pipeline_2_room_hwadam_clamp",
        input_window=35,
        seed=777,
        n_estimators=n_estimators,
        tweedie_powers=(1.20, 1.40, 1.60),
        apply_weekday_holiday_impute=True,
        apply_spike_clamp=True,
        apply_room_zero_fix=True,
        apply_weekpart_second_clamp=True,
        use_hwadam_features=True,
        round_threshold=0.130,
    )
    return pipeline_1, pipeline_2

def run_full_pipeline(data_root: str | Path, output_dir: str | Path, n_estimators: int = 1200) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg1, cfg2 = default_configs(n_estimators=n_estimators)
    path1 = run_single_pipeline(data_root, output_dir, cfg1)
    path2 = run_single_pipeline(data_root, output_dir, cfg2)
    final = median_ensemble(path1, path2, output_dir / "submission_median_ensemble.csv")
    print(f"[final] saved: {final}")
    return final
