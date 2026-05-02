# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .config import DATE_COL, KEY_COL, TARGET_COL, PREDICT_DAYS, PipelineConfig
from .data_io import (
    read_csv_robust,
    find_file,
    list_test_files,
    standardize_sales_frame,
    load_price,
    build_daily_meta,
)
from .features import make_full_panel, make_supervised_dataset, build_window_features, append_horizon_features
from .modeling import fit_lightgbm_tweedie_ensemble
from .postprocess import clip_holiday_spikes, postprocess_predictions, build_submission, median_ensemble


def prepare_test_matrix(
    data_root: str | Path,
    test_path: Path,
    train_panel: pd.DataFrame,
    feature_cols: List[str],
    train_daily_meta: pd.DataFrame,
    price_map: Dict[str, float],
    key_encoder_map: Dict[str, int],
) -> Tuple[pd.DataFrame, List[str], pd.DataFrame]:
    test_id = test_path.stem
    test_history = standardize_sales_frame(read_csv_robust(test_path))
    test_daily_meta = build_daily_meta(data_root, test_id=test_id)
    daily_meta = pd.concat([train_daily_meta, test_daily_meta], ignore_index=True).drop_duplicates(DATE_COL).sort_values(DATE_COL)

    history = pd.concat([train_panel[[DATE_COL, KEY_COL, TARGET_COL]], test_history], ignore_index=True)
    keys = sorted(test_history[KEY_COL].unique())
    cutoff = test_history[DATE_COL].max()

    rows = []
    for key in keys:
        features = build_window_features(history, key, cutoff, daily_meta, price_map, key_encoder_map)
        for horizon in range(1, PREDICT_DAYS + 1):
            append_horizon_features(features, cutoff + pd.Timedelta(days=horizon), horizon, daily_meta)
        rows.append(features)

    X_test = pd.DataFrame(rows)
    for col in feature_cols:
        if col not in X_test.columns:
            X_test[col] = 0
    X_test = X_test[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    return X_test, keys, test_history


def run_single_pipeline(data_root: str | Path, output_dir: str | Path, config: PipelineConfig) -> Path:
    data_root = Path(data_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = find_file(data_root, ["train/train.csv", "train.csv"])
    sample_path = find_file(data_root, ["sample_submission.csv"])

    train = standardize_sales_frame(read_csv_robust(train_path))
    if config.apply_holiday_spike_clip:
        train = clip_holiday_spikes(train)

    train_panel = make_full_panel(train)
    price = load_price(data_root)
    train_daily_meta = build_daily_meta(data_root)

    print(f"[{config.name}] train rows={len(train):,}, panel rows={len(train_panel):,}, keys={train_panel[KEY_COL].nunique():,}")
    X, Y, feature_cols, key_encoder_map, price_map = make_supervised_dataset(train_panel, train_daily_meta, price)
    print(f"[{config.name}] supervised X={X.shape}, Y={Y.shape}")

    models = fit_lightgbm_tweedie_ensemble(X, Y, config)

    prediction_dict: Dict[str, pd.DataFrame] = {}
    for test_path in list_test_files(data_root):
        test_id = test_path.stem
        X_test, keys, test_history = prepare_test_matrix(
            data_root=data_root,
            test_path=test_path,
            train_panel=train_panel,
            feature_cols=feature_cols,
            train_daily_meta=train_daily_meta,
            price_map=price_map,
            key_encoder_map=key_encoder_map,
        )

        predictions = [model.predict(X_test) for model in models]
        prediction = np.mean(predictions, axis=0)
        prediction = postprocess_predictions(prediction, keys, test_history, config)
        prediction_dict[test_id] = pd.DataFrame(prediction, index=keys, columns=[f"h{h}" for h in range(1, PREDICT_DAYS + 1)])
        print(f"[{config.name}] predicted {test_id}: {prediction_dict[test_id].shape}")

    sample = read_csv_robust(sample_path)
    submission = build_submission(sample, prediction_dict)
    output_path = output_dir / f"submission_{config.name}.csv"
    submission.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"[{config.name}] saved: {output_path}")
    return output_path


def run_ensemble_pipeline(data_root: str | Path, output_dir: str | Path, n_estimators: int = 1200, use_gpu: bool = False) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_a = PipelineConfig(
        name="pipeline_a",
        seed=42,
        n_estimators=n_estimators,
        tweedie_powers=(1.10, 1.30, 1.50),
        round_threshold=0.13,
        apply_holiday_spike_clip=True,
        apply_weekpart_max_clip=False,
        use_gpu=use_gpu,
    )

    config_b = PipelineConfig(
        name="pipeline_b",
        seed=777,
        n_estimators=n_estimators,
        learning_rate=0.035,
        num_leaves=95,
        tweedie_powers=(1.20, 1.40, 1.60),
        round_threshold=0.18,
        apply_holiday_spike_clip=True,
        apply_weekpart_max_clip=True,
        use_gpu=use_gpu,
    )

    path_a = run_single_pipeline(data_root, output_dir, config_a)
    path_b = run_single_pipeline(data_root, output_dir, config_b)
    final_path = median_ensemble(path_a, path_b, output_dir / "submission_median_ensemble.csv")
    print(f"[final] saved: {final_path}")
    return final_path
