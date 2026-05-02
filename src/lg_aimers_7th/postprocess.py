# -*- coding: utf-8 -*-
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from .config import DATE_COL, KEY_COL, TARGET_COL, PREDICT_DAYS, PipelineConfig
from .data_io import read_csv_robust
from .features import add_calendar_features


def clip_holiday_spikes(train: pd.DataFrame) -> pd.DataFrame:
    output = add_calendar_features(train, DATE_COL)
    clipped_groups = []

    for key, group in output.groupby(KEY_COL, sort=False):
        group = group.sort_values(DATE_COL).copy()
        baseline = group[TARGET_COL].rolling(7, min_periods=1).median().shift(1)
        mask = (group["is_holiday"] == 1) & (group["is_weekend"] == 0) & baseline.notna()
        upper = np.maximum(baseline * 2.2 + 2, baseline + 5)
        group.loc[mask, TARGET_COL] = np.minimum(group.loc[mask, TARGET_COL], upper.loc[mask])
        clipped_groups.append(group[[DATE_COL, KEY_COL, TARGET_COL]])

    return pd.concat(clipped_groups, ignore_index=True)


def postprocess_predictions(prediction: np.ndarray, keys: List[str], test_history: pd.DataFrame, config: PipelineConfig) -> np.ndarray:
    output = np.asarray(prediction, dtype=float).copy()
    output = np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)
    output = np.clip(output, 0, None)

    for row_idx, key in enumerate(keys):
        values = test_history[test_history[KEY_COL] == key].sort_values(DATE_COL)[TARGET_COL].to_numpy(dtype=float)
        if len(values) == 0:
            continue

        recent = values[-28:]
        if np.sum(recent) == 0:
            output[row_idx, :] = 0
            continue

        if np.mean(recent == 0) >= 0.90 and np.max(recent) <= 1:
            output[row_idx, :] = np.minimum(output[row_idx, :], 1)

        if config.apply_weekpart_max_clip:
            dates = pd.to_datetime(test_history[test_history[KEY_COL] == key].sort_values(DATE_COL)[DATE_COL])
            dows = dates.dt.dayofweek.to_numpy()
            weekday_max = float(np.max(values[dows < 5])) if np.any(dows < 5) else float(np.max(values))
            weekend_max = float(np.max(values[dows >= 5])) if np.any(dows >= 5) else float(np.max(values))
            global_cap = max(float(np.max(values)) * 2.0 + 3, 3.0)
            for horizon in range(PREDICT_DAYS):
                cap = weekend_max if horizon in [4, 5] else weekday_max
                output[row_idx, horizon] = min(output[row_idx, horizon], max(cap * 2.2 + 2, global_cap))

    output = np.where(output < config.round_threshold, 0, output)
    output = np.rint(output).astype(int)

    if config.force_positive_minimum and config.min_prediction > 0:
        positive_mask = output > 0
        output[positive_mask] = np.maximum(output[positive_mask], config.min_prediction)

    return output


def build_submission(sample: pd.DataFrame, prediction_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    submission = sample.copy()
    key_columns = [col for col in submission.columns if col != DATE_COL]

    for row_idx, row in submission.iterrows():
        label = str(row[DATE_COL])
        match = re.match(r"(TEST_\d{2})\+(\d+)일", label)
        if not match:
            continue

        test_id, horizon = match.group(1), int(match.group(2))
        prediction_frame = prediction_dict[test_id]

        for col in key_columns:
            submission.at[row_idx, col] = int(prediction_frame.loc[col, f"h{horizon}"]) if col in prediction_frame.index else 0

    for col in key_columns:
        submission[col] = pd.to_numeric(submission[col], errors="coerce").fillna(0).clip(lower=0).round().astype(int)

    return submission


def median_ensemble(path_a: str | Path, path_b: str | Path, out_path: str | Path) -> Path:
    submission_a = read_csv_robust(path_a)
    submission_b = read_csv_robust(path_b)

    if list(submission_a.columns) != list(submission_b.columns):
        raise ValueError("Submission columns are different.")

    if not submission_a[DATE_COL].equals(submission_b[DATE_COL]):
        submission_b = submission_b.set_index(DATE_COL).loc[submission_a[DATE_COL]].reset_index()

    output = submission_a.copy()
    value_cols = [col for col in output.columns if col != DATE_COL]
    output[value_cols] = np.median(
        np.stack([submission_a[value_cols].to_numpy(float), submission_b[value_cols].to_numpy(float)], axis=0),
        axis=0,
    )
    output[value_cols] = np.nan_to_num(output[value_cols].to_numpy(float), nan=0.0, posinf=0.0, neginf=0.0)
    output[value_cols] = np.clip(output[value_cols], 0, None).round().astype(int)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(out_path, index=False, encoding="utf-8-sig")
    return out_path
