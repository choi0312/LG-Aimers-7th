# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from .config import DATE_COL, KEY_COL, TARGET_COL, PREDICT_DAYS


def get_kr_holidays(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    manual = pd.to_datetime([
        "2023-01-23", "2023-01-24", "2023-03-01", "2023-05-05", "2023-05-29", "2023-06-06", "2023-08-15",
        "2023-09-28", "2023-09-29", "2023-10-02", "2023-10-03", "2023-10-09", "2023-12-25",
        "2024-01-01", "2024-02-09", "2024-02-12", "2024-03-01", "2024-04-10", "2024-05-06", "2024-05-15",
        "2024-06-06", "2024-08-15", "2024-09-16", "2024-09-17", "2024-09-18", "2024-10-01", "2024-10-03", "2024-10-09", "2024-12-25",
        "2025-01-01", "2025-01-27", "2025-01-28", "2025-01-29", "2025-01-30", "2025-03-03", "2025-05-05", "2025-05-06",
        "2025-06-03", "2025-06-06", "2025-08-15", "2025-10-03", "2025-10-09", "2025-12-25",
    ]).normalize()

    try:
        import holidays
        years = sorted({date.year for date in pd.date_range(start, end, freq="D")})
        kr = holidays.KR(years=years)
        auto = pd.DatetimeIndex([pd.Timestamp(day) for day in kr.keys()]).normalize()
        holidays_index = pd.DatetimeIndex(sorted(set(manual) | set(auto)))
    except Exception:
        holidays_index = pd.DatetimeIndex(sorted(set(manual)))

    return holidays_index[(holidays_index >= start.normalize()) & (holidays_index <= end.normalize())]


def add_calendar_features(df: pd.DataFrame, date_col: str = DATE_COL) -> pd.DataFrame:
    output = df.copy()
    dates = pd.to_datetime(output[date_col])
    holidays_index = get_kr_holidays(dates.min() - pd.Timedelta(days=7), dates.max() + pd.Timedelta(days=14))

    output["year"] = dates.dt.year
    output["month"] = dates.dt.month
    output["day"] = dates.dt.day
    output["dayofweek"] = dates.dt.dayofweek
    output["weekofyear"] = dates.dt.isocalendar().week.astype(int)
    output["is_weekend"] = (output["dayofweek"] >= 5).astype(int)
    output["is_holiday"] = dates.dt.normalize().isin(holidays_index).astype(int)
    output["is_weekend_or_holiday"] = ((output["is_weekend"] == 1) | (output["is_holiday"] == 1)).astype(int)
    output["is_month_start"] = dates.dt.is_month_start.astype(int)
    output["is_month_end"] = dates.dt.is_month_end.astype(int)
    output["sin_dow"] = np.sin(2 * np.pi * output["dayofweek"] / 7)
    output["cos_dow"] = np.cos(2 * np.pi * output["dayofweek"] / 7)
    output["sin_month"] = np.sin(2 * np.pi * output["month"] / 12)
    output["cos_month"] = np.cos(2 * np.pi * output["month"] / 12)
    return output


def make_full_panel(train: pd.DataFrame) -> pd.DataFrame:
    dates = pd.date_range(train[DATE_COL].min(), train[DATE_COL].max(), freq="D")
    keys = sorted(train[KEY_COL].unique())
    panel = pd.MultiIndex.from_product([keys, dates], names=[KEY_COL, DATE_COL]).to_frame(index=False)
    panel = panel.merge(train, on=[KEY_COL, DATE_COL], how="left")
    panel[TARGET_COL] = panel[TARGET_COL].fillna(0).clip(lower=0)
    return panel.sort_values([KEY_COL, DATE_COL]).reset_index(drop=True)


def slope(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values), dtype=float)
    y = np.nan_to_num(values, nan=0.0)
    x = x - x.mean()
    denom = np.sum(x ** 2)
    if denom <= 0:
        return 0.0
    return float(np.sum(x * (y - y.mean())) / denom)


def build_window_features(
    history: pd.DataFrame,
    key: str,
    cutoff: pd.Timestamp,
    daily_meta: pd.DataFrame,
    price_map: Dict[str, float],
    key_encoder_map: Dict[str, int],
    max_window: int = 28,
) -> Dict[str, float]:
    item_history = history[(history[KEY_COL] == key) & (history[DATE_COL] <= cutoff)].sort_values(DATE_COL).tail(max_window)
    values = item_history[TARGET_COL].to_numpy(dtype=float)

    if len(values) < max_window:
        values = np.concatenate([np.zeros(max_window - len(values)), values])

    features: Dict[str, float] = {
        "key_id": float(key_encoder_map.get(key, -1)),
        "avg_price": float(price_map.get(key, 0.0) if pd.notna(price_map.get(key, 0.0)) else 0.0),
    }

    for lag in [0, 1, 2, 3, 4, 5, 6, 7, 14, 21, 27]:
        features[f"lag_{lag}"] = float(values[-1 - lag])

    for window in [3, 7, 14, 21, 28]:
        arr = values[-window:]
        features[f"roll_mean_{window}"] = float(np.mean(arr))
        features[f"roll_median_{window}"] = float(np.median(arr))
        features[f"roll_std_{window}"] = float(np.std(arr))
        features[f"roll_min_{window}"] = float(np.min(arr))
        features[f"roll_max_{window}"] = float(np.max(arr))
        features[f"roll_sum_{window}"] = float(np.sum(arr))
        features[f"zero_ratio_{window}"] = float(np.mean(arr == 0))
        nonzero = arr[arr > 0]
        features[f"nonzero_mean_{window}"] = float(np.mean(nonzero)) if len(nonzero) else 0.0

    for span in [3, 7, 14]:
        features[f"ewm_{span}"] = float(pd.Series(values).ewm(span=span, adjust=False).mean().iloc[-1])

    for window in [7, 14, 28]:
        features[f"slope_{window}"] = slope(values[-window:])

    features["last_nonzero_gap"] = float(next((idx for idx, val in enumerate(values[::-1]) if val > 0), max_window))
    features["all_zero_28"] = float(np.sum(values) == 0)
    features["positive_days_28"] = float(np.sum(values > 0))

    cutoff_calendar = add_calendar_features(pd.DataFrame({DATE_COL: [cutoff]}), DATE_COL)
    for col in cutoff_calendar.columns:
        if col != DATE_COL:
            features[f"cutoff_{col}"] = float(cutoff_calendar[col].iloc[0])

    if daily_meta is not None and not daily_meta.empty:
        meta_row = daily_meta[daily_meta[DATE_COL] <= cutoff].sort_values(DATE_COL).tail(1)
        if not meta_row.empty:
            for col in meta_row.columns:
                if col != DATE_COL:
                    features[f"meta_last_{col}"] = float(meta_row[col].iloc[0]) if pd.notna(meta_row[col].iloc[0]) else 0.0

    return features


def append_horizon_features(features: Dict[str, float], target_date: pd.Timestamp, horizon: int, daily_meta: pd.DataFrame) -> None:
    calendar = add_calendar_features(pd.DataFrame({DATE_COL: [target_date]}), DATE_COL)
    for col in calendar.columns:
        if col != DATE_COL:
            features[f"h{horizon}_{col}"] = float(calendar[col].iloc[0])

    if daily_meta is not None and not daily_meta.empty:
        meta_row = daily_meta[daily_meta[DATE_COL] == target_date]
        if meta_row.empty:
            meta_row = daily_meta[daily_meta[DATE_COL] <= target_date].sort_values(DATE_COL).tail(1)
        if not meta_row.empty:
            for col in meta_row.columns:
                if col != DATE_COL:
                    features[f"h{horizon}_meta_{col}"] = float(meta_row[col].iloc[0]) if pd.notna(meta_row[col].iloc[0]) else 0.0


def make_supervised_dataset(panel: pd.DataFrame, daily_meta: pd.DataFrame, price: pd.DataFrame, min_cutoff_index: int = 28):
    keys = sorted(panel[KEY_COL].unique())
    encoder = LabelEncoder().fit(keys)
    key_encoder_map = {key: int(value) for key, value in zip(encoder.classes_, encoder.transform(encoder.classes_))}
    price_map = dict(zip(price[KEY_COL], price["avg_price"])) if not price.empty else {}

    x_rows: List[Dict[str, float]] = []
    y_rows: List[List[float]] = []

    for key, group in panel.groupby(KEY_COL, sort=False):
        group = group.sort_values(DATE_COL).reset_index(drop=True)
        for idx in range(min_cutoff_index - 1, len(group) - PREDICT_DAYS):
            cutoff = group.loc[idx, DATE_COL]
            features = build_window_features(panel, key, cutoff, daily_meta, price_map, key_encoder_map)
            for horizon in range(1, PREDICT_DAYS + 1):
                append_horizon_features(features, cutoff + pd.Timedelta(days=horizon), horizon, daily_meta)
            target = group.loc[idx + 1: idx + PREDICT_DAYS, TARGET_COL].to_numpy(dtype=float)
            if len(target) == PREDICT_DAYS:
                x_rows.append(features)
                y_rows.append(target.tolist())

    X = pd.DataFrame(x_rows)
    Y = pd.DataFrame(y_rows, columns=[f"target_h{horizon}" for horizon in range(1, PREDICT_DAYS + 1)])
    feature_cols = sorted(X.columns)
    X = X[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    return X, Y, feature_cols, key_encoder_map, price_map
