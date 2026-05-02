# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DATE_COL, KEY_COL, TARGET_COL, PREDICT_DAYS
from .preprocess import add_calendar

def add_hwadam_store_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    store = out[KEY_COL].astype(str).str.rsplit("_", n=1).str[0]
    mask = store.isin({"화담숲주막", "화담숲카페"})
    for source, new_name in [("hwadam_화담숲", "hw_forest"), ("hwadam_화담채", "hw_chae"), ("hwadam_모노레일", "hw_mono")]:
        out[new_name] = 0.0
        if source in out.columns:
            out.loc[mask, new_name] = pd.to_numeric(out.loc[mask, source], errors="coerce").fillna(0.0)
    return out

def merge_meta(df: pd.DataFrame, meta: pd.DataFrame, use_hwadam: bool = False) -> pd.DataFrame:
    out = df.copy()
    if meta is not None and not meta.empty:
        out = out.merge(meta, on=DATE_COL, how="left")
    numeric_cols = [c for c in out.columns if c not in [DATE_COL, KEY_COL]]
    for c in numeric_cols:
        if c != TARGET_COL:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    if use_hwadam:
        out = add_hwadam_store_features(out)
    return out

def slope(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if values.size < 2:
        return np.nan
    x = np.arange(values.size, dtype=float)
    denom = ((x - x.mean()) ** 2).sum()
    if denom == 0:
        return 0.0
    return float(((x - x.mean()) * (values - values.mean())).sum() / denom)

def flag_long_zero_block(group: pd.DataFrame, min_days: int = 14) -> pd.Series:
    zeros = (group[TARGET_COL] == 0).astype(int)
    block_id = (zeros != zeros.shift()).cumsum()
    block_len = zeros.groupby(block_id).transform("sum")
    return ((zeros == 1) & (block_len >= min_days)).astype(int)

def build_feature_frame(df: pd.DataFrame, key_to_id: dict[str, int], input_window: int = 35) -> pd.DataFrame:
    f = df.copy().sort_values([KEY_COL, DATE_COL]).reset_index(drop=True)
    f = add_calendar(f, DATE_COL)

    for h in range(1, PREDICT_DAYS + 1):
        future = pd.DataFrame({DATE_COL: f[DATE_COL] + pd.to_timedelta(h, unit="D")})
        future = add_calendar(future, DATE_COL, prefix=f"h{h}")
        for c in future.columns:
            if c != DATE_COL:
                f[c] = future[c].to_numpy()

    for lag in [1, 2, 3, 4, 5, 6, 7, 14, 21, 27, 28, 35]:
        f[f"lag_{lag}"] = f.groupby(KEY_COL)[TARGET_COL].shift(lag)

    for window in [7, 14, 21, 28, 35]:
        shifted = f.groupby(KEY_COL)[TARGET_COL].shift(1)
        f[f"rolling_mean_{window}"] = shifted.groupby(f[KEY_COL]).rolling(window).mean().reset_index(level=0, drop=True)
        f[f"rolling_median_{window}"] = shifted.groupby(f[KEY_COL]).rolling(window).median().reset_index(level=0, drop=True)
        f[f"rolling_std_{window}"] = shifted.groupby(f[KEY_COL]).rolling(window).std().reset_index(level=0, drop=True)
        f[f"rolling_max_{window}"] = shifted.groupby(f[KEY_COL]).rolling(window).max().reset_index(level=0, drop=True)
        f[f"zero_ratio_{window}"] = (shifted == 0).groupby(f[KEY_COL]).rolling(window).mean().reset_index(level=0, drop=True)

    f["ratio_mean7_21"] = f["rolling_mean_7"] / f["rolling_mean_21"]
    f["ratio_mean14_21"] = f["rolling_mean_14"] / f["rolling_mean_21"]
    f["slope_7"] = f.groupby(KEY_COL)[TARGET_COL].transform(lambda x: x.shift(1).rolling(7).apply(slope, raw=True))
    f["slope_14"] = f.groupby(KEY_COL)[TARGET_COL].transform(lambda x: x.shift(1).rolling(14).apply(slope, raw=True))
    f["ewm_mean_7"] = f.groupby(KEY_COL)[TARGET_COL].transform(lambda x: x.shift(1).ewm(span=7, adjust=False, min_periods=2).mean())
    f["key_encoded"] = f[KEY_COL].astype(str).map(key_to_id).fillna(-1).astype(int)
    f["row_number_by_key"] = f.groupby(KEY_COL).cumcount()
    f["is_long_zero_block"] = f.groupby(KEY_COL, sort=False).apply(flag_long_zero_block).reset_index(level=0, drop=True).astype(int)

    for h in range(1, PREDICT_DAYS + 1):
        f[f"target_h{h}"] = f.groupby(KEY_COL)[TARGET_COL].shift(-h)

    f = f.replace([np.inf, -np.inf], np.nan)
    return f

def make_train_matrix(feature_frame: pd.DataFrame, input_window: int) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    target_cols = [f"target_h{h}" for h in range(1, PREDICT_DAYS + 1)]
    drop_cols = [DATE_COL, KEY_COL, TARGET_COL] + target_cols
    feature_cols = [c for c in feature_frame.columns if c not in drop_cols]
    train = feature_frame[(feature_frame["row_number_by_key"] >= input_window) & feature_frame[target_cols].notna().all(axis=1)].copy()
    X = train[feature_cols].fillna(0)
    y = train[target_cols].fillna(0)
    return X, y, feature_cols

def make_test_matrix(feature_frame: pd.DataFrame, test_history: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    cutoff = test_history[DATE_COL].max()
    rows = feature_frame[feature_frame[DATE_COL] == cutoff].copy()
    rows = rows.sort_values(KEY_COL)
    keys = rows[KEY_COL].astype(str).tolist()
    for col in feature_cols:
        if col not in rows.columns:
            rows[col] = 0
    return rows[feature_cols].fillna(0), keys
