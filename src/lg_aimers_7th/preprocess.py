# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DATE_COL, KEY_COL, TARGET_COL

def korea_holiday_index(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    manual = pd.to_datetime([
        "2023-01-01", "2023-01-23", "2023-01-24", "2023-03-01", "2023-05-05", "2023-05-29",
        "2023-06-06", "2023-08-15", "2023-09-28", "2023-09-29", "2023-10-02", "2023-10-03",
        "2023-10-09", "2023-12-25", "2024-01-01", "2024-02-09", "2024-02-12", "2024-03-01",
        "2024-04-10", "2024-05-06", "2024-05-15", "2024-06-06", "2024-08-15", "2024-09-16",
        "2024-09-17", "2024-09-18", "2024-10-01", "2024-10-03", "2024-10-09", "2024-12-25",
        "2025-01-01", "2025-01-27", "2025-01-28", "2025-01-29", "2025-01-30", "2025-03-03",
        "2025-05-05", "2025-05-06", "2025-06-03", "2025-06-06", "2025-08-15", "2025-10-03",
        "2025-10-09", "2025-12-25",
    ]).normalize()
    try:
        import holidays
        years = sorted({d.year for d in pd.date_range(start, end, freq="D")})
        kr = pd.DatetimeIndex([pd.Timestamp(x) for x in holidays.KR(years=years).keys()]).normalize()
        out = pd.DatetimeIndex(sorted(set(manual) | set(kr)))
    except Exception:
        out = pd.DatetimeIndex(sorted(set(manual)))
    return out[(out >= start.normalize()) & (out <= end.normalize())]

def add_calendar(df: pd.DataFrame, date_col: str = DATE_COL, prefix: str = "") -> pd.DataFrame:
    out = df.copy()
    dt = pd.to_datetime(out[date_col])
    holidays_idx = korea_holiday_index(dt.min() - pd.Timedelta(days=7), dt.max() + pd.Timedelta(days=14))
    p = f"{prefix}_" if prefix else ""
    out[f"{p}month"] = dt.dt.month
    out[f"{p}dayofweek"] = dt.dt.dayofweek
    out[f"{p}day"] = dt.dt.day
    out[f"{p}dayofyear"] = dt.dt.dayofyear
    out[f"{p}is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
    out[f"{p}is_holiday"] = dt.dt.normalize().isin(holidays_idx).astype(int)
    out[f"{p}is_month_end"] = dt.dt.is_month_end.astype(int)
    out[f"{p}is_quarter_end"] = dt.dt.is_quarter_end.astype(int)
    out[f"{p}dow_sin"] = np.sin(2 * np.pi * dt.dt.dayofweek / 7)
    out[f"{p}dow_cos"] = np.cos(2 * np.pi * dt.dt.dayofweek / 7)
    return out

def weighted_mean(values, dates, end_date, decay=0.90, recent_window=7, boost=0.50):
    if len(values) == 0:
        return np.nan
    values = np.asarray(values, dtype=float)
    dates = pd.to_datetime(dates)
    delta = (end_date - dates).days.astype(float)
    weights = decay ** delta
    weights = np.where(dates >= end_date - pd.Timedelta(days=recent_window - 1), weights * (1 + boost), weights)
    return float(np.sum(weights * values) / np.sum(weights))

def weekday_holiday_impute(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    hol = korea_holiday_index(df[DATE_COL].min(), df[DATE_COL].max())
    x = df.copy()
    x["dow"] = x[DATE_COL].dt.dayofweek
    x["is_weekday"] = (x["dow"] < 5).astype(int)
    x["is_weekday_hol"] = x[DATE_COL].isin(hol).astype(int)

    groups = []
    for _, g in x.groupby(KEY_COL, sort=False):
        g = g.sort_values(DATE_COL).copy()
        end_date = g[DATE_COL].max()
        candidates = g[(g["is_weekday"] == 1) & (g["is_weekday_hol"] == 0)].copy()
        all_weekday_zero = candidates[TARGET_COL].sum() == 0
        for idx, row in g[(g["is_weekday"] == 1) & (g["is_weekday_hol"] == 1)].iterrows():
            if all_weekday_zero:
                g.at[idx, TARGET_COL] = 0
                continue
            same_dow = candidates[candidates["dow"] == int(row["dow"])]
            base = weighted_mean(same_dow[TARGET_COL].values, same_dow[DATE_COL].values, end_date) if len(same_dow) else weighted_mean(candidates[TARGET_COL].values, candidates[DATE_COL].values, end_date)
            prev = g[g[DATE_COL] == row[DATE_COL] - pd.Timedelta(days=7)]
            prev_value = float(prev[TARGET_COL].iloc[0]) if len(prev) == 1 else np.nan
            if np.isfinite(base):
                g.at[idx, TARGET_COL] = int(max(0, round(base if not np.isfinite(prev_value) else 0.65 * base + 0.35 * prev_value)))
        groups.append(g.drop(columns=["dow", "is_weekday", "is_weekday_hol"]))
    return pd.concat(groups, ignore_index=True)

def spike_clamp(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    hol = korea_holiday_index(df[DATE_COL].min(), df[DATE_COL].max())
    x = df.copy()
    x["dow"] = x[DATE_COL].dt.dayofweek
    x["is_nonholiday_weekday"] = ((x["dow"] < 5) & (~x[DATE_COL].isin(hol))).astype(int)
    x["is_weekday_holiday"] = x[DATE_COL].isin(hol).astype(int)

    groups = []
    for key, g in x.groupby(KEY_COL, sort=False):
        g = g.sort_values(DATE_COL).copy()
        idx = g["is_nonholiday_weekday"] == 1
        med = g.loc[idx, TARGET_COL].astype(float).shift(1).rolling(window=4, min_periods=2).median()
        g.loc[idx, "weekday_median"] = med.values
        is_sensitive = ("단체" in str(key)) or ("브런치" in str(key))
        if is_sensitive:
            threshold = 1.25 * g["weekday_median"]
            mask = (g["is_weekday_holiday"] == 1) & np.isfinite(g["weekday_median"]) & (g[TARGET_COL] > threshold)
            g.loc[mask, TARGET_COL] = np.maximum(0.0, np.round(1.10 * g.loc[mask, "weekday_median"]))
        groups.append(g.drop(columns=["dow", "is_nonholiday_weekday", "is_weekday_holiday", "weekday_median"], errors="ignore"))
    return pd.concat(groups, ignore_index=True)

def apply_room_zero_fix(df: pd.DataFrame, meta: pd.DataFrame, window: int = 28, min_obs: int = 2) -> pd.DataFrame:
    if meta is None or meta.empty or "room_all_zero" not in meta.columns:
        return df.copy()
    x = df.merge(meta[[DATE_COL, "room_all_zero"]], on=DATE_COL, how="left")
    x["room_all_zero"] = x["room_all_zero"].fillna(0).astype(int)
    x["is_weekend"] = (x[DATE_COL].dt.dayofweek >= 5).astype(int)

    groups = []
    for _, g in x.groupby(KEY_COL, sort=False):
        g = g.sort_values(DATE_COL).copy()
        for idx in g.index[g["room_all_zero"] == 1]:
            d = g.at[idx, DATE_COL]
            we = int(g.at[idx, "is_weekend"])
            past = g[(g[DATE_COL] < d) & (g["is_weekend"] == we) & (g["room_all_zero"] == 0)]
            values = past[TARGET_COL].dropna().astype(float).values
            if values.size < min_obs:
                values = g[(g[DATE_COL] < d) & (g["room_all_zero"] == 0)][TARGET_COL].dropna().astype(float).values
            if values.size >= min_obs:
                g.at[idx, TARGET_COL] = max(0.0, float(np.median(values[-window:])))
        groups.append(g.drop(columns=["room_all_zero", "is_weekend"]))
    return pd.concat(groups, ignore_index=True)

def clamp_max_to_second_by_weekpart(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["_weekend"] = (x[DATE_COL].dt.dayofweek >= 5).astype(int)
    groups = []
    for _, g in x.groupby([KEY_COL, "_weekend"], sort=False):
        g = g.copy()
        positive = np.unique(g[TARGET_COL].astype(float).values[g[TARGET_COL].astype(float).values > 0])
        if positive.size == 0:
            g[TARGET_COL] = 0.0
        elif positive.size >= 2:
            largest, second = positive[-1], positive[-2]
            g.loc[g[TARGET_COL] == largest, TARGET_COL] = second
            if int(g["_weekend"].iloc[0]) == 0 and positive.size >= 3:
                third = positive[-3]
                g.loc[g[TARGET_COL] == second, TARGET_COL] = third
        groups.append(g.drop(columns=["_weekend"]))
    return pd.concat(groups, ignore_index=True)
