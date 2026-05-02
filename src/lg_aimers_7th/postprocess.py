# -*- coding: utf-8 -*-
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from .config import DATE_COL, KEY_COL, TARGET_COL, PREDICT_DAYS, ModelConfig
from .io import read_csv_robust

def round_nonnegative(pred: np.ndarray, threshold: float = 0.130, min_prediction: int = 1) -> np.ndarray:
    arr = np.nan_to_num(np.asarray(pred, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.clip(arr, 0, None)
    floor = np.floor(arr)
    frac = arr - floor
    rounded = np.where(frac >= threshold, floor + 1, floor)
    if min_prediction is not None and min_prediction > 0:
        rounded = np.clip(rounded, min_prediction, None)
    return rounded.astype(np.int64)

def sparse_item_guard(pred: np.ndarray, keys: list[str], test_history: pd.DataFrame) -> np.ndarray:
    out = np.asarray(pred, dtype=float).copy()
    for i, key in enumerate(keys):
        values = test_history[test_history[KEY_COL] == key].sort_values(DATE_COL)[TARGET_COL].to_numpy(dtype=float)
        if len(values) == 0:
            continue
        recent = values[-28:]
        if np.sum(recent) == 0:
            out[i, :] = 0
        elif np.mean(recent == 0) >= 0.90 and np.max(recent) <= 1:
            out[i, :] = np.minimum(out[i, :], 1)
    return out

def postprocess_prediction(pred: np.ndarray, keys: list[str], test_history: pd.DataFrame, config: ModelConfig) -> np.ndarray:
    pred = sparse_item_guard(pred, keys, test_history)
    return round_nonnegative(pred, threshold=config.round_threshold, min_prediction=config.min_prediction)

def build_submission(sample: pd.DataFrame, predictions: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    submission = sample.copy()
    key_cols = [c for c in submission.columns if c != DATE_COL]
    for idx, row in submission.iterrows():
        label = str(row[DATE_COL])
        match = re.match(r"(TEST_\d{2})\+(\d+)일", label)
        if not match:
            continue
        test_id, horizon = match.group(1), int(match.group(2))
        pred_df = predictions[test_id]
        for col in key_cols:
            submission.at[idx, col] = int(pred_df.loc[col, f"h{horizon}"]) if col in pred_df.index else 0
    for col in key_cols:
        submission[col] = pd.to_numeric(submission[col], errors="coerce").fillna(0).clip(lower=0).round().astype(int)
    return submission

def median_ensemble(path_a: str | Path, path_b: str | Path, out_path: str | Path, min_clip: int = 1) -> Path:
    a = read_csv_robust(path_a)
    b = read_csv_robust(path_b)
    id_col = a.columns[0]
    if id_col not in b.columns:
        raise ValueError(f"Common id column is required: {id_col}")
    pred_cols = [c for c in a.columns[1:] if c in b.columns]
    left = a[[id_col] + pred_cols].copy()
    right = b[[id_col] + pred_cols].set_index(id_col).reindex(left[id_col]).reset_index()
    med = np.median(np.stack([
        np.maximum(left[pred_cols].to_numpy(dtype=float), 0.0),
        np.maximum(right[pred_cols].to_numpy(dtype=float), 0.0),
    ], axis=0), axis=0)
    out = left[[id_col]].copy()
    out[pred_cols] = np.clip(np.rint(med), min_clip, None).astype(np.int64)
    for col in a.columns[1:]:
        if col not in out.columns:
            out[col] = a[col].values
    out = out[a.columns]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    return out_path
