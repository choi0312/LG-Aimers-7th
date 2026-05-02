# -*- coding: utf-8 -*-
from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from .config import DATE_COL, KEY_COL, TARGET_COL

def canon(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value).replace("\ufeff", "").replace("\xa0", " ").strip())
    text = re.sub(r"\s+", "", text.lower())
    return text.replace("_", "").replace("-", "")

def read_csv_robust(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="cp949")

def normalize_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.normalize()

def find_file(data_root: str | Path, candidates: Sequence[str], required: bool = True) -> Optional[Path]:
    root = Path(data_root)
    for candidate in candidates:
        path = root / candidate
        if path.exists():
            return path
    basenames = {Path(c).name for c in candidates}
    for path in root.rglob("*"):
        if path.is_file() and path.name in basenames:
            return path
    if required:
        raise FileNotFoundError(f"Required file not found. root={root}, candidates={candidates}")
    return None

def list_test_files(data_root: str | Path) -> list[Path]:
    root = Path(data_root)
    files = sorted(root.glob("test/TEST_*.csv"))
    if not files:
        files = sorted(p for p in root.rglob("TEST_*.csv") if re.fullmatch(r"TEST_\d{2}\.csv", p.name))
    if not files:
        raise FileNotFoundError("TEST_00.csv ~ TEST_09.csv files were not found.")
    return files

def standardize_sales_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    columns = list(df.columns)
    cmap = {c: canon(c) for c in columns}

    date_alias = {canon(x) for x in ["영업일자", "일자", "날짜", "date"]}
    key_alias = {canon(x) for x in ["영업장명_메뉴명", "영업장명메뉴명", "key", "메뉴키"]}
    store_alias = {canon(x) for x in ["영업장명", "매장명", "점포명", "지점명", "영업장"]}
    menu_alias = {canon(x) for x in ["메뉴명", "상품명", "제품명", "메뉴"]}
    target_alias = {canon(x) for x in ["매출수량", "판매수량", "수량", "qty", "판매량"]}

    date_col = next((c for c in columns if cmap[c] in date_alias), None)
    key_col = next((c for c in columns if cmap[c] in key_alias), None)
    target_col = next((c for c in columns if cmap[c] in target_alias), None)

    if date_col is None:
        raise ValueError("Could not identify date column.")
    if target_col is None:
        raise ValueError("Could not identify target column.")

    if key_col is None:
        store_col = next((c for c in columns if cmap[c] in store_alias), None)
        menu_col = next((c for c in columns if cmap[c] in menu_alias), None)
        if store_col is None or menu_col is None:
            raise ValueError("Could not identify outlet/menu key columns.")
        df[KEY_COL] = df[store_col].astype(str).str.strip() + "_" + df[menu_col].astype(str).str.strip()
    elif key_col != KEY_COL:
        df = df.rename(columns={key_col: KEY_COL})

    if date_col != DATE_COL:
        df = df.rename(columns={date_col: DATE_COL})
    if target_col != TARGET_COL:
        df = df.rename(columns={target_col: TARGET_COL})

    df[DATE_COL] = normalize_date(df[DATE_COL])
    df[KEY_COL] = df[KEY_COL].astype(str).str.strip()
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce").fillna(0).clip(lower=0)
    return df[[DATE_COL, KEY_COL, TARGET_COL]].sort_values([KEY_COL, DATE_COL]).reset_index(drop=True)

def load_price(data_root: str | Path) -> pd.DataFrame:
    path = find_file(data_root, ["train/price.csv", "price.csv"], required=False)
    if path is None:
        return pd.DataFrame({KEY_COL: [], "avg_price": []})
    price = read_csv_robust(path)
    key_col = next((c for c in price.columns if canon(c) == canon(KEY_COL)), price.columns[0])
    val_col = next((c for c in price.columns if "평균" in str(c) or "price" in str(c).lower()), price.columns[-1])
    out = price.rename(columns={key_col: KEY_COL, val_col: "avg_price"})[[KEY_COL, "avg_price"]].copy()
    out[KEY_COL] = out[KEY_COL].astype(str).str.strip()
    out["avg_price"] = pd.to_numeric(out["avg_price"], errors="coerce")
    return out

def _numeric_by_date(df: pd.DataFrame, date_col: str, prefix: str) -> pd.DataFrame:
    out = df.copy()
    out[date_col] = normalize_date(out[date_col])
    for col in out.columns:
        if col != date_col:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.groupby(date_col, as_index=False).mean(numeric_only=True)
    out = out.rename(columns={date_col: DATE_COL})
    out = out.rename(columns={c: f"{prefix}_{canon(c)}" for c in out.columns if c != DATE_COL})
    return out

def load_weather(data_root: str | Path, test_id: str | None = None) -> pd.DataFrame:
    suffix = "" if test_id is None else "_" + test_id.split("_")[-1]
    candidates = ["train/meta/TRAIN_weather.csv", "meta/TRAIN_weather.csv", "TRAIN_weather.csv"] if test_id is None else [
        f"test/meta/TEST_weather{suffix}.csv", f"meta/TEST_weather{suffix}.csv", f"TEST_weather{suffix}.csv"
    ]
    path = find_file(data_root, candidates, required=False)
    if path is None:
        return pd.DataFrame({DATE_COL: []})
    df = read_csv_robust(path)
    date_col = next((c for c in df.columns if canon(c) in {canon(DATE_COL), canon("일시"), canon("일자"), canon("date")}), df.columns[0])
    out = _numeric_by_date(df, date_col, "wx")
    return out

def load_room(data_root: str | Path, test_id: str | None = None) -> pd.DataFrame:
    suffix = "" if test_id is None else "_" + test_id.split("_")[-1]
    candidates = ["train/meta/TRAIN_room.csv", "meta/TRAIN_room.csv", "TRAIN_room.csv"] if test_id is None else [
        f"test/meta/TEST_room{suffix}.csv", f"meta/TEST_room{suffix}.csv", f"TEST_room{suffix}.csv"
    ]
    path = find_file(data_root, candidates, required=False)
    if path is None:
        return pd.DataFrame({DATE_COL: []})
    df = read_csv_robust(path)
    date_col = next((c for c in df.columns if canon(c) in {canon(DATE_COL), canon("일자"), canon("date")}), df.columns[0])
    out = _numeric_by_date(df, date_col, "room")

    raw = read_csv_robust(path)
    raw[date_col] = normalize_date(raw[date_col])
    numeric_cols = [c for c in raw.columns if c != date_col]
    for c in numeric_cols:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    room_sum = raw.groupby(date_col)[numeric_cols].sum(min_count=1).sum(axis=1).rename("room_total")
    flags = (room_sum.fillna(0) == 0).astype(int).rename("room_all_zero").reset_index().rename(columns={date_col: DATE_COL})
    out = out.merge(flags, on=DATE_COL, how="left")
    return out

def load_generic_meta(data_root: str | Path, name: str, prefix: str, test_id: str | None = None) -> pd.DataFrame:
    suffix = "" if test_id is None else "_" + test_id.split("_")[-1]
    candidates = [f"train/meta/TRAIN_{name}.csv", f"meta/TRAIN_{name}.csv", f"TRAIN_{name}.csv"] if test_id is None else [
        f"test/meta/TEST_{name}{suffix}.csv", f"meta/TEST_{name}{suffix}.csv", f"TEST_{name}{suffix}.csv"
    ]
    path = find_file(data_root, candidates, required=False)
    if path is None:
        return pd.DataFrame({DATE_COL: []})
    df = read_csv_robust(path)
    date_col = next((c for c in df.columns if canon(c) in {canon(DATE_COL), canon("일시"), canon("일자"), canon("date")}), df.columns[0])
    return _numeric_by_date(df, date_col, prefix)

def build_daily_meta(data_root: str | Path, test_id: str | None = None) -> pd.DataFrame:
    frames = [
        load_weather(data_root, test_id),
        load_room(data_root, test_id),
        load_generic_meta(data_root, "group", "group", test_id),
        load_generic_meta(data_root, "ski", "ski", test_id),
        load_generic_meta(data_root, "hwadam", "hwadam", test_id),
    ]
    merged = None
    for frame in frames:
        if frame is None or frame.empty or DATE_COL not in frame.columns:
            continue
        merged = frame if merged is None else merged.merge(frame, on=DATE_COL, how="outer")
    if merged is None:
        return pd.DataFrame({DATE_COL: []})
    merged = merged.sort_values(DATE_COL).reset_index(drop=True)
    num_cols = [c for c in merged.columns if c != DATE_COL]
    merged[num_cols] = merged[num_cols].ffill().bfill().fillna(0)
    return merged
