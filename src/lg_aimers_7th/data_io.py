# -*- coding: utf-8 -*-
from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Optional, Sequence, List

import numpy as np
import pandas as pd

from .config import DATE_COL, KEY_COL, TARGET_COL


def read_csv_robust(path: str | Path) -> pd.DataFrame:
    path = str(path)
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="cp949")


def canon(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value).replace("\ufeff", "").replace("\xa0", " ").strip())
    text = re.sub(r"\s+", "", text.lower())
    return text.replace("_", "").replace("-", "")


def normalize_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.normalize()


def find_file(data_root: str | Path, candidates: Sequence[str], required: bool = True) -> Optional[Path]:
    root = Path(data_root)
    for candidate in candidates:
        path = root / candidate
        if path.exists():
            return path

    basenames = {Path(candidate).name for candidate in candidates}
    for path in root.rglob("*"):
        if path.is_file() and path.name in basenames:
            return path

    if required:
        raise FileNotFoundError(f"Required file not found. root={root}, candidates={candidates}")
    return None


def list_test_files(data_root: str | Path) -> List[Path]:
    root = Path(data_root)
    files = sorted(root.glob("test/TEST_*.csv"))
    if not files:
        files = sorted(root.rglob("TEST_*.csv"))
        files = [path for path in files if re.fullmatch(r"TEST_\d{2}\.csv", path.name)]
    if not files:
        raise FileNotFoundError("TEST_00.csv ~ TEST_09.csv files were not found.")
    return files


def standardize_sales_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    columns = list(df.columns)
    normalized = {col: canon(col) for col in columns}

    date_aliases = {canon(x) for x in ["영업일자", "일자", "날짜", "date"]}
    key_aliases = {canon(x) for x in ["영업장명_메뉴명", "영업장명메뉴명", "key", "메뉴키"]}
    store_aliases = {canon(x) for x in ["영업장명", "매장명", "지점명", "점포명", "매장", "영업장"]}
    menu_aliases = {canon(x) for x in ["메뉴명", "상품명", "제품명", "메뉴"]}
    target_aliases = {canon(x) for x in ["매출수량", "판매수량", "수량", "qty", "판매량"]}

    date_col = next((col for col in columns if normalized[col] in date_aliases), None)
    key_col = next((col for col in columns if normalized[col] in key_aliases), None)
    target_col = next((col for col in columns if normalized[col] in target_aliases), None)

    if date_col is None:
        raise ValueError("Could not identify the date column.")
    if target_col is None:
        raise ValueError("Could not identify the target column.")

    if key_col is None:
        store_col = next((col for col in columns if normalized[col] in store_aliases), None)
        menu_col = next((col for col in columns if normalized[col] in menu_aliases), None)
        if store_col is None or menu_col is None:
            raise ValueError("Could not identify key columns.")
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

    price = read_csv_robust(path).copy()
    key_col = next((col for col in price.columns if canon(col) == canon(KEY_COL)), price.columns[0])
    value_col = next((col for col in price.columns if "평균" in str(col) or "price" in str(col).lower()), price.columns[-1])

    price = price.rename(columns={key_col: KEY_COL, value_col: "avg_price"})[[KEY_COL, "avg_price"]]
    price[KEY_COL] = price[KEY_COL].astype(str).str.strip()
    price["avg_price"] = pd.to_numeric(price["avg_price"], errors="coerce")
    return price


def numeric_date_frame(df: pd.DataFrame, date_col: str, prefix: str) -> pd.DataFrame:
    output = df.copy()
    output[date_col] = normalize_date(output[date_col])
    for col in output.columns:
        if col != date_col:
            output[col] = pd.to_numeric(output[col], errors="coerce")
    output = output.groupby(date_col, as_index=False).mean(numeric_only=True)
    output = output.rename(columns={date_col: DATE_COL})
    output = output.rename(columns={col: f"{prefix}_{canon(col)}" for col in output.columns if col != DATE_COL})
    return output


def load_generic_meta(data_root: str | Path, meta_name: str, prefix: str, test_id: str | None = None) -> pd.DataFrame:
    if test_id is None:
        candidates = [f"train/meta/TRAIN_{meta_name}.csv", f"meta/TRAIN_{meta_name}.csv", f"TRAIN_{meta_name}.csv"]
    else:
        suffix = test_id.split("_")[-1]
        candidates = [f"test/meta/TEST_{meta_name}_{suffix}.csv", f"meta/TEST_{meta_name}_{suffix}.csv", f"TEST_{meta_name}_{suffix}.csv"]

    path = find_file(data_root, candidates, required=False)
    if path is None:
        return pd.DataFrame({DATE_COL: []})

    df = read_csv_robust(path)
    date_col = next((col for col in df.columns if canon(col) in {canon(DATE_COL), canon("일시"), canon("일자"), canon("date")}), df.columns[0])
    return numeric_date_frame(df, date_col, prefix)


def load_room_meta(data_root: str | Path, test_id: str | None = None) -> pd.DataFrame:
    if test_id is None:
        room_path = find_file(data_root, ["train/meta/TRAIN_room.csv", "meta/TRAIN_room.csv", "TRAIN_room.csv"], required=False)
    else:
        suffix = test_id.split("_")[-1]
        room_path = find_file(data_root, [f"test/meta/TEST_room_{suffix}.csv", f"meta/TEST_room_{suffix}.csv", f"TEST_room_{suffix}.csv"], required=False)

    if room_path is None:
        return pd.DataFrame({DATE_COL: []})

    room = read_csv_robust(room_path).copy()
    date_col = next((col for col in room.columns if canon(col) in {canon(DATE_COL), canon("일자"), canon("date")}), room.columns[0])
    room[date_col] = normalize_date(room[date_col])

    room_type_path = find_file(data_root, ["train/room_type.csv", "room_type.csv"], required=False)
    if room_type_path is not None:
        room_type = read_csv_robust(room_type_path)
        if {"객실타입", "기준인원"}.issubset(set(room_type.columns)):
            capacity = dict(zip(room_type["객실타입"].astype(str), pd.to_numeric(room_type["기준인원"], errors="coerce").fillna(0)))
            proxy = np.zeros(len(room), dtype=float)
            for room_col, cap in capacity.items():
                if room_col in room.columns:
                    proxy += pd.to_numeric(room[room_col], errors="coerce").fillna(0).to_numpy() * float(cap)
            room["room_capacity_proxy"] = proxy

    return numeric_date_frame(room, date_col, "room")


def build_daily_meta(data_root: str | Path, test_id: str | None = None) -> pd.DataFrame:
    frames = [
        load_room_meta(data_root, test_id),
        load_generic_meta(data_root, "group", "group", test_id),
        load_generic_meta(data_root, "hwadam", "hwadam", test_id),
        load_generic_meta(data_root, "ski", "ski", test_id),
        load_generic_meta(data_root, "weather", "weather", test_id),
    ]

    merged = None
    for frame in frames:
        if frame is None or frame.empty or DATE_COL not in frame.columns:
            continue
        merged = frame if merged is None else merged.merge(frame, on=DATE_COL, how="outer")

    if merged is None:
        return pd.DataFrame({DATE_COL: []})

    merged = merged.sort_values(DATE_COL).reset_index(drop=True)
    numeric_cols = [col for col in merged.columns if col != DATE_COL]
    merged[numeric_cols] = merged[numeric_cols].ffill().bfill().fillna(0)
    return merged
