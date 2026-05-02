# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

DATE_COL = "영업일자"
KEY_COL = "영업장명_메뉴명"
TARGET_COL = "매출수량"
PREDICT_DAYS = 7

@dataclass(frozen=True)
class ModelConfig:
    name: str
    input_window: int = 35
    seed: int = 42
    n_estimators: int = 1200
    learning_rate: float = 0.04
    num_leaves: int = 63
    tweedie_powers: Tuple[float, ...] = (1.10, 1.30, 1.50)
    apply_weekday_holiday_impute: bool = True
    apply_spike_clamp: bool = True
    apply_room_zero_fix: bool = False
    apply_weekpart_second_clamp: bool = False
    use_hwadam_features: bool = False
    min_prediction: int = 1
    round_threshold: float = 0.130
    num_threads: int = -1
