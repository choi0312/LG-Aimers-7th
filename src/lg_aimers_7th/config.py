# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

DATE_COL = "영업일자"
KEY_COL = "영업장명_메뉴명"
TARGET_COL = "매출수량"
PREDICT_DAYS = 7


@dataclass
class PipelineConfig:
    name: str
    seed: int = 42
    n_estimators: int = 1200
    learning_rate: float = 0.04
    num_leaves: int = 63
    tweedie_powers: Tuple[float, ...] = (1.10, 1.30, 1.50)
    round_threshold: float = 0.13
    min_prediction: int = 1
    apply_holiday_spike_clip: bool = True
    apply_weekpart_max_clip: bool = False
    force_positive_minimum: bool = True
    use_gpu: bool = False
