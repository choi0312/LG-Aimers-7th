# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import List

import lightgbm as lgb
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor

from .config import PipelineConfig


def fit_lightgbm_tweedie_ensemble(X: pd.DataFrame, Y: pd.DataFrame, config: PipelineConfig) -> List[MultiOutputRegressor]:
    models: List[MultiOutputRegressor] = []

    for index, power in enumerate(config.tweedie_powers):
        params = {
            "objective": "tweedie",
            "tweedie_variance_power": power,
            "n_estimators": config.n_estimators,
            "learning_rate": config.learning_rate,
            "num_leaves": config.num_leaves,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "reg_alpha": 0.1,
            "reg_lambda": 0.3,
            "min_child_samples": 20,
            "random_state": config.seed + index,
            "n_jobs": -1,
            "verbosity": -1,
        }
        if config.use_gpu:
            params["device_type"] = "gpu"

        base_model = lgb.LGBMRegressor(**params)
        model = MultiOutputRegressor(base_model, n_jobs=1)
        model.fit(X, Y)
        models.append(model)

    return models
