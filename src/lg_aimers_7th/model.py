# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.multioutput import MultiOutputRegressor

from .config import ModelConfig

def fit_tweedie_ensemble(X: pd.DataFrame, y: pd.DataFrame, config: ModelConfig) -> list[MultiOutputRegressor]:
    models = []
    for i, power in enumerate(config.tweedie_powers):
        params = dict(
            objective="tweedie",
            metric="mae",
            tweedie_variance_power=float(power),
            n_estimators=config.n_estimators,
            learning_rate=config.learning_rate,
            num_leaves=config.num_leaves,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=1,
            lambda_l1=0.1,
            lambda_l2=0.1,
            random_state=config.seed + i,
            num_threads=config.num_threads,
            verbose=-1,
        )
        if config.num_threads == -1:
            params.pop("num_threads", None)
        base = lgb.LGBMRegressor(**params)
        model = MultiOutputRegressor(base, n_jobs=1)
        model.fit(X, y.astype(np.float32))
        models.append(model)
    return models

def predict_ensemble(models: list[MultiOutputRegressor], X: pd.DataFrame) -> np.ndarray:
    preds = [m.predict(X) for m in models]
    return np.mean(preds, axis=0)
