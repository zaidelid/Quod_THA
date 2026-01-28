from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import TweedieRegressor

from ..config import DEFAULT_TWEEDIE_ALPHA, DEFAULT_TWEEDIE_MAX_ITER, DEFAULT_TWEEDIE_POWER


def train_tweedie_model(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    *,
    power: float = DEFAULT_TWEEDIE_POWER,
    alpha: float = DEFAULT_TWEEDIE_ALPHA,
    max_iter: int = DEFAULT_TWEEDIE_MAX_ITER,
) -> TweedieRegressor:
    """Train a Tweedie GLM for non-negative count-like targets.

    Returns a fitted `sklearn.linear_model.TweedieRegressor`.
    """
    model = TweedieRegressor(power=power, link="log", alpha=alpha, max_iter=max_iter)
    model.fit(X_train, y_train)
    return model


def tweedie_predict(model: TweedieRegressor, X: pd.DataFrame) -> np.ndarray:
    """Predict with a fitted Tweedie model and clamp to non-negative values."""
    return np.clip(model.predict(X), 0.0, None)

