from __future__ import annotations

import numpy as np
import pandas as pd


def baseline_a_predict(X: pd.DataFrame, *, feature_name: str = "n_transactions_3m") -> np.ndarray:
    """Baseline A (spec): predict y_hat = N_i(t-3, t).

    This "model" does not require training. It simply returns the chosen feature column.
    """
    if feature_name not in X.columns:
        raise KeyError(f"Missing required feature: {feature_name}")
    return X[feature_name].to_numpy(dtype=float)

