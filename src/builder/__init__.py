"""Model builders (code) for forecasting.

Naming note:
- This package is called `builder` to avoid confusion with the **project-root** `models/`
  directory, which stores *saved model artifacts* (joblib files) produced by training.
"""

from .baseline_a import baseline_a_predict
from .tweedie import train_tweedie_model, tweedie_predict
from .xgb_poisson import evaluate_xgb_poisson_on_cutoff, train_or_load_xgb_poisson_for_cutoff

__all__ = [
    "baseline_a_predict",
    "train_tweedie_model",
    "tweedie_predict",
    "train_or_load_xgb_poisson_for_cutoff",
    "evaluate_xgb_poisson_on_cutoff",
]

