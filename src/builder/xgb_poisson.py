from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

from ..config import (
    DEFAULT_XGB_COLSAMPLE_BYTREE,
    DEFAULT_XGB_EARLY_STOPPING_ROUNDS,
    DEFAULT_XGB_GAMMA,
    DEFAULT_XGB_MAX_DEPTH,
    DEFAULT_XGB_MIN_CHILD_WEIGHT,
    DEFAULT_XGB_RANDOM_STATE,
    DEFAULT_XGB_REG_ALPHA,
    DEFAULT_XGB_REG_LAMBDA,
    DEFAULT_XGB_SUBSAMPLE,
    DEFAULT_XGB_VAL_OFFSET_MONTHS,
    MODELS_DIR,
    PROJECT_ROOT,
)
from ..panel import PanelDataset, _norm_ts


def _month_end(ts: pd.Timestamp) -> pd.Timestamp:
    ts = _norm_ts(ts)
    return ts + pd.offsets.MonthEnd(0)


def _xgb_model_path(
    *,
    model_dir: str | Path,
    horizon_months: int,
    cutoff: pd.Timestamp,
    val_cutoff: pd.Timestamp,
) -> Path:
    """Deterministic cache path for a given (cutoff, validation cutoff, horizon)."""
    model_dir = Path(model_dir)
    if not model_dir.is_absolute():
        model_dir = PROJECT_ROOT / model_dir
    model_dir.mkdir(parents=True, exist_ok=True)
    c = _norm_ts(cutoff).date().isoformat()
    v = _norm_ts(val_cutoff).date().isoformat()
    return model_dir / f"xgb_poisson_h{horizon_months}_cutoff={c}_val={v}.joblib"


def train_or_load_xgb_poisson_for_cutoff(
    panel: PanelDataset,
    cutoff,
    *,
    horizon_months: int = 3,
    val_offset_months: int = DEFAULT_XGB_VAL_OFFSET_MONTHS,
    max_depth: int = DEFAULT_XGB_MAX_DEPTH,
    min_child_weight: float = DEFAULT_XGB_MIN_CHILD_WEIGHT,
    subsample: float = DEFAULT_XGB_SUBSAMPLE,
    colsample_bytree: float = DEFAULT_XGB_COLSAMPLE_BYTREE,
    gamma: float = DEFAULT_XGB_GAMMA,
    reg_alpha: float = DEFAULT_XGB_REG_ALPHA,
    reg_lambda: float = DEFAULT_XGB_REG_LAMBDA,
    early_stopping_rounds: int = DEFAULT_XGB_EARLY_STOPPING_ROUNDS,
    n_estimators_max: int = 4000,
    random_state: int = DEFAULT_XGB_RANDOM_STATE,
    model_dir: str | Path = MODELS_DIR,
    force_retrain: bool = False,
):
    """Train (or load) an XGBoost Poisson regressor for an inference cutoff T.

    No-leakage time split for inference cutoff T (month-end):
    - Validation cutoff V = (T - val_offset_months) month-end
    - Early-stopping training cutoffs satisfy: cutoff + horizon <= V
    - Final training cutoffs satisfy:          cutoff + horizon <= T
    - Evaluation (if you do it) should be done ONLY on T.
    """
    cutoff = _month_end(pd.Timestamp(cutoff))
    val_cutoff = _month_end(cutoff - pd.DateOffset(months=val_offset_months))
    if not (val_cutoff < cutoff):
        raise ValueError("Validation cutoff must be strictly before cutoff")

    path = _xgb_model_path(
        model_dir=model_dir,
        horizon_months=horizon_months,
        cutoff=cutoff,
        val_cutoff=val_cutoff,
    )
    if path.exists() and not force_retrain:
        payload = joblib.load(path)
        meta = dict(payload.get("meta", {}))
        meta["saved_to"] = str(path)
        return payload["model"], meta

    # --- Early-stopping split (NO test leakage) ---
    last_es_train_cutoff = _month_end(val_cutoff - pd.DateOffset(months=horizon_months))
    es_train_cutoffs = [c for c in panel.cutoffs if c <= last_es_train_cutoff]

    X_train_es, y_train_es, _ = panel.for_cutoffs(es_train_cutoffs)
    X_val, y_val, _ = panel.for_cutoff(val_cutoff)

    xgb_es = XGBRegressor(
        objective="count:poisson",
        eval_metric="rmse",
        n_estimators=n_estimators_max,
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        gamma=gamma,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        n_jobs=-1,
        random_state=random_state,
        early_stopping_rounds=early_stopping_rounds,
    )
    xgb_es.fit(X_train_es, y_train_es, eval_set=[(X_val, y_val)], verbose=False)

    best_iteration = int(xgb_es.best_iteration)
    best_n_estimators = best_iteration + 1  # 0-based

    # --- Final training split (use all available pre-cutoff data) ---
    last_full_train_cutoff = _month_end(cutoff - pd.DateOffset(months=horizon_months))
    full_train_cutoffs = [c for c in panel.cutoffs if c <= last_full_train_cutoff]

    X_train_full, y_train_full, _ = panel.for_cutoffs(full_train_cutoffs)

    xgb_final = XGBRegressor(
        objective="count:poisson",
        eval_metric="rmse",
        n_estimators=best_n_estimators,
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        gamma=gamma,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        n_jobs=-1,
        random_state=random_state,
    )
    xgb_final.fit(X_train_full, y_train_full)

    meta = {
        "cutoff": _norm_ts(cutoff),
        "val_cutoff": _norm_ts(val_cutoff),
        "best_iteration": best_iteration,
        "n_estimators": best_n_estimators,
        "horizon_months": horizon_months,
        "val_offset_months": val_offset_months,
        "max_depth": max_depth,
        "min_child_weight": float(min_child_weight),
        "subsample": float(subsample),
        "colsample_bytree": float(colsample_bytree),
        "gamma": float(gamma),
        "reg_alpha": float(reg_alpha),
        "reg_lambda": float(reg_lambda),
        "n_estimators_max": n_estimators_max,
        "early_stopping_rounds": early_stopping_rounds,
        "random_state": random_state,
        "saved_to": str(path),
    }
    joblib.dump({"model": xgb_final, "meta": meta}, path)
    return xgb_final, meta


def evaluate_xgb_poisson_on_cutoff(model: XGBRegressor, panel: PanelDataset, cutoff) -> dict:
    """Evaluate a fitted model on a single cutoff block from the pre-built panel."""
    cutoff = _month_end(pd.Timestamp(cutoff))
    X_test, y_test, _ = panel.for_cutoff(cutoff)
    y_pred = np.clip(model.predict(X_test), 0.0, None)
    mse = mean_squared_error(y_test, y_pred)
    return {
        "rmse": float(mse**0.5),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "n_test": int(len(y_test)),
        "y_mean": float(np.mean(y_test)),
    }

