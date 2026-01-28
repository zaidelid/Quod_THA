#!/usr/bin/env python
from __future__ import annotations

"""
Benchmark models for a single cutoff month (2019).

Usage (examples):
  poetry run scripts/benchmark.py --cutoff 2019-07
  poetry run scripts/benchmark.py --cutoff 2019-07-31
"""

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_tweedie_deviance

from src.builder import (
    baseline_a_predict,
    evaluate_xgb_poisson_on_cutoff,
    train_or_load_xgb_poisson_for_cutoff,
    train_tweedie_model,
    tweedie_predict,
)
from src.config import DATA_PROCESSED, DEFAULT_TWEEDIE_POWER, MODELS_DIR
from src.data import load_and_process_transactions
from src.panel import build_or_load_full_panel_dataset


def _month_end(x) -> pd.Timestamp:
    """Parse a user cutoff (YYYY-MM or YYYY-MM-DD) and convert to month-end Timestamp."""
    ts = pd.Timestamp(x)
    if ts.tz is not None:
        ts = ts.tz_localize(None)
    return ts.normalize() + pd.offsets.MonthEnd(0)


def _require_2019(cutoff: pd.Timestamp) -> None:
    if int(cutoff.year) != 2019:
        raise ValueError(f"Benchmark is intended for 2019 cutoffs only. Got {cutoff.date()}.")


def _train_cutoffs_for_inference(panel, cutoff: pd.Timestamp, horizon_months: int) -> list[pd.Timestamp]:
    last_train_cutoff = (cutoff - pd.DateOffset(months=horizon_months)).normalize() + pd.offsets.MonthEnd(0)
    return [c for c in panel.cutoffs if c <= last_train_cutoff]


@dataclass
class Metrics:
    rmse: float
    mae: float
    deviance: float | None = None
    n_test: int | None = None


def _rmse(y_true, y_pred) -> float:
    return float(mean_squared_error(y_true, y_pred) ** 0.5)


def _eval_common(y_true, y_pred) -> Metrics:
    return Metrics(
        rmse=_rmse(y_true, y_pred),
        mae=float(mean_absolute_error(y_true, y_pred)),
        n_test=int(len(y_true)),
    )


def benchmark(cutoff, *, horizon_months: int = 3, min_history_months: int = 6, n_estimators_max: int = 5000) -> None:
    cutoff = _month_end(cutoff)
    _require_2019(cutoff)

    df = load_and_process_transactions(force_reprocess=False)

    panel_cache = DATA_PROCESSED / f"panel_h{horizon_months}_mh{min_history_months}.joblib"
    panel = build_or_load_full_panel_dataset(
        df,
        horizon_months=horizon_months,
        min_history_months=min_history_months,
        cache_path=panel_cache,
        force_rebuild=False,
    )

    train_cutoffs = _train_cutoffs_for_inference(panel, cutoff, horizon_months)
    X_train, y_train, _ = panel.for_cutoffs(train_cutoffs)
    X_test, y_test, _ = panel.for_cutoff(cutoff)

    print(f"Cutoff: {cutoff.date()} | horizon={horizon_months} months")
    print(f"Panel cache: {panel_cache}")
    print(f"Train cutoffs: {len(train_cutoffs)} (.. {train_cutoffs[-1].date()})")
    print(f"Train: X={X_train.shape} | Test: X={X_test.shape} | y_test_mean={float(np.mean(y_test)):.3f}")

    # --- Baseline A ---
    yb = baseline_a_predict(X_test)
    m_b = _eval_common(y_test, yb)

    # --- Tweedie ---
    power = DEFAULT_TWEEDIE_POWER
    tw = train_tweedie_model(X_train, y_train, power=power)
    y_tw = tweedie_predict(tw, X_test)
    m_tw = _eval_common(y_test, y_tw)
    # Tweedie deviance requires strictly positive predictions
    eps = 1e-9
    m_tw.deviance = float(mean_tweedie_deviance(y_test, np.maximum(y_tw, eps), power=power))

    # --- XGB Poisson (cached per cutoff) ---
    xgb_model, meta = train_or_load_xgb_poisson_for_cutoff(
        panel,
        cutoff,
        horizon_months=horizon_months,
        n_estimators_max=n_estimators_max,
        model_dir=MODELS_DIR,
    )
    m_xgb_raw = evaluate_xgb_poisson_on_cutoff(xgb_model, panel, cutoff)
    m_xgb = Metrics(rmse=m_xgb_raw["rmse"], mae=m_xgb_raw["mae"], n_test=m_xgb_raw["n_test"])

    print("\n=== Results ===")
    print(f"BaselineA      RMSE={m_b.rmse:.4f} | MAE={m_b.mae:.4f}")
    print(f"Tweedie(p={power}) RMSE={m_tw.rmse:.4f} | MAE={m_tw.mae:.4f} | dev={m_tw.deviance:.4f}")
    print(
        f"XGBPoisson     RMSE={m_xgb.rmse:.4f} | MAE={m_xgb.mae:.4f} | "
        f"val={pd.Timestamp(meta['val_cutoff']).date()} | n_estimators={meta['n_estimators']}"
    )
    print(f"\nSaved XGB model: {meta['saved_to']}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Benchmark Baseline/Tweedie/XGB for a single 2019 cutoff.")
    p.add_argument("--cutoff", required=True, help="Cutoff month (YYYY-MM) or date (YYYY-MM-DD). Must be in 2019.")
    p.add_argument("--horizon-months", type=int, default=3, help="Forecast horizon in months (default: 3).")
    p.add_argument("--min-history-months", type=int, default=6, help="Minimum history required for features (default: 6).")
    p.add_argument(
        "--xgb-n-estimators-max",
        type=int,
        default=5000,
        help="Upper bound for XGB estimators before early stopping (default: 5000).",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    benchmark(
        args.cutoff,
        horizon_months=args.horizon_months,
        min_history_months=args.min_history_months,
        n_estimators_max=args.xgb_n_estimators_max,
    )


if __name__ == "__main__":
    main()

