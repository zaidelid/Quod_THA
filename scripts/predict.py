#!/usr/bin/env python
from __future__ import annotations

"""
Predict next-3-month purchases for a given customer at a cutoff month-end.

This is the "point" script: choose a model, a cutoff month, and a customer_id,
and get:
- predicted transaction count in [cutoff, cutoff+horizon)
- realized transaction count in [cutoff, cutoff+horizon)

Usage (examples):
  poetry run scripts/predict.py --cutoff 2019-07 --customer-id 9447359 --model baseline
  poetry run scripts/predict.py --cutoff 2019-07 --customer-id 9447359 --model tweedie
  poetry run scripts/predict.py --cutoff 2019-07 --customer-id 9447359 --model xgb
"""

import argparse
import json
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd

from src.builder import (
    baseline_a_predict,
    train_or_load_xgb_poisson_for_cutoff,
    train_tweedie_model,
    tweedie_predict,
)
from src.config import (
    DATA_PROCESSED,
    DEFAULT_TWEEDIE_ALPHA,
    DEFAULT_TWEEDIE_MAX_ITER,
    DEFAULT_TWEEDIE_POWER,
    MODELS_DIR,
)
from src.data import load_and_process_transactions
from src.panel import build_or_load_full_panel_dataset, make_future_tx_count

ModelName = Literal["baseline", "tweedie", "xgb"]


def _month_end(x) -> pd.Timestamp:
    ts = pd.Timestamp(x)
    if ts.tz is not None:
        ts = ts.tz_localize(None)
    return ts.normalize() + pd.offsets.MonthEnd(0)


def _train_cutoffs_for_inference(panel, cutoff: pd.Timestamp, horizon_months: int) -> list[pd.Timestamp]:
    last_train_cutoff = (cutoff - pd.DateOffset(months=horizon_months)).normalize() + pd.offsets.MonthEnd(0)
    return [c for c in panel.cutoffs if c <= last_train_cutoff]


def _tweedie_model_path(*, cutoff: pd.Timestamp, horizon_months: int, power: float, alpha: float, max_iter: int) -> Path:
    c = cutoff.date().isoformat()
    # keep filenames readable but deterministic
    p = str(power).replace(".", "p")
    a = str(alpha).replace(".", "p")
    return MODELS_DIR / f"tweedie_h{horizon_months}_cutoff={c}_p{p}_a{a}_it{max_iter}.joblib"


def _train_or_load_tweedie_for_cutoff(
    panel,
    cutoff: pd.Timestamp,
    *,
    horizon_months: int,
    power: float,
    alpha: float,
    max_iter: int,
    force_retrain: bool,
):
    path = _tweedie_model_path(
        cutoff=cutoff, horizon_months=horizon_months, power=power, alpha=alpha, max_iter=max_iter
    )
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists() and not force_retrain:
        payload = joblib.load(path)
        return payload["model"], dict(payload.get("meta", {}))

    train_cutoffs = _train_cutoffs_for_inference(panel, cutoff, horizon_months)
    X_train, y_train, _ = panel.for_cutoffs(train_cutoffs)

    model = train_tweedie_model(X_train, y_train, power=power, alpha=alpha, max_iter=max_iter)
    meta = {
        "cutoff": cutoff,
        "horizon_months": horizon_months,
        "power": power,
        "alpha": alpha,
        "max_iter": max_iter,
        "saved_to": str(path),
        "n_train": int(len(y_train)),
        "n_train_cutoffs": int(len(train_cutoffs)),
    }
    joblib.dump({"model": model, "meta": meta}, path)
    return model, meta


def predict_one(
    cutoff,
    customer_id: int,
    model: ModelName,
    *,
    horizon_months: int = 3,
    min_history_months: int = 6,
    n_estimators_max: int = 5000,
    force_retrain: bool = False,
) -> dict:
    cutoff = _month_end(cutoff)

    df = load_and_process_transactions(force_reprocess=False)
    panel_cache = DATA_PROCESSED / f"panel_h{horizon_months}_mh{min_history_months}.joblib"
    panel = build_or_load_full_panel_dataset(
        df,
        horizon_months=horizon_months,
        min_history_months=min_history_months,
        cache_path=panel_cache,
        force_rebuild=False,
    )

    # Realized target (ground truth)
    y_future = make_future_tx_count(df, cutoff, horizon_months)
    actual = int(y_future.get(customer_id, 0))

    # Features row for this customer at this cutoff (must exist in the panel block)
    Xc, _, keys = panel.for_cutoff(cutoff)
    mask = keys["customer_id"].astype(int) == int(customer_id)
    if not bool(mask.any()):
        return {
            "cutoff": cutoff.date().isoformat(),
            "customer_id": int(customer_id),
            "model": model,
            "horizon_months": int(horizon_months),
            "panel_cache": str(panel_cache),
            "eligible": False,
            "reason": "customer has no purchase history before cutoff (not present in panel block)",
            "prediction": None,
            "actual": actual,
        }
    X_row = Xc.loc[mask].reset_index(drop=True)

    # Prediction
    meta = {}
    if model == "baseline":
        pred = float(baseline_a_predict(X_row)[0])
    elif model == "tweedie":
        tw, meta = _train_or_load_tweedie_for_cutoff(
            panel,
            cutoff,
            horizon_months=horizon_months,
            power=DEFAULT_TWEEDIE_POWER,
            alpha=DEFAULT_TWEEDIE_ALPHA,
            max_iter=DEFAULT_TWEEDIE_MAX_ITER,
            force_retrain=force_retrain,
        )
        pred = float(tweedie_predict(tw, X_row)[0])
    elif model == "xgb":
        xgb, meta = train_or_load_xgb_poisson_for_cutoff(
            panel,
            cutoff,
            horizon_months=horizon_months,
            n_estimators_max=n_estimators_max,
            model_dir=MODELS_DIR,
            force_retrain=force_retrain,
        )
        pred = float(np.clip(xgb.predict(X_row)[0], 0.0, None))
    else:
        raise ValueError(f"Unknown model: {model}")

    return {
        "cutoff": cutoff.date().isoformat(),
        "customer_id": int(customer_id),
        "model": model,
        "horizon_months": int(horizon_months),
        "panel_cache": str(panel_cache),
        "eligible": True,
        "prediction": pred,
        "actual": actual,
        "model_meta": meta,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Predict next-3-month purchases for one customer at a cutoff month.")
    p.add_argument("--cutoff", required=True, help="Cutoff month (YYYY-MM) or date (YYYY-MM-DD).")
    p.add_argument("--customer-id", required=True, type=int, help="Customer ID.")
    p.add_argument("--model", required=True, choices=["baseline", "tweedie", "xgb"], help="Model to use.")
    p.add_argument("--horizon-months", type=int, default=3, help="Forecast horizon in months (default: 3).")
    p.add_argument("--min-history-months", type=int, default=6, help="Minimum history required (default: 6).")
    p.add_argument(
        "--xgb-n-estimators-max",
        type=int,
        default=5000,
        help="Upper bound for XGB estimators before early stopping (default: 5000).",
    )
    p.add_argument("--force-retrain", action="store_true", help="Ignore model caches and retrain.")
    p.add_argument("--json", action="store_true", help="Print machine-readable JSON output.")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    out = predict_one(
        args.cutoff,
        customer_id=args.customer_id,
        model=args.model,
        horizon_months=args.horizon_months,
        min_history_months=args.min_history_months,
        n_estimators_max=args.xgb_n_estimators_max,
        force_retrain=args.force_retrain,
    )

    if args.json:
        print(json.dumps(out, default=str, indent=2))
        return

    print(f"cutoff={out['cutoff']} | customer_id={out['customer_id']} | model={out['model']}")
    if not out["eligible"]:
        print(f"eligible=False | reason={out['reason']}")
        print(f"actual={out['actual']}")
        return
    print(f"prediction={out['prediction']:.4f} | actual={out['actual']}")
    if out.get("model_meta") and out["model"] in ("tweedie", "xgb"):
        saved_to = out["model_meta"].get("saved_to")
        if saved_to:
            print(f"model_cache={saved_to}")


if __name__ == "__main__":
    main()

