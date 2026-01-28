from __future__ import annotations

"""
Feature engineering (no-leakage spec)
====================================

This module defines *time-aware* customer-level features used for forecasting.

## Expected input schema
- **df**: transactions table with at least:
  - `customer_id`
  - `date` (datetime64[ns], tz-naive recommended)

## Cutoff / leakage rule (critical)
For a given cutoff \(t\):
- **Features MUST use only past information**: strictly `date < t`
- No aggregation is allowed to look at transactions on/after `t`.

## Outputs
- `create_feature_vector(...)` returns a numeric `pd.Series`.
- `create_feature_matrix(...)` returns a `pd.DataFrame` with:
  - join-back columns: `customer_id`, `cutoff_date`
  - numeric feature columns only (no IDs/timestamps beyond join-back columns)

## Customer inclusion rule
`create_feature_matrix` returns **only customers with at least one transaction strictly
before the cutoff** (customers with no purchase history are dropped). This matches the
original notebook behavior.
"""

from typing import Iterable, Optional

import numpy as np
import pandas as pd


def _to_ts_naive(x) -> pd.Timestamp:
    """Convert to tz-naive Timestamp (no normalization)."""
    ts = pd.Timestamp(x)
    if ts.tz is not None:
        ts = ts.tz_localize(None)
    return ts


def compute_transaction_counts(
    df: pd.DataFrame,
    customer_id: int | str,
    cutoff_date,
    window_months: int,
) -> int:
    """Compute \(N_i(t-w, t)\): number of transactions in the lookback window \([t-w, t)\)."""
    cutoff_date = _to_ts_naive(cutoff_date)
    start_date = cutoff_date - pd.DateOffset(months=window_months)

    mask = (
        (df["customer_id"] == customer_id)
        & (df["date"] >= start_date)
        & (df["date"] < cutoff_date)
    )
    return int(mask.sum())


def compute_days_since_last_transaction(
    df: pd.DataFrame, customer_id: int | str, cutoff_date
) -> float:
    """Days since last tx strictly before cutoff. Returns NaN if no history."""
    cutoff_date = _to_ts_naive(cutoff_date)

    customer_tx = df[df["customer_id"] == customer_id]
    customer_tx_before = customer_tx[customer_tx["date"] < cutoff_date]
    if len(customer_tx_before) == 0:
        return float("nan")

    last_tx_date = pd.Timestamp(customer_tx_before["date"].max())
    return float((cutoff_date - last_tx_date).days)


def compute_active_months(df: pd.DataFrame, customer_id: int | str, cutoff_date) -> int:
    """Calendar months from first tx (before cutoff) to cutoff month, inclusive."""
    cutoff_date = _to_ts_naive(cutoff_date)

    customer_tx = df[df["customer_id"] == customer_id]
    customer_tx_before = customer_tx[customer_tx["date"] < cutoff_date]
    if len(customer_tx_before) == 0:
        return 0

    first_tx_date = pd.Timestamp(customer_tx_before["date"].min())
    first_period = pd.Period(first_tx_date, freq="M")
    cutoff_period = pd.Period(cutoff_date, freq="M")
    return int((cutoff_period - first_period).n + 1)


def create_feature_vector(df: pd.DataFrame, customer_id: int | str, cutoff_date) -> pd.Series:
    """Create feature vector for a customer at cutoff \(t\) using ONLY `date < t`."""
    cutoff_date = _to_ts_naive(cutoff_date)

    n_1m = compute_transaction_counts(df, customer_id, cutoff_date, window_months=1)
    n_3m = compute_transaction_counts(df, customer_id, cutoff_date, window_months=3)
    n_6m = compute_transaction_counts(df, customer_id, cutoff_date, window_months=6)

    t_minus_3 = cutoff_date - pd.DateOffset(months=3)
    n_3m_prev = compute_transaction_counts(df, customer_id, t_minus_3, window_months=3)
    change_rate = n_3m - n_3m_prev

    days_since_last = compute_days_since_last_transaction(df, customer_id, cutoff_date)
    active_months = compute_active_months(df, customer_id, cutoff_date)
    month_of_year = int(cutoff_date.month)

    return pd.Series(
        {
            "n_transactions_1m": float(n_1m),
            "n_transactions_3m": float(n_3m),
            "n_transactions_6m": float(n_6m),
            "change_rate_3m": float(change_rate),
            "days_since_last_tx": float(days_since_last),
            "active_months": float(active_months),
            "month_of_year": float(month_of_year),
        }
    )


def create_feature_matrix(
    df: pd.DataFrame,
    cutoff_date,
    customer_ids: Optional[Iterable[int | str]] = None,
    *,
    verbose: bool = False,
) -> pd.DataFrame:
    """Create a feature matrix for a single cutoff \(t\) using ONLY `date < t`.

    Notes:
    - **No leakage**: all aggregations are restricted to strictly-before-cutoff data.
    - **Rows**: one row per customer with purchase history before \(t\).
    - **Join-back**: includes `customer_id` and `cutoff_date`; these are NOT model features.
    """
    cutoff_date = _to_ts_naive(cutoff_date)

    df_before_cutoff = df[df["date"] < cutoff_date]
    customers_with_history = set(df_before_cutoff["customer_id"].unique())

    if customer_ids is not None:
        customer_ids = pd.Series(list(customer_ids)).unique().tolist()
        valid_customers = [c for c in customer_ids if c in customers_with_history]
        if len(valid_customers) == 0:
            raise ValueError(
                "No customers in provided list have purchase history before cutoff_date"
            )
    else:
        valid_customers = list(customers_with_history)

    if verbose:
        print(f"Computing features for {len(valid_customers):,} customers before {cutoff_date}")

    feature_rows = [create_feature_vector(df, cid, cutoff_date) for cid in valid_customers]
    feature_df = pd.DataFrame(feature_rows)
    feature_df.insert(0, "customer_id", np.array(valid_customers))
    feature_df.insert(1, "cutoff_date", cutoff_date)
    return feature_df

