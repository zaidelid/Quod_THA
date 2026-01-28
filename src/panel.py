from __future__ import annotations

"""
Panel dataset construction (no-leakage spec)
==========================================

This module turns raw transactions into a *panel* suitable for time-based training:
one row per `(customer_id, cutoff_date)` and a future-horizon label.

## Definitions
For each customer \(i\) and cutoff month-end \(t\):
- **Features** \(X_{i,t}\): computed using ONLY transactions strictly before cutoff:
  - `date < t`
- **Label** \(y_{i,t}\): number of transactions in the future horizon:
  - `date in [t, t + horizon_months)`

## Why we cache a “full panel”
When you want to evaluate many cutoffs (e.g. multiple months in 2019), rebuilding
features for each cutoff is expensive. We therefore build a comprehensive no-leakage
panel for *all valid cutoffs*, then slice it in O(1) time per cutoff.

The cached file (e.g. `data/processed/panel_h3_mh6.joblib`) is a serialized
`PanelDataset` object (see class docs below).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import joblib
import numpy as np
import pandas as pd

from .features import create_feature_matrix
from .config import PROJECT_ROOT, DATA_PROCESSED


def _norm_ts(x) -> pd.Timestamp:
    """Convert to tz-naive Timestamp and normalize to midnight (date-only semantics)."""
    ts = pd.Timestamp(x)
    if ts.tz is not None:
        ts = ts.tz_localize(None)
    return ts.normalize()


def make_future_tx_count(
    df: pd.DataFrame, cutoff_date: pd.Timestamp, horizon_months: int
) -> pd.Series:
    """Compute \(y_i(t)\) = #transactions for customer \(i\) in \([t, t+horizon)\)."""
    cutoff_date = _norm_ts(cutoff_date)
    horizon_end = _norm_ts(cutoff_date + pd.DateOffset(months=horizon_months))
    mask = (df["date"] >= cutoff_date) & (df["date"] < horizon_end)
    y = df.loc[mask].groupby("customer_id").size()
    y.name = "y_future_tx_count"
    return y


def build_panel_dataset(df: pd.DataFrame, cutoffs: Iterable, horizon_months: int):
    """Return X, y, keys for a stacked `(customer_id, cutoff_date)` panel.

    Cutoff rule:
    - Features X_{i,t} use ONLY transactions strictly before the cutoff: date < t
    - Labels   y_{i,t} count transactions in the future horizon:        [t, t + horizon)

    Returns:
    - keys: customer_id, cutoff_date (join-back only)
    - X: numeric features only (no IDs, no timestamps)
    - y: aligned to keys, defaults to 0 if customer has no tx in horizon

    Uniqueness assertion:
    - exactly one row per (customer_id, cutoff_date)
    """
    cutoffs = [_norm_ts(c) for c in cutoffs]

    keys_parts: list[pd.DataFrame] = []
    X_parts: list[pd.DataFrame] = []
    y_parts: list[np.ndarray] = []

    for cutoff in cutoffs:
        fm = create_feature_matrix(df, cutoff, verbose=False)

        # exactly one row per (customer_id, cutoff_date)
        assert not fm[["customer_id", "cutoff_date"]].duplicated().any()

        keys = fm[["customer_id", "cutoff_date"]].copy()
        X = fm.drop(columns=["customer_id", "cutoff_date"])

        non_numeric = X.select_dtypes(exclude="number").columns.tolist()
        assert non_numeric == [], f"Non-numeric feature columns found: {non_numeric}"

        y_s = make_future_tx_count(df, cutoff, horizon_months)
        y = y_s.reindex(keys["customer_id"]).fillna(0).to_numpy(dtype=float)

        keys_parts.append(keys)
        X_parts.append(X)
        y_parts.append(y)

    keys_all = pd.concat(keys_parts, ignore_index=True)
    X_all = pd.concat(X_parts, ignore_index=True)
    y_all = np.concatenate(y_parts, axis=0)

    assert not keys_all[["customer_id", "cutoff_date"]].duplicated().any(), (
        "Duplicate (customer_id, cutoff_date) rows"
    )
    assert len(X_all) == len(keys_all) == len(y_all)

    return X_all, y_all, keys_all


@dataclass(frozen=True)
class PanelDataset:
    """Precomputed panel for many cutoffs; slice by cutoff without rebuilding features.

    ## What this object contains
    This object stores the *full* stacked panel for many monthly cutoffs:
    - **X**: numeric feature matrix (pandas DataFrame)
    - **y**: target vector aligned to X (numpy array)
    - **keys**: join-back columns `customer_id`, `cutoff_date` (pandas DataFrame)
    - **cutoffs**: the list of cutoffs included (month-ends)
    - **cutoff_slices**: mapping cutoff -> `(start, end)` row indices into X/y/keys

    ## How it is built
    `build_full_panel_dataset(...)`:
    - computes the list of valid month-end cutoffs:
      - earliest cutoff requires `min_history_months` of history (e.g. 6 months)
      - latest cutoff is `max_date - horizon_months` so the label window exists
    - calls `build_panel_dataset(...)` which:
      - builds a feature matrix at each cutoff using `create_feature_matrix(df, cutoff)`
      - builds the future label with `make_future_tx_count(df, cutoff, horizon_months)`
      - stacks all cutoffs into one big X/y/keys
    - builds `cutoff_slices` by grouping the stacked `keys` by `cutoff_date`

    ## Why slicing is correct (and fast)
    The panel is stacked in cutoff order. Each cutoff occupies a contiguous block of
    rows in X/y/keys. `cutoff_slices` stores the exact block boundaries, so selecting
    a cutoff does not require recomputing any features or re-grouping the raw data.

    ## Important nuance: customer coverage can vary by cutoff
    `create_feature_matrix` drops customers without history before the cutoff, so the
    set of customers can grow over time. This is expected and consistent with the
    original notebook logic.
    """

    X: pd.DataFrame
    y: np.ndarray
    keys: pd.DataFrame
    cutoffs: list[pd.Timestamp]
    cutoff_slices: dict[pd.Timestamp, tuple[int, int]]  # [start, end)
    horizon_months: int

    def for_cutoff(self, cutoff):
        """Return (X, y, keys) for a single cutoff by slicing the stacked panel."""
        c = _norm_ts(cutoff)
        start, end = self.cutoff_slices[c]
        return self.X.iloc[start:end].reset_index(drop=True), self.y[start:end].copy(), self.keys.iloc[
            start:end
        ].reset_index(drop=True)

    def for_cutoffs(self, cutoffs: Iterable):
        """Return (X, y, keys) for multiple cutoffs by concatenating per-cutoff slices."""
        parts_X: list[pd.DataFrame] = []
        parts_y: list[np.ndarray] = []
        parts_k: list[pd.DataFrame] = []
        for c in [_norm_ts(c) for c in cutoffs]:
            Xc, yc, kc = self.for_cutoff(c)
            parts_X.append(Xc)
            parts_y.append(yc)
            parts_k.append(kc)
        return (
            pd.concat(parts_X, ignore_index=True),
            np.concatenate(parts_y, axis=0),
            pd.concat(parts_k, ignore_index=True),
        )


def _build_cutoff_slices(keys: pd.DataFrame) -> dict[pd.Timestamp, tuple[int, int]]:
    # keys are stacked in cutoff order, so group sizes (sort=False) preserve block order.
    sizes = keys.groupby("cutoff_date", sort=False).size()
    out: dict[pd.Timestamp, tuple[int, int]] = {}
    start = 0
    for cutoff, n in sizes.items():
        end = start + int(n)
        out[_norm_ts(cutoff)] = (start, end)
        start = end
    return out


def generate_all_valid_cutoffs(
    df: pd.DataFrame,
    *,
    horizon_months: int,
    min_history_months: int = 6,
    freq: str = "ME",
    start_cutoff: Optional[pd.Timestamp] = None,
    end_cutoff: Optional[pd.Timestamp] = None,
) -> list[pd.Timestamp]:
    """All month-end cutoffs where:
    - a full `min_history_months` lookback exists for features, and
    - the future label window `[cutoff, cutoff+horizon)` is observable in the data.
    """
    min_date = _norm_ts(df["date"].min())
    max_date = _norm_ts(df["date"].max())

    if start_cutoff is None:
        start_cutoff = (
            (min_date + pd.DateOffset(months=min_history_months)).normalize()
            + pd.offsets.MonthEnd(0)
        )
    if end_cutoff is None:
        end_cutoff = (
            (_norm_ts(max_date - pd.DateOffset(months=horizon_months)))
            + pd.offsets.MonthEnd(0)
        )

    start_cutoff = _norm_ts(start_cutoff)
    end_cutoff = _norm_ts(end_cutoff)

    cutoffs = list(pd.date_range(start=start_cutoff, end=end_cutoff, freq=freq))
    return [_norm_ts(c) for c in cutoffs]


def build_full_panel_dataset(
    df: pd.DataFrame,
    *,
    horizon_months: int,
    min_history_months: int = 6,
    freq: str = "ME",
    start_cutoff: Optional[pd.Timestamp] = None,
    end_cutoff: Optional[pd.Timestamp] = None,
) -> PanelDataset:
    """Build a comprehensive no-leakage panel for all valid cutoffs (once)."""
    cutoffs = generate_all_valid_cutoffs(
        df,
        horizon_months=horizon_months,
        min_history_months=min_history_months,
        freq=freq,
        start_cutoff=start_cutoff,
        end_cutoff=end_cutoff,
    )

    X, y, keys = build_panel_dataset(df, cutoffs, horizon_months=horizon_months)
    cutoff_slices = _build_cutoff_slices(keys)
    return PanelDataset(
        X=X,
        y=y,
        keys=keys,
        cutoffs=cutoffs,
        cutoff_slices=cutoff_slices,
        horizon_months=horizon_months,
    )


def build_or_load_full_panel_dataset(
    df: pd.DataFrame,
    *,
    horizon_months: int,
    min_history_months: int = 6,
    cache_path: str | Path = DATA_PROCESSED / "panel_dataset.joblib",
    force_rebuild: bool = False,
) -> PanelDataset:
    """Cache the full panel on disk so repeated cutoff experiments are fast.

    The cached file is a serialized `PanelDataset` object. The typical naming convention
    used in the notebook is:
    - `panel_h{horizon}_mh{min_history}.joblib`
    """
    cache_path = Path(cache_path)
    # Avoid notebook CWD issues: resolve relative paths against PROJECT_ROOT.
    if not cache_path.is_absolute():
        cache_path = PROJECT_ROOT / cache_path
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists() and not force_rebuild:
        panel = joblib.load(cache_path)
        if isinstance(panel, PanelDataset) and panel.horizon_months == horizon_months:
            return panel

    panel = build_full_panel_dataset(
        df,
        horizon_months=horizon_months,
        min_history_months=min_history_months,
    )
    joblib.dump(panel, cache_path)
    return panel

