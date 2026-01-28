## Quod_THA — transaction count forecasting (2019)

Predict the **number of customer transactions in the next 3 months** for month-end cutoffs in **2019**, using **strict no-leakage** time splits (features use only `date < cutoff`, targets are in `[cutoff, cutoff + 3 months)`).

### Models (brief, as used here)

- **Baseline A**: naive predictor that shows strong **recency effects** (uses recent transaction counts as the forecast).
- **Tweedie (GLM)**: a smooth GLM; shows how **smooth models are bad** for this dataset.
- **XGBoost (Poisson)**: tries to model **sharp / non-smooth** behavior; performs **near the naive Baseline A**.

## CLI usage

Run all commands from the **project root**.

### Install (Poetry)

```bash
poetry install
```

### Install (no Poetry: venv + pip)

```bash
python -m venv venv
source venv/bin/activate
pip install -U pip
pip install .
```

### `scripts/benchmark.py`

Benchmarks BaselineA / Tweedie / XGBPoisson on a single **2019** cutoff month.

```bash
poetry run scripts/benchmark.py --cutoff 2019-07
```

- **--cutoff** (required): cutoff month (`YYYY-MM`) or date (`YYYY-MM-DD`). Must be in 2019.
- **--horizon-months** (optional, default `3`): forecast horizon.
- **--min-history-months** (optional, default `6`): minimum history for features/panel.
- **--xgb-n-estimators-max** (optional, default `5000`): upper bound for early stopping (final `n_estimators` chosen by validation).

### `scripts/predict.py`

Predicts next-3-month purchases for **one customer** at a cutoff month-end and returns the **actual** realized count.

```bash
poetry run scripts/predict.py --cutoff 2019-07 --customer-id 9447359 --model xgb --json
```

- **--cutoff** (required): cutoff month (`YYYY-MM`) or date (`YYYY-MM-DD`).
- **--customer-id** (required): customer id (int).
- **--model** (required): one of `baseline`, `tweedie`, `xgb`.
- **--horizon-months** (optional, default `3`): forecast horizon.
- **--min-history-months** (optional, default `6`): minimum history for features/panel.
- **--xgb-n-estimators-max** (optional, default `5000`): upper bound for early stopping (final `n_estimators` chosen by validation).
- **--force-retrain** (optional): ignore saved model caches and retrain.
- **--json** (optional): print JSON output.

