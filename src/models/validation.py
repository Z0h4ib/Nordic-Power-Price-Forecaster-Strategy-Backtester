"""
src/models/validation.py

Walk-forward (expanding window) validation framework for time-series forecasting.

Design decisions:
- Expanding training window (not rolling) — more data = better over time
- Non-overlapping 7-day test windows — no leakage between folds
- Models receive only a fit() / predict() interface — works with sklearn, XGBoost wrappers, or callables

Typical usage::

    from src.models.validation import walk_forward_split, evaluate_model

    for fold_id, train_df, test_df in walk_forward_split(df, "2023-06-30"):
        ...

    metrics_df = evaluate_model(model, df, feature_cols, "price_eur_mwh", "xgboost")
"""

import logging
from datetime import timedelta
from typing import Generator, Iterable, Protocol, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocol — what a "model" must expose
# ---------------------------------------------------------------------------

class ForecastModel(Protocol):
    """
    Minimal interface required by :func:`evaluate_model`.

    Any object with ``fit(X, y)`` and ``predict(X)`` satisfies this protocol —
    sklearn estimators, XGBoost wrappers, and custom classes all qualify.
    """

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        ...

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        ...


# ---------------------------------------------------------------------------
# Walk-forward split generator
# ---------------------------------------------------------------------------

def walk_forward_split(
    df: pd.DataFrame,
    initial_train_end: str,
    test_days: int = 7,
    step_days: int = 7,
) -> Generator[Tuple[int, pd.DataFrame, pd.DataFrame], None, None]:
    """
    Expanding-window walk-forward split for time-series data.

    The training window grows by ``step_days`` after each fold; the test
    window is a fixed ``test_days``-long block immediately following the
    training cutoff.

    Example timeline (test_days=7, step_days=7)::

        Fold 1: train [start … 2023-06-30], test [2023-07-01 … 2023-07-07]
        Fold 2: train [start … 2023-07-07], test [2023-07-08 … 2023-07-14]
        ...

    Parameters
    ----------
    df : pd.DataFrame
        Must have a ``timestamp_utc`` column (datetime) sorted chronologically
        within each bidding zone (or globally). Duplicate timestamps across
        bidding zones are fine — the split is applied to the global timestamp.
    initial_train_end : str
        Last date (inclusive) of the first training window, e.g. ``"2023-06-30"``.
    test_days : int
        Number of days in each test window.  Default: 7.
    step_days : int
        How many days to advance between consecutive folds.  Default: 7.

    Yields
    ------
    fold_id : int
        1-based fold counter.
    train_df : pd.DataFrame
        Rows whose ``timestamp_utc`` falls within the training window.
    test_df : pd.DataFrame
        Rows whose ``timestamp_utc`` falls within the test window.

    Notes
    -----
    - Stops yielding once the test window start date passes the last
      timestamp in ``df``.
    - Empty test windows (data gaps) are silently skipped — the training
      cutoff still advances by ``step_days``.  This handles datasets with
      large gaps (e.g. ENTSO-E outages) without stopping the walk.
    - Logs the date range and row count of every yielded fold at INFO level.
    """
    train_end = pd.Timestamp(initial_train_end)
    test_delta = timedelta(days=test_days)
    step_delta = timedelta(days=step_days)

    ts = df["timestamp_utc"]
    data_end = ts.max()

    fold_id = 0

    while True:
        test_start = train_end + timedelta(hours=1)
        test_end   = train_end + test_delta  # inclusive upper bound (by date)

        # Stop once the test window would start beyond available data
        if test_start > data_end:
            break

        # Cap test_end at the last available timestamp
        effective_test_end = min(test_end, data_end)

        train_mask = ts <= train_end
        test_mask  = (ts >= test_start) & (ts <= effective_test_end)

        train_df = df.loc[train_mask].copy()
        test_df  = df.loc[test_mask].copy()

        # Skip empty test windows (data gap) — advance and try the next week
        if test_df.empty:
            log.debug("Skipping empty test window %s → %s (data gap)", test_start.date(), effective_test_end.date())
            train_end += step_delta
            continue

        fold_id += 1

        log.info(
            "Fold %02d | train: %s → %s (%d rows) | test: %s → %s (%d rows)",
            fold_id,
            train_df["timestamp_utc"].min().date(),
            train_df["timestamp_utc"].max().date(),
            len(train_df),
            test_df["timestamp_utc"].min().date(),
            test_df["timestamp_utc"].max().date(),
            len(test_df),
        )

        yield fold_id, train_df, test_df

        train_end += step_delta


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(actual - predicted)))


def _rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def _mape(actual: np.ndarray, predicted: np.ndarray, eps: float = 1e-6) -> float:
    """
    Mean Absolute Percentage Error.

    Near-zero actual prices are masked (|actual| < eps) to avoid division
    explosions — common in Nordic power markets where negative/zero prices occur.
    """
    mask = np.abs(actual) >= eps
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def _directional_accuracy(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Percentage of hours where the model correctly predicts the direction of
    price change relative to the previous observation.

    Returns NaN if there are fewer than 2 observations.
    """
    if len(actual) < 2:
        return float("nan")
    actual_diff    = np.diff(actual)
    predicted_diff = np.diff(predicted)
    correct = np.sign(actual_diff) == np.sign(predicted_diff)
    return float(np.mean(correct) * 100)


# ---------------------------------------------------------------------------
# Walk-forward evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model: ForecastModel,
    df: pd.DataFrame,
    feature_cols: Iterable[str],
    target_col: str,
    model_name: str,
    initial_train_end: str = "2023-06-30",
    test_days: int = 7,
    step_days: int = 7,
) -> pd.DataFrame:
    """
    Run a model through all walk-forward folds and collect evaluation metrics.

    The model is re-fitted from scratch on the (expanding) training set for
    each fold.  Predictions are made on the test set, then four metrics are
    computed.

    Parameters
    ----------
    model : object with fit(X, y) and predict(X)
        Any sklearn-compatible estimator or custom class.
    df : pd.DataFrame
        Full feature DataFrame (single bidding zone recommended; or pass the
        full multi-zone frame and filter inside the model if desired).
    feature_cols : list of str
        Column names to use as model inputs.
    target_col : str
        Column name of the forecast target (e.g. ``"price_eur_mwh"``).
    model_name : str
        Label stored in the output DataFrame (e.g. ``"persistence"``,
        ``"ridge"``, ``"xgboost"``).
    initial_train_end : str
        Last date of the first training period.  Default: ``"2023-06-30"``.
    test_days : int
        Test window length in days.  Default: 7.
    step_days : int
        Step size between folds in days.  Default: 7.

    Returns
    -------
    pd.DataFrame
        One row per fold, columns:
        model_name, fold_id, train_start, train_end,
        test_start, test_end, mae, rmse, mape, directional_accuracy.

    Notes
    -----
    - NaN counts per fold are logged at WARNING level.
    - If a fold raises an exception it is logged and skipped (no row added).
    """
    feature_cols = list(feature_cols)
    records = []

    for fold_id, train_df, test_df in walk_forward_split(
        df, initial_train_end, test_days=test_days, step_days=step_days
    ):
        try:
            X_train = train_df[feature_cols]
            y_train = train_df[target_col]
            X_test  = test_df[feature_cols]
            y_test  = test_df[target_col].values

            # Log NaN counts
            nan_train = X_train.isna().sum().sum()
            nan_test  = X_test.isna().sum().sum()
            if nan_train or nan_test:
                log.warning(
                    "Fold %02d | NaNs in features — train: %d, test: %d",
                    fold_id, nan_train, nan_test,
                )

            model.fit(X_train, y_train)
            preds = np.asarray(model.predict(X_test), dtype=float)

            records.append({
                "model_name":          model_name,
                "fold_id":             fold_id,
                "train_start":         train_df["timestamp_utc"].min(),
                "train_end":           train_df["timestamp_utc"].max(),
                "test_start":          test_df["timestamp_utc"].min(),
                "test_end":            test_df["timestamp_utc"].max(),
                "mae":                 _mae(y_test, preds),
                "rmse":                _rmse(y_test, preds),
                "mape":                _mape(y_test, preds),
                "directional_accuracy": _directional_accuracy(y_test, preds),
            })

        except Exception as exc:
            log.error("Fold %02d | %s | unexpected error: %s", fold_id, model_name, exc)

    metrics_df = pd.DataFrame(records)
    if not metrics_df.empty:
        log.info(
            "%s | %d folds | mean MAE: %.2f | mean RMSE: %.2f",
            model_name, len(metrics_df),
            metrics_df["mae"].mean(),
            metrics_df["rmse"].mean(),
        )
    return metrics_df


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from pathlib import Path

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    parquet_path = PROJECT_ROOT / "data" / "processed" / "features_dk1.parquet"

    if not parquet_path.exists():
        log.error("Parquet not found at %s — run engineer.py first.", parquet_path)
        raise SystemExit(1)

    df = pd.read_parquet(parquet_path)
    log.info("Loaded %d rows from %s", len(df), parquet_path.name)

    fold_count = sum(
        1 for _ in walk_forward_split(df, initial_train_end="2023-06-30")
    )
    log.info(
        "Walk-forward folds generated: %d | data range: %s → %s",
        fold_count,
        df["timestamp_utc"].min().date(),
        df["timestamp_utc"].max().date(),
    )
