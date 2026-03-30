"""
src/models/forecaster.py

XGBoost forecasting model for DK1/DK2 day-ahead electricity prices.

Pipeline:
  1. Tune hyperparameters once on the initial training set using
     RandomizedSearchCV with TimeSeriesSplit inner CV.
  2. Fix the best hyperparameters and re-train on each expanding walk-forward
     fold — NO tuning on test data.
  3. Collect per-fold metrics (MAE, RMSE, MAPE, directional accuracy).
  4. Collect out-of-sample predictions alongside actual prices.
  5. Save trained models, metrics CSV, forecasts parquet, and best_params.json.

Features used (all from phase3.md):
  Lag:      price_lag_1h, price_lag_2h, price_lag_24h, price_lag_48h,
            price_lag_168h
  Rolling:  price_rolling_mean_24h, price_rolling_std_24h
  Calendar: hour_sin, hour_cos, weekday_sin, weekday_cos, month_sin,
            month_cos, is_weekend, is_danish_holiday
  Weather:  temp_mean_dk, wind_speed_mean
  Gen:      wind_total_mw, solar_mw

Usage::

    python -m src.models.forecaster              # runs DK1 then DK2
    python -m src.models.forecaster --zone DK1
"""

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from xgboost import XGBRegressor

from src.models.validation import (
    _directional_accuracy,
    _mae,
    _mape,
    _rmse,
    walk_forward_split,
)

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
# Paths & constants
# ---------------------------------------------------------------------------

PROJECT_ROOT    = Path(__file__).resolve().parents[2]
RESULTS_DIR     = PROJECT_ROOT / "data" / "results"
PROCESSED_DIR   = PROJECT_ROOT / "data" / "processed"

TARGET_COL         = "price_next_24h"
INITIAL_TRAIN_END  = "2023-06-30"
RANDOM_SEED        = 42

#: Number of random hyperparameter combinations evaluated during tuning.
TUNE_N_ITER   = 20
#: Number of inner TimeSeriesSplit folds used during hyperparameter search.
TUNE_N_SPLITS = 5

# ---------------------------------------------------------------------------
# Feature specification
# ---------------------------------------------------------------------------

#: Columns that exist in the parquet as-is.
_PARQUET_FEATURES = [
    # Lag features
    "price_lag_1h",
    "price_lag_2h",
    "price_lag_24h",
    "price_lag_48h",
    "price_lag_168h",
    # Rolling statistics
    "price_rolling_mean_24h",
    "price_rolling_std_24h",
    # Cyclical calendar
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    # Binary calendar
    "is_weekend",
    "is_danish_holiday",
    # Weather (parquet originals, used for wind_speed_mean derivation)
    "wind_speed_aarhus",
    "wind_speed_cph",
    "temp_mean_dk",
    # Generation
    "wind_total_mw",
    "solar_mw",
    # Needed to derive weekday_sin/cos
    "day_of_week",
]

#: Final feature columns fed to XGBoost after preprocessing.
XGB_FEATURES = [
    "price_lag_1h",
    "price_lag_2h",
    "price_lag_24h",
    "price_lag_48h",
    "price_lag_168h",
    "price_rolling_mean_24h",
    "price_rolling_std_24h",
    "hour_sin",
    "hour_cos",
    "weekday_sin",       # derived from day_of_week
    "weekday_cos",       # derived from day_of_week
    "month_sin",
    "month_cos",
    "is_weekend",
    "is_danish_holiday",
    "temp_mean_dk",
    "wind_speed_mean",   # derived from aarhus + cph average
    "wind_total_mw",
    "solar_mw",
]

#: Hyperparameter search space (phase3.md spec).
PARAM_GRID = {
    "n_estimators":      [200, 500, 1000],
    "max_depth":         [4, 6, 8],
    "learning_rate":     [0.01, 0.05, 0.1],
    "subsample":         [0.8],
    "colsample_bytree":  [0.8],
    "min_child_weight":  [3, 5],
}

# ---------------------------------------------------------------------------
# Feature preprocessing
# ---------------------------------------------------------------------------

def prepare_xgb_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive cyclical weekday encodings and mean wind speed from raw parquet columns.

    XGBoost handles NaNs natively, so missing values are intentionally left as
    NaN rather than imputed (unlike the Ridge baseline which fills with 0).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``day_of_week``, ``wind_speed_aarhus``, ``wind_speed_cph``.

    Returns
    -------
    pd.DataFrame
        Subset of df with exactly the columns in :data:`XGB_FEATURES`.
    """
    out = df.copy()
    out["weekday_sin"]    = np.sin(2 * np.pi * out["day_of_week"] / 7)
    out["weekday_cos"]    = np.cos(2 * np.pi * out["day_of_week"] / 7)
    out["wind_speed_mean"] = (out["wind_speed_aarhus"] + out["wind_speed_cph"]) / 2
    return out[XGB_FEATURES]


# ---------------------------------------------------------------------------
# Hyperparameter tuning
# ---------------------------------------------------------------------------

def tune_hyperparameters(
    train_df: pd.DataFrame,
    n_iter: int = 20,
    n_splits: int = 5,
) -> dict:
    """
    Run RandomizedSearchCV on the initial training set to find the best
    XGBoost hyperparameters.

    TimeSeriesSplit is used for inner CV so no future data contaminates
    the search.  Tuning happens ONCE — the best params are then fixed for
    all walk-forward folds.

    Parameters
    ----------
    train_df : pd.DataFrame
        Initial training set (rows up to ``INITIAL_TRAIN_END``).
    n_iter : int
        Number of random parameter combinations to evaluate.  Default: 20.
    n_splits : int
        Number of TimeSeriesSplit folds for inner CV.  Default: 5.

    Returns
    -------
    dict
        Best hyperparameters from the search (keys match XGBRegressor params).
    """
    log.info("Starting hyperparameter search | n_iter=%d | inner_splits=%d", n_iter, n_splits)

    X = prepare_xgb_features(train_df)
    y = train_df[TARGET_COL]

    # Drop rows where target is NaN (shouldn't be any after engineer.py, but defensive)
    valid = y.notna()
    X, y = X[valid], y[valid]

    base_model = XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",          # fast histogram-based
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbosity=0,
    )

    tscv = TimeSeriesSplit(n_splits=n_splits)

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=PARAM_GRID,
        n_iter=n_iter,
        scoring="neg_mean_absolute_error",
        cv=tscv,
        refit=True,
        random_state=RANDOM_SEED,
        n_jobs=1,       # parallelism is inside XGBoost (n_jobs=-1 above)
        verbose=1,
    )

    search.fit(X, y)

    best = search.best_params_
    best_score = -search.best_score_
    log.info("Best hyperparameters: %s", best)
    log.info("Best inner-CV MAE: %.3f EUR/MWh", best_score)

    return best


# ---------------------------------------------------------------------------
# XGBoost model wrapper
# ---------------------------------------------------------------------------

class XGBoostForecaster:
    """
    Sklearn-compatible XGBoost wrapper that applies :func:`prepare_xgb_features`
    before fitting and predicting.

    Parameters
    ----------
    params : dict
        XGBoost hyperparameters (e.g. from :func:`tune_hyperparameters`).
        If None, uses sensible defaults.
    """

    def __init__(self, params: dict | None = None) -> None:
        """
        Parameters
        ----------
        params : dict, optional
            XGBoost hyperparameters.  Keys matching XGBRegressor args
            (e.g. n_estimators, max_depth).  Merged with sensible defaults;
            provided keys take precedence.
        """
        base_params = {
            "objective":       "reg:squarederror",
            "tree_method":     "hist",
            "random_state":    RANDOM_SEED,
            "n_jobs":          -1,
            "verbosity":       0,
        }
        if params:
            base_params.update(params)
        self._params = base_params
        self._model: XGBRegressor | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "XGBoostForecaster":
        """
        Fit XGBoost on the training set after feature preprocessing.

        Parameters
        ----------
        X : pd.DataFrame
            Must contain all raw parquet columns needed by
            :func:`prepare_xgb_features`.
        y : pd.Series
            Target values.

        Returns
        -------
        self
        """
        X_proc = prepare_xgb_features(X)
        self._model = XGBRegressor(**self._params)
        self._model.fit(X_proc, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions after feature preprocessing.

        Parameters
        ----------
        X : pd.DataFrame
            Same schema as training data.

        Returns
        -------
        np.ndarray
        """
        if self._model is None:
            raise RuntimeError("Call fit() before predict().")
        X_proc = prepare_xgb_features(X)
        return self._model.predict(X_proc)

    @property
    def booster(self) -> XGBRegressor | None:
        """Underlying XGBRegressor, accessible after fit()."""
        return self._model


# ---------------------------------------------------------------------------
# Walk-forward loop with forecast collection
# ---------------------------------------------------------------------------

def run_walk_forward(
    model_params: dict,
    df: pd.DataFrame,
    zone: str,
) -> tuple[pd.DataFrame, pd.DataFrame, XGBoostForecaster | None]:
    """
    Run the tuned XGBoost through all walk-forward folds.

    The model is re-fitted from scratch on the expanding training set at
    each fold using the fixed ``model_params``.  Raw predictions and actuals
    are collected alongside metrics.

    Parameters
    ----------
    model_params : dict
        Best hyperparameters from :func:`tune_hyperparameters`.
    df : pd.DataFrame
        Full feature DataFrame for one bidding zone.
    zone : str
        Label used in logging (e.g. ``"DK1"``).

    Returns
    -------
    metrics_df : pd.DataFrame
        One row per fold with model_name, fold_id, train_start, train_end,
        test_start, test_end, mae, rmse, mape, directional_accuracy.
    forecasts_df : pd.DataFrame
        One row per test-set hour with timestamp_utc, actual, forecast, fold_id.
    """
    metric_records: list[dict]   = []
    forecast_records: list[dict] = []

    trained_model: XGBoostForecaster | None = None  # keep last trained model

    for fold_id, train_df, test_df in walk_forward_split(
        df, INITIAL_TRAIN_END
    ):
        nan_count = train_df[_PARQUET_FEATURES].isna().sum().sum()
        if nan_count:
            log.warning("Fold %02d [%s] | NaNs in training features: %d (XGBoost handles natively)", fold_id, zone, nan_count)

        model = XGBoostForecaster(params=model_params)
        model.fit(train_df[_PARQUET_FEATURES], train_df[TARGET_COL])

        X_test  = test_df[_PARQUET_FEATURES]
        y_test  = test_df[TARGET_COL].values
        preds   = model.predict(X_test)

        # Metrics
        metric_records.append({
            "model_name":           "xgboost",
            "fold_id":              fold_id,
            "train_start":          train_df["timestamp_utc"].min(),
            "train_end":            train_df["timestamp_utc"].max(),
            "test_start":           test_df["timestamp_utc"].min(),
            "test_end":             test_df["timestamp_utc"].max(),
            "mae":                  _mae(y_test, preds),
            "rmse":                 _rmse(y_test, preds),
            "mape":                 _mape(y_test, preds),
            "directional_accuracy": _directional_accuracy(y_test, preds),
        })

        # Forecasts
        for ts, actual, forecast in zip(
            test_df["timestamp_utc"].values, y_test, preds
        ):
            forecast_records.append({
                "timestamp_utc": ts,
                "actual":        float(actual),
                "forecast":      float(forecast),
                "fold_id":       fold_id,
            })

        trained_model = model  # keep reference to last model

        log.info(
            "Fold %02d [%s] | MAE: %.2f | RMSE: %.2f",
            fold_id, zone,
            metric_records[-1]["mae"],
            metric_records[-1]["rmse"],
        )

    metrics_df   = pd.DataFrame(metric_records)
    forecasts_df = pd.DataFrame(forecast_records)

    if not metrics_df.empty:
        log.info(
            "XGBoost [%s] | %d folds | mean MAE: %.2f | mean RMSE: %.2f",
            zone, len(metrics_df),
            metrics_df["mae"].mean(),
            metrics_df["rmse"].mean(),
        )

    return metrics_df, forecasts_df, trained_model


# ---------------------------------------------------------------------------
# Main pipeline for one zone
# ---------------------------------------------------------------------------

def run_zone(zone: str) -> None:
    """
    Full XGBoost pipeline for one bidding zone.

    Steps:
    1. Load feature parquet.
    2. Tune hyperparameters on the initial training set.
    3. Save best params to data/results/best_params.json.
    4. Run tuned model through walk-forward folds.
    5. Append metrics to data/results/model_metrics.csv.
    6. Save forecasts to data/results/forecasts_{zone.lower()}.parquet.
    7. Save final trained model to data/results/xgb_{zone.lower()}.joblib.

    Parameters
    ----------
    zone : str
        ``"DK1"`` or ``"DK2"``.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    parquet_path = PROCESSED_DIR / f"features_{zone.lower()}.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Feature parquet not found: {parquet_path}. Run engineer.py first."
        )

    df = pd.read_parquet(parquet_path)
    log.info("[%s] Loaded %d rows", zone, len(df))

    # ---- Step 1: Tune on initial training set --------------------------------
    initial_train = df[df["timestamp_utc"] <= pd.Timestamp(INITIAL_TRAIN_END)]
    log.info("[%s] Initial training set: %d rows (up to %s)", zone, len(initial_train), INITIAL_TRAIN_END)

    if initial_train.empty:
        log.warning("[%s] Initial training set is empty — using full dataset for tuning.", zone)
        initial_train = df

    best_params = tune_hyperparameters(initial_train, n_iter=TUNE_N_ITER, n_splits=TUNE_N_SPLITS)

    # ---- Step 2: Save best params -------------------------------------------
    params_path = RESULTS_DIR / "best_params.json"
    # Merge with any existing params from other zones
    all_params: dict = {}
    if params_path.exists():
        with open(params_path) as f:
            all_params = json.load(f)
    all_params[zone] = best_params
    with open(params_path, "w") as f:
        json.dump(all_params, f, indent=2)
    log.info("[%s] Best params saved to %s", zone, params_path)

    # ---- Step 3: Walk-forward evaluation ------------------------------------
    log.info("[%s] Starting walk-forward evaluation…", zone)
    metrics_df, forecasts_df, _final_model = run_walk_forward(best_params, df, zone)

    # ---- Step 4: Append metrics to model_metrics.csv ------------------------
    metrics_path = RESULTS_DIR / "model_metrics.csv"
    save_metrics = metrics_df.rename(columns={"model_name": "model"})
    save_metrics.insert(0, "zone", zone)

    if metrics_path.exists():
        existing = pd.read_csv(metrics_path)
        # Drop any previous xgboost rows for this zone to avoid duplicates
        existing = existing[~((existing["model"] == "xgboost") & (existing["zone"] == zone))]
        combined = pd.concat([existing, save_metrics], ignore_index=True)
    else:
        combined = save_metrics

    combined.to_csv(metrics_path, index=False)
    log.info("[%s] Metrics appended to %s", zone, metrics_path)

    # ---- Step 5: Save forecasts ---------------------------------------------
    forecasts_path = RESULTS_DIR / f"forecasts_{zone.lower()}.parquet"
    forecasts_df.to_parquet(forecasts_path, index=False)
    log.info("[%s] Forecasts saved to %s (%d rows)", zone, forecasts_path, len(forecasts_df))

    # ---- Step 6: Save final trained model -----------------------------------
    # Re-train on the full dataset (all available data) for the saved model
    full_model = XGBoostForecaster(params=best_params)
    full_model.fit(df[_PARQUET_FEATURES], df[TARGET_COL])
    model_path = RESULTS_DIR / f"xgb_{zone.lower()}.joblib"
    joblib.dump(full_model, model_path)
    log.info("[%s] Model saved to %s", zone, model_path)

    # ---- Summary ------------------------------------------------------------
    if not metrics_df.empty:
        log.info(
            "[%s] Walk-forward summary | %d folds | mean MAE: %.2f EUR/MWh | "
            "mean RMSE: %.2f EUR/MWh | mean dir. acc: %.1f%%",
            zone, len(metrics_df),
            metrics_df["mae"].mean(),
            metrics_df["rmse"].mean(),
            metrics_df["directional_accuracy"].mean(),
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate XGBoost forecaster.")
    parser.add_argument(
        "--zone",
        choices=["DK1", "DK2", "both"],
        default="both",
        help="Bidding zone to run (default: both)",
    )
    args = parser.parse_args()

    np.random.seed(RANDOM_SEED)

    zones = ["DK1", "DK2"] if args.zone == "both" else [args.zone]
    for z in zones:
        log.info("=" * 60)
        log.info("Starting XGBoost pipeline for %s", z)
        log.info("=" * 60)
        run_zone(z)
