"""
src/models/baseline.py

Two baseline forecasting models for DK1/DK2 day-ahead electricity prices.

Model 1 — Persistence
    Forecast at hour t = actual price at hour t-24 (price_lag_24h).
    No fitting required — the prediction is simply the lag feature.
    Sets the performance floor: any useful model must beat this.

Model 2 — Ridge regression
    Regularised linear model using a hand-picked feature set.
    Establishes whether engineered features add value over pure persistence.
    Coefficients are stored per fold to check for temporal stability.

Both models implement the sklearn fit/predict interface so they plug directly
into :func:`src.models.validation.evaluate_model`.

Usage::

    python -m src.models.baseline          # runs DK1 walk-forward, prints metrics
    python -m src.models.baseline --zone DK2
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Feature sets
# ---------------------------------------------------------------------------

#: Single feature needed by the persistence model.
PERSISTENCE_FEATURE = "price_lag_24h"

#: Raw columns from the parquet that the Ridge model consumes.
#: weekday_sin/cos are derived from day_of_week inside the wrapper.
#: wind_speed_mean is derived from the two location columns inside the wrapper.
RIDGE_RAW_COLS = [
    "price_lag_24h",
    "price_lag_168h",
    "hour_sin",
    "hour_cos",
    "day_of_week",        # → weekday_sin, weekday_cos
    "month_sin",
    "month_cos",
    "temp_mean_dk",       # proxy for temperature_c
    "wind_speed_aarhus",  # \
    "wind_speed_cph",     #  → wind_speed_mean
    "wind_total_mw",
    "solar_mw",
]

#: Target column in the parquet (price 24 h ahead).
TARGET_COL = "price_next_24h"

#: Walk-forward parameters (match phase3.md).
INITIAL_TRAIN_END = "2023-06-30"


# ---------------------------------------------------------------------------
# Feature preprocessing
# ---------------------------------------------------------------------------

def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add weekday cyclical encodings and mean wind speed to a feature DataFrame.

    These columns are required by :class:`RidgeModel` but are not stored in
    the parquet (the parquet only has ``day_of_week`` as an integer and
    separate per-location wind speeds).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``day_of_week``, ``wind_speed_aarhus``, ``wind_speed_cph``.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with three new columns appended in-place:
        ``weekday_sin``, ``weekday_cos``, ``wind_speed_mean``.
    """
    df = df.copy()
    df["weekday_sin"]    = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["weekday_cos"]    = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["wind_speed_mean"] = (
        df["wind_speed_aarhus"].fillna(0) + df["wind_speed_cph"].fillna(0)
    ) / 2
    return df


# ---------------------------------------------------------------------------
# Model 1 — Persistence
# ---------------------------------------------------------------------------

class PersistenceModel:
    """
    Naive persistence baseline: predict price_lag_24h for price_next_24h.

    "The price tomorrow at this hour will equal the price at the same hour
    today" — i.e., the day-ahead price 24 h from now is approximated by
    the known price from 24 h ago.

    The model requires no fitting and is deterministic: predict() simply
    returns the ``price_lag_24h`` column from X.

    Methods
    -------
    fit(X, y) : no-op, returns self
    predict(X) : returns X["price_lag_24h"].values
    """

    def fit(self, _X: pd.DataFrame, _y: pd.Series) -> "PersistenceModel":
        """No-op — persistence model has no learnable parameters."""
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Return price_lag_24h as the forecast.

        Parameters
        ----------
        X : pd.DataFrame
            Must contain ``price_lag_24h``.

        Returns
        -------
        np.ndarray
            Forecast values (same dtype as the lag column).
        """
        return X[PERSISTENCE_FEATURE].values


# ---------------------------------------------------------------------------
# Model 2 — Ridge regression
# ---------------------------------------------------------------------------

#: Final feature list consumed by the Ridge Pipeline after preprocessing.
RIDGE_MODEL_FEATURES = [
    "price_lag_24h",
    "price_lag_168h",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "month_sin",
    "month_cos",
    "temp_mean_dk",
    "wind_speed_mean",
    "wind_total_mw",
    "solar_mw",
]


class RidgeModel:
    """
    Regularised linear regression baseline using calendar, weather, and
    generation features.

    Internally runs :func:`add_derived_features` to create cyclical weekday
    encodings and a mean wind-speed column before fitting/predicting.
    Features are standardised via :class:`sklearn.preprocessing.StandardScaler`
    (Ridge is sensitive to scale).

    Attributes
    ----------
    pipeline : sklearn.pipeline.Pipeline
        StandardScaler → Ridge. Accessible after the first :meth:`fit` call.
    coef_history : list of dict
        One entry per :meth:`fit` call with keys ``fold_id`` and one key per
        feature holding the fitted coefficient. Useful for stability analysis.
    alpha : float
        Ridge regularisation strength (default 1.0).
    """

    def __init__(self, alpha: float = 1.0) -> None:
        """
        Parameters
        ----------
        alpha : float
            Ridge regularisation strength.  Default: 1.0.
        """
        self.alpha = alpha
        self.pipeline: Pipeline | None = None
        self.coef_history: list[dict] = []
        self._fold_counter: int = 0

    def _preprocess(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add derived columns and select final feature set."""
        X = add_derived_features(X)
        return X[RIDGE_MODEL_FEATURES].fillna(0)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RidgeModel":
        """
        Fit the StandardScaler + Ridge pipeline on the training set.

        Also appends the fitted coefficients to :attr:`coef_history`.

        Parameters
        ----------
        X : pd.DataFrame
            Must contain all columns in ``RIDGE_RAW_COLS``.
        y : pd.Series
            Target values (price_next_24h).

        Returns
        -------
        self
        """
        X_proc = self._preprocess(X)
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge",  Ridge(alpha=self.alpha)),
        ])
        self.pipeline.fit(X_proc, y)

        # Record coefficients for stability analysis
        self._fold_counter += 1
        coef_row: dict = {"fold_id": self._fold_counter}
        coef_row.update(dict(zip(RIDGE_MODEL_FEATURES, self.pipeline["ridge"].coef_)))
        self.coef_history.append(coef_row)

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions on the test set.

        Parameters
        ----------
        X : pd.DataFrame
            Must contain all columns in ``RIDGE_RAW_COLS``.

        Returns
        -------
        np.ndarray
        """
        if self.pipeline is None:
            raise RuntimeError("Call fit() before predict().")
        X_proc = self._preprocess(X)
        return self.pipeline.predict(X_proc)

    def coef_dataframe(self) -> pd.DataFrame:
        """
        Return per-fold coefficients as a tidy DataFrame.

        Returns
        -------
        pd.DataFrame
            Columns: fold_id, then one column per feature.
        """
        return pd.DataFrame(self.coef_history)


# ---------------------------------------------------------------------------
# Walk-forward evaluation runner
# ---------------------------------------------------------------------------

def run_baselines(zone: str = "DK1") -> pd.DataFrame:
    """
    Run persistence and Ridge baselines through the walk-forward loop for one
    bidding zone and return the combined metrics DataFrame.

    Parameters
    ----------
    zone : str
        ``"DK1"`` or ``"DK2"``.

    Returns
    -------
    pd.DataFrame
        Combined metrics from both models with columns:
        model_name, fold_id, train_start, train_end,
        test_start, test_end, mae, rmse, mape, directional_accuracy.
    """
    # Import here to avoid circular dependency at module level
    from src.models.validation import evaluate_model

    parquet_path = PROJECT_ROOT / "data" / "processed" / f"features_{zone.lower()}.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Feature parquet not found: {parquet_path}. Run engineer.py first."
        )

    df = pd.read_parquet(parquet_path)
    log.info("Loaded %d rows for zone %s", len(df), zone)

    # --- Persistence ---
    log.info("=== Running persistence baseline (%s) ===", zone)
    persistence = PersistenceModel()
    persistence_metrics = evaluate_model(
        model=persistence,
        df=df,
        feature_cols=[PERSISTENCE_FEATURE],
        target_col=TARGET_COL,
        model_name="persistence",
        initial_train_end=INITIAL_TRAIN_END,
    )

    # --- Ridge ---
    log.info("=== Running Ridge baseline (%s) ===", zone)
    ridge = RidgeModel(alpha=1.0)
    ridge_metrics = evaluate_model(
        model=ridge,
        df=df,
        feature_cols=RIDGE_RAW_COLS,
        target_col=TARGET_COL,
        model_name="ridge",
        initial_train_end=INITIAL_TRAIN_END,
    )

    # Log coefficient stability summary
    coef_df = ridge.coef_dataframe()
    if not coef_df.empty:
        log.info("Ridge coefficient stability (std across folds):")
        for feat in RIDGE_MODEL_FEATURES:
            if feat in coef_df.columns:
                log.info("  %-25s std=%.4f", feat, coef_df[feat].std())

    combined = pd.concat([persistence_metrics, ridge_metrics], ignore_index=True)
    return combined


def print_summary(metrics_df: pd.DataFrame) -> None:
    """
    Print a human-readable summary of mean metrics per model.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        Output of :func:`run_baselines`.
    """
    summary = (
        metrics_df
        .groupby("model_name")[["mae", "rmse", "mape", "directional_accuracy"]]
        .mean()
        .round(3)
    )
    log.info("Baseline model summary (mean across folds):\n%s", summary.to_string())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run baseline models.")
    parser.add_argument("--zone", default="DK1", choices=["DK1", "DK2"],
                        help="Bidding zone to evaluate (default: DK1)")
    args = parser.parse_args()

    metrics = run_baselines(zone=args.zone)

    print_summary(metrics)

    # Save to data/results/model_metrics.csv
    results_dir = PROJECT_ROOT / "data" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "model_metrics.csv"

    # Rename model_name → model to match phase3.md column spec
    save_df = metrics.rename(columns={"model_name": "model"})

    if out_path.exists():
        existing = pd.read_csv(out_path)
        # Remove any previous baseline rows for this zone before appending
        zone_label = args.zone
        if "zone" in existing.columns:
            # Remove only baseline rows for this zone — preserve xgboost rows
            existing = existing[
                ~((existing["zone"] == zone_label) &
                  (existing["model"].isin(["persistence", "ridge"])))
            ]
        save_df.insert(0, "zone", zone_label)
        combined_save = pd.concat([existing, save_df], ignore_index=True)
    else:
        save_df.insert(0, "zone", args.zone)
        combined_save = save_df

    combined_save.to_csv(out_path, index=False)
    log.info("Metrics saved to %s", out_path)
