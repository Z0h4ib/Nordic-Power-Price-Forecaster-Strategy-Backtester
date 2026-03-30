"""
src/models/shap_analysis.py

SHAP feature importance analysis for the DK1/DK2 XGBoost forecasting models.

Uses TreeExplainer (exact, fast for tree-based models) to compute SHAP values
on the last walk-forward test fold.  Produces:
  - A beeswarm summary plot saved as data/results/shap_summary_{zone.lower()}.png
  - A printed table of the top 10 features by mean |SHAP|

The beeswarm encodes two things simultaneously:
  - Y-axis: features ranked by mean |SHAP| (most important at top)
  - X-axis: SHAP value (positive → pushes prediction up)
  - Color: feature value (red = high, blue = low)

This is the standard exhibit for explaining tree model feature importance in
interviews — it shows not just which features matter but the direction and
nonlinearity of their effect.

Usage::

    python -m src.models.shap_analysis           # runs DK1 (default)
    python -m src.models.shap_analysis --zone DK2
    python -m src.models.shap_analysis --zone both
"""

import argparse
import logging
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

# Import the forecaster module so joblib can find XGBoostForecaster
import src.models.forecaster as _forecaster_module  # noqa: F401

# Patch __main__ so joblib can deserialise models saved from __main__ context
sys.modules["__main__"].XGBoostForecaster = _forecaster_module.XGBoostForecaster

from src.models.forecaster import (
    XGB_FEATURES,
    _PARQUET_FEATURES,
    prepare_xgb_features,
)
from src.models.validation import walk_forward_split

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR  = PROJECT_ROOT / "data" / "results"

INITIAL_TRAIN_END = "2023-06-30"


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def run_shap_analysis(zone: str = "DK1") -> pd.DataFrame:
    """
    Compute SHAP values for the saved XGBoost model on the last test fold.

    Parameters
    ----------
    zone : str
        ``"DK1"`` or ``"DK2"``.

    Returns
    -------
    pd.DataFrame
        Top 10 features ranked by mean |SHAP| with columns:
        feature, mean_abs_shap.
    """
    # ---- Load model ---------------------------------------------------------
    model_path = RESULTS_DIR / f"xgb_{zone.lower()}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}. Run forecaster.py first."
        )
    wrapper = joblib.load(model_path)
    xgb_model = wrapper.booster
    log.info("[%s] Loaded model from %s", zone, model_path)

    # ---- Load data and get last test fold -----------------------------------
    parquet_path = PROJECT_ROOT / "data" / "processed" / f"features_{zone.lower()}.parquet"
    df = pd.read_parquet(parquet_path)
    log.info("[%s] Loaded %d rows for feature data", zone, len(df))

    folds = list(walk_forward_split(df, INITIAL_TRAIN_END))
    if not folds:
        raise RuntimeError(f"No walk-forward folds found for {zone}.")

    fold_id, _train_df, test_df = folds[-1]
    log.info(
        "[%s] Using last fold (fold %d): %s → %s (%d rows)",
        zone, fold_id,
        test_df["timestamp_utc"].min().date(),
        test_df["timestamp_utc"].max().date(),
        len(test_df),
    )

    # ---- Prepare features ---------------------------------------------------
    X_test = prepare_xgb_features(test_df[_PARQUET_FEATURES])
    log.info("[%s] Feature matrix shape: %s", zone, X_test.shape)

    # ---- SHAP values --------------------------------------------------------
    log.info("[%s] Computing SHAP values with TreeExplainer…", zone)
    explainer   = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_test)  # shape: (n_samples, n_features)

    # ---- Beeswarm summary plot ----------------------------------------------
    plt.figure(figsize=(10, 7))
    shap.summary_plot(
        shap_values,
        X_test,
        feature_names=XGB_FEATURES,
        plot_type="dot",          # beeswarm
        max_display=19,           # show all 19 features
        show=False,
    )
    plt.title(
        f"SHAP Feature Importance — XGBoost [{zone}]\n"
        f"Last test fold: {test_df['timestamp_utc'].min().date()} → "
        f"{test_df['timestamp_utc'].max().date()}",
        fontsize=11,
        pad=12,
    )
    plt.tight_layout()

    plot_path = RESULTS_DIR / f"shap_summary_{zone.lower()}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("[%s] Beeswarm plot saved to %s", zone, plot_path)

    # ---- Top 10 features ----------------------------------------------------
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    ranking = (
        pd.DataFrame({"feature": XGB_FEATURES, "mean_abs_shap": mean_abs_shap})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    top10 = ranking.head(10)

    rows_str = "\n".join(
        f"  {i+1:<4} {row['feature']:<28} {row['mean_abs_shap']:>10.3f}"
        for i, row in top10.iterrows()
    )
    log.info(
        "[%s] Top 10 SHAP features (mean |SHAP|, EUR/MWh):\n%s",
        zone, rows_str,
    )

    return top10


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SHAP analysis for XGBoost forecaster.")
    parser.add_argument(
        "--zone",
        choices=["DK1", "DK2", "both"],
        default="DK1",
        help="Bidding zone to analyse (default: DK1)",
    )
    args = parser.parse_args()

    zones = ["DK1", "DK2"] if args.zone == "both" else [args.zone]
    for z in zones:
        run_shap_analysis(z)
