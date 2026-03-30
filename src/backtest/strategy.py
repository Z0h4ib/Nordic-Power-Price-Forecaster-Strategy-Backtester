"""
src/backtest/strategy.py

Trading signal generation for the Nordic Power Price Forecaster backtester.

Strategy logic
--------------
The strategy compares the XGBoost 24h-ahead price forecast against a forward
price proxy — the rolling 24-hour mean of observed actual prices, representing
the best estimate a trader could form of where the market is trading right now.

    forward_price[t] = mean(actual[t-24 : t-1])   ← strictly past data only

Signal rule (default threshold = 5%):

    signal = +1   if forecast > forward_price × (1 + threshold)   → go long
    signal = -1   if forecast < forward_price × (1 - threshold)   → go short
    signal =  0   otherwise                                         → stay flat

The threshold avoids overtrading noise — only take a position when the model's
view diverges meaningfully from the observed market level.

Limitations acknowledged
------------------------
- The forward price proxy is a simplification. In production you would pull
  actual forward curve quotes (e.g. ICE EEX DK1/DK2 day-ahead forward).
- The rolling window is seeded only from data in the parquet. For the first
  24 rows the window is shorter (min_periods=1); these early signals may be
  noisier. A production system would seed the window from historical prices
  loaded separately.

Usage::

    python -m src.backtest.strategy            # DK1, threshold=0.05
    python -m src.backtest.strategy --zone DK2 --threshold 0.02
"""

import argparse
import logging
from pathlib import Path

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
# Paths & constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR  = PROJECT_ROOT / "data" / "results"

FORWARD_WINDOW   = 24    # hours of history used to compute the forward proxy
DEFAULT_THRESHOLD = 0.05


# ---------------------------------------------------------------------------
# Forward price proxy
# ---------------------------------------------------------------------------

def compute_forward_price(
    actual: pd.Series,
    window: int = FORWARD_WINDOW,
) -> pd.Series:
    """
    Compute the forward price proxy as a rolling mean of past actual prices.

    At each hour t the forward price is the mean of the ``window`` most
    recent observed actual prices *excluding* hour t itself.  This guarantees
    strict no-look-ahead: the signal at t is formed only from information
    available before t.

    The implementation uses ``shift(1)`` applied before the rolling operation
    so the window [t-window, t-1] is used rather than [t-window+1, t].

    Parameters
    ----------
    actual : pd.Series
        Hourly actual prices, indexed chronologically.
    window : int
        Number of past hours to average.  Default: 24.

    Returns
    -------
    pd.Series
        Rolling forward price proxy, same index as ``actual``.
        The first ``window`` values will use fewer than ``window``
        observations (min_periods=1).
    """
    # shift(1) moves values one step forward in time so that at position t
    # we see actual[t-1].  The subsequent rolling(window) then covers
    # actual[t-1] back to actual[t-window], never touching actual[t].
    forward = actual.shift(1).rolling(window, min_periods=1).mean()
    log.debug("Forward price computed | window=%d | NaN count=%d", window, forward.isna().sum())
    return forward


# ---------------------------------------------------------------------------
# Signal generation
# ---------------------------------------------------------------------------

def generate_signals(
    forecasts_df: pd.DataFrame,
    threshold: float = DEFAULT_THRESHOLD,
) -> pd.DataFrame:
    """
    Generate directional trading signals from XGBoost forecasts.

    Compares the model forecast against the rolling forward price proxy and
    emits a +1 (long), -1 (short), or 0 (flat) signal for each hour.

    Parameters
    ----------
    forecasts_df : pd.DataFrame
        Must contain columns: ``timestamp_utc``, ``actual``, ``forecast``.
        Rows must be sorted chronologically.
    threshold : float
        Fractional threshold for signal generation.  Only trade when the
        forecast deviates from the forward price by more than this fraction.
        Default: 0.05 (5%).

    Returns
    -------
    pd.DataFrame
        Input DataFrame with three additional columns:

        ``forward_price``
            Rolling 24-hour mean of past actual prices.
        ``signal``
            Integer: +1 (long), -1 (short), 0 (flat).
        ``signal_label``
            Human-readable string: ``"long"``, ``"short"``, ``"flat"``.

    Notes
    -----
    - The forward price uses only past actual prices (strict no-leakage).
    - Rows where ``forward_price`` is NaN (can occur if the first row has no
      history) receive ``signal = 0`` (flat).
    - The threshold is applied symmetrically: abs(forecast/forward - 1) > threshold.
    """
    df = forecasts_df.copy().sort_values("timestamp_utc").reset_index(drop=True)

    df["forward_price"] = compute_forward_price(df["actual"])

    # Signal logic — vectorised for performance
    forecast     = df["forecast"].values
    forward      = df["forward_price"].values

    long_mask  = forecast > forward * (1.0 + threshold)
    short_mask = forecast < forward * (1.0 - threshold)

    # Default flat; apply long/short where conditions are met
    signal = np.zeros(len(df), dtype=np.int8)
    signal[long_mask]  =  1
    signal[short_mask] = -1

    # Flat where forward_price is NaN (first row has no history)
    nan_mask = np.isnan(forward)
    signal[nan_mask] = 0

    df["signal"] = signal
    df["signal_label"] = pd.Categorical(
        np.where(signal == 1, "long", np.where(signal == -1, "short", "flat")),
        categories=["long", "short", "flat"],
    )

    n_long  = int((signal ==  1).sum())
    n_short = int((signal == -1).sum())
    n_flat  = int((signal ==  0).sum())
    n_total = len(df)
    pct_in_market = (n_long + n_short) / n_total * 100

    log.info(
        "Signals | threshold=%.0f%% | long=%d (%.1f%%) | short=%d (%.1f%%) | "
        "flat=%d (%.1f%%) | in-market=%.1f%%",
        threshold * 100,
        n_long,  n_long  / n_total * 100,
        n_short, n_short / n_total * 100,
        n_flat,  n_flat  / n_total * 100,
        pct_in_market,
    )

    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate trading signals.")
    parser.add_argument("--zone",      default="DK1", choices=["DK1", "DK2"])
    parser.add_argument("--threshold", default=0.05,  type=float)
    args = parser.parse_args()

    parquet_path = RESULTS_DIR / f"forecasts_{args.zone.lower()}.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Forecasts not found: {parquet_path}. Run forecaster.py first."
        )

    df = pd.read_parquet(parquet_path)
    log.info(
        "Loaded %d forecast rows for %s | %s → %s",
        len(df), args.zone,
        df["timestamp_utc"].min().date(),
        df["timestamp_utc"].max().date(),
    )

    result = generate_signals(df, threshold=args.threshold)

    # ---- Summary output ─────────────────────────────────────────────────────
    n_long  = (result["signal"] ==  1).sum()
    n_short = (result["signal"] == -1).sum()
    n_flat  = (result["signal"] ==  0).sum()
    n_total = len(result)

    log.info("=" * 55)
    log.info("  Signal summary — %s | threshold=%.0f%%", args.zone, args.threshold * 100)
    log.info("=" * 55)
    log.info("  Total hours    : %d", n_total)
    log.info("  Long  (+1)     : %d  (%.1f%%)", n_long,  n_long  / n_total * 100)
    log.info("  Short (-1)     : %d  (%.1f%%)", n_short, n_short / n_total * 100)
    log.info("  Flat  ( 0)     : %d  (%.1f%%)", n_flat,  n_flat  / n_total * 100)
    log.info("  %% in market   : %.1f%%", (n_long + n_short) / n_total * 100)
    log.info("=" * 55)

    # ── Forward price sanity check ────────────────────────────────────────────
    log.info(
        "Forward price stats | mean=%.2f | std=%.2f | min=%.2f | max=%.2f",
        result["forward_price"].mean(),
        result["forward_price"].std(),
        result["forward_price"].min(),
        result["forward_price"].max(),
    )

    # ── Leakage check: forward_price at t must not equal actual at t ──────────
    same_as_actual = (result["forward_price"] == result["actual"]).sum()
    if same_as_actual > 1:   # allow 1 for the edge-case first row
        log.warning(
            "LEAKAGE WARNING: forward_price equals actual at %d rows — "
            "check the shift direction.", same_as_actual
        )
    else:
        log.info("Leakage check passed — forward_price never equals current actual ✓")

    # ── Sample output ─────────────────────────────────────────────────────────
    log.info(
        "Sample rows:\n%s",
        result[["timestamp_utc", "actual", "forecast", "forward_price", "signal_label"]]
        .head(6)
        .to_string(index=False),
    )
