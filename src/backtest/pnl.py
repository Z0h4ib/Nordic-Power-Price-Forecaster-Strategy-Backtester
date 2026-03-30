"""
src/backtest/pnl.py

P&L calculation and trade log construction for the Nordic Power strategy backtester.

For each timestep the hourly P&L is computed as::

    hourly_pnl = signal × (actual_price − forward_price)

where ``signal`` ∈ {+1, −1, 0} is produced by :mod:`src.backtest.strategy`.
Results are accumulated into a cumulative P&L series and aggregated to
daily totals for downstream metric and Monte Carlo calculations.

Usage::

    python -m src.backtest.pnl            # DK1, threshold=5%
    python -m src.backtest.pnl --zone DK2 --threshold 0.02
"""

import argparse
import logging
from pathlib import Path

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
RESULTS_DIR = PROJECT_ROOT / "data" / "results"

# ---------------------------------------------------------------------------
# P&L Calculation
# ---------------------------------------------------------------------------

def calculate_pnl(signals_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Computes hourly P&L from trading signals and aggregates to daily P&L.
    
    Parameters
    ----------
    signals_df : pd.DataFrame
        DataFrame containing forecasts and signals. Must include:
        `timestamp_utc`, `signal`, `forecast`, `forward_price`, `actual`
        
    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        - A trade log DataFrame with hourly P&L and cumulative P&L.
        - A Series of daily aggregated P&L.
    """
    log.info("Calculating P&L...")
    
    df = signals_df.copy()
    
    # Compute hourly P&L
    # P&L = signal * (actual_price - forward_price)
    df["actual_price"] = df["actual"] # Rename actual to actual_price for clarity
    df["hourly_pnl"] = df["signal"] * (df["actual_price"] - df["forward_price"])
    
    # Track trades separately (where signal != 0)
    # We build the requested trade log
    trade_log = df[[
        "timestamp_utc",
        "signal",
        "forecast",
        "forward_price",
        "actual_price",
        "hourly_pnl"
    ]].copy()
    
    # Compute cumulative P&L series
    trade_log = trade_log.sort_values(by="timestamp_utc")
    trade_log["cumulative_pnl"] = trade_log["hourly_pnl"].cumsum()
    
    # Aggregate to daily P&L series
    # We need timestamp_utc to be the index to use resample, or group by date
    trade_log["date"] = pd.to_datetime(trade_log["timestamp_utc"]).dt.date
    
    # Type-safe casting to pd.Series to satisfy Pylance/Pyright
    daily_pnl_raw = trade_log.groupby("date")["hourly_pnl"].sum()
    if not isinstance(daily_pnl_raw, pd.Series):
        daily_pnl = pd.Series(daily_pnl_raw)
    else:
        daily_pnl = daily_pnl_raw
        
    daily_pnl = daily_pnl.rename("daily_pnl")
    
    return trade_log, daily_pnl

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from src.backtest.strategy import generate_signals

    parser = argparse.ArgumentParser(description="Calculate P&L for a trading strategy.")
    parser.add_argument("--zone", default="DK1", choices=["DK1", "DK2"])
    parser.add_argument("--threshold", default=0.05, type=float)
    args = parser.parse_args()

    # Load forecasts
    parquet_path = RESULTS_DIR / f"forecasts_{args.zone.lower()}.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"Forecasts not found: {parquet_path}. Run forecaster.py first.")

    forecasts_df = pd.read_parquet(parquet_path)
    
    # Generate signals
    signals_df = generate_signals(forecasts_df, threshold=args.threshold)
    
    # Compute P&L
    trade_log, daily_pnl = calculate_pnl(signals_df)
    
    # Filter to actual trades (signal != 0) for some statistics
    trades_only = trade_log[trade_log["signal"] != 0]
    total_pnl = trade_log["hourly_pnl"].sum()
    num_trades = len(trades_only)
    
    first_trade_date = trades_only["timestamp_utc"].min().date() if num_trades > 0 else None
    last_trade_date = trades_only["timestamp_utc"].max().date() if num_trades > 0 else None
    
    log.info("=" * 55)
    log.info("  P&L Summary — %s | threshold=%.0f%%", args.zone, args.threshold * 100)
    log.info("=" * 55)
    log.info("  Total P&L (EUR/MWh equivalent): %.2f", total_pnl)
    log.info("  Number of trades taken        : %d", num_trades)
    log.info("  First trade date              : %s", first_trade_date)
    log.info("  Last trade date               : %s", last_trade_date)
    log.info("=" * 55)
    
    # Save trade log
    out_path = RESULTS_DIR / f"trades_{args.zone.lower()}.parquet"
    trade_log.to_parquet(out_path, index=False)
    log.info("Saved trade log to %s", out_path)
