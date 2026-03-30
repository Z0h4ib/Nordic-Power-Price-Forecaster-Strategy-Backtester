"""
src/backtest/metrics.py

Performance metric computation for the Nordic Power strategy backtester.

Computes the full suite of trading metrics required by PHASE4.md:
Sharpe ratio, Sortino ratio, max drawdown, win rate, profit factor,
average winning/losing trade, trade count, and max consecutive losses.

All ratio metrics are annualised using 252 trading days.

Usage::

    python -m src.backtest.metrics --threshold 0.05
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
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR  = PROJECT_ROOT / "data" / "results"

#: Annualisation factor for Sharpe/Sortino (daily P&L → annual).
TRADING_DAYS_PER_YEAR = 252

# ---------------------------------------------------------------------------
# Metrics Engine
# ---------------------------------------------------------------------------

def compute_metrics(trade_log: pd.DataFrame, daily_pnl: pd.Series, zone: str, threshold: float) -> dict:
    """
    Computes a comprehensive set of performance metrics for the trading strategy.
    
    Parameters
    ----------
    trade_log : pd.DataFrame
        Hourly trade log containing 'signal' and 'hourly_pnl'.
    daily_pnl : pd.Series
        Aggregated daily P&L.
    zone : str
        Bidding zone (e.g., 'DK1').
    threshold : float
        The trading threshold used to generate these signals.
        
    Returns
    -------
    dict
        Dictionary containing all requested metrics.
    """
    metrics = {
        "zone": zone,
        "threshold": threshold,
    }

    # -- Overview --
    metrics["total_return"] = daily_pnl.sum()
    
    # -- Risk Adjusted Returns --
    # Sharpe Ratio: Annualized (daily * sqrt(252))
    daily_mean = daily_pnl.mean()
    daily_std = daily_pnl.std()
    
    metrics["sharpe_ratio"] = (
        (daily_mean / daily_std * np.sqrt(TRADING_DAYS_PER_YEAR))
        if daily_std != 0 and pd.notna(daily_std)
        else 0.0
    )
    
    # Sortino Ratio: Downside deviation only
    downside_returns = daily_pnl[daily_pnl < 0]
    downside_std = np.sqrt((downside_returns ** 2).mean())
    metrics["sortino_ratio"] = (
        (daily_mean / downside_std * np.sqrt(TRADING_DAYS_PER_YEAR))
        if downside_std != 0 and pd.notna(downside_std)
        else 0.0
    )
    
    # Max Drawdown (calculated on cumulative daily P&L)
    cum_pnl = daily_pnl.cumsum()
    peak = cum_pnl.cummax()
    drawdown = cum_pnl - peak
    metrics["max_drawdown"] = drawdown.min()  # Negative value, representing max loss from peak
    
    # -- Trade Level Statistics --
    # Filter to actual trade hours
    trades = trade_log[trade_log["signal"] != 0]["hourly_pnl"]
    trade_count = len(trades)
    metrics["trade_count"] = trade_count
    
    if trade_count > 0:
        win_mask = trades > 0
        loss_mask = trades < 0
        
        metrics["win_rate"] = win_mask.sum() / trade_count
        
        gross_profit = trades[win_mask].sum()
        gross_loss = abs(trades[loss_mask].sum())
        
        metrics["profit_factor"] = (
            (gross_profit / gross_loss) if gross_loss != 0 else np.nan
        )
        
        metrics["avg_win"] = trades[win_mask].mean() if win_mask.any() else 0.0
        metrics["avg_loss"] = trades[loss_mask].mean() if loss_mask.any() else 0.0
        
        # Max consecutive losses (longest losing streak of trades)
        is_loss = loss_mask
        streak = is_loss.groupby((~is_loss).cumsum()).sum()
        metrics["max_consecutive_losses"] = int(streak.max()) if not streak.empty else 0
    else:
        metrics["win_rate"] = 0.0
        metrics["profit_factor"] = 0.0
        metrics["avg_win"] = 0.0
        metrics["avg_loss"] = 0.0
        metrics["max_consecutive_losses"] = 0

    return metrics

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", default=0.05, type=float)
    args = parser.parse_args()

    results = []

    # Local import to reuse pnl module avoiding circular logic
    from src.backtest.strategy import generate_signals
    from src.backtest.pnl import calculate_pnl

    for zone in ["DK1", "DK2"]:
        log.info("Processing zone: %s", zone)
        forecasts_path = RESULTS_DIR / f"forecasts_{zone.lower()}.parquet"
        
        if not forecasts_path.exists():
            log.warning("Forecasts for %s not found. Skipping.", zone)
            continue
            
        forecasts_df = pd.read_parquet(forecasts_path)
        signals_df = generate_signals(forecasts_df, threshold=args.threshold)
        trade_log, daily_pnl = calculate_pnl(signals_df)
        
        # Compute metrics
        m = compute_metrics(trade_log, daily_pnl, zone, args.threshold)
        results.append(m)
        
        # Log summary for DK1
        if zone == "DK1":
            lines = [f"  {'='*55}",
                     f"  --- Backtest Metrics: {zone} | Threshold: {args.threshold*100:.0f}% ---",
                     f"  {'='*55}"]
            for k, v in m.items():
                lines.append(f"  {k:25}: {v:10.4f}" if isinstance(v, float) else f"  {k:25}: {v}")
            lines.append(f"  {'='*55}")
            log.info("\n".join(lines))

    # Save to CSV
    if results:
        df_results = pd.DataFrame(results)
        out_csv = RESULTS_DIR / "backtest_metrics.csv"
        df_results.to_csv(out_csv, index=False)
        log.info("Saved %d metrics rows to %s", len(df_results), out_csv)
    else:
        log.warning("No results to save.")
