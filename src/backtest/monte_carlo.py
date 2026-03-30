"""
src/backtest/monte_carlo.py

Bootstrap Monte Carlo stress test for the Nordic Power trading strategy.

Takes the historical daily P&L series from the backtest and resamples it
with replacement (1 000 simulations, same length as the original) to build
a distribution of Sharpe ratios and max drawdowns.  This tests whether the
strategy's edge is robust or overfitted to a specific historical path.

If the 5th-percentile Sharpe is still positive the strategy has a genuine
edge.  If it turns negative the results may be path-dependent.

Usage::

    python -m src.backtest.monte_carlo --threshold 0.05
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Logging & Paths
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR  = PROJECT_ROOT / "data" / "results"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Annualisation factor used for Sharpe ratio (daily P&L → annual).
TRADING_DAYS_PER_YEAR = 252

#: Minimum recommended history length; warn if dataset is shorter.
MIN_RECOMMENDED_DAYS = 252

# ---------------------------------------------------------------------------
# Monte Carlo Bootstrap
# ---------------------------------------------------------------------------

def bootstrap_strategy(daily_pnl: pd.Series, n_simulations: int = 1000, seed: int = 42) -> pd.DataFrame:
    """
    Bootstrap resamples strategy daily P&L to stress-test robustness.
    
    This simulation involves drawing random samples with replacement from the historical 
    daily P&L distribution to generate alternative performance pathways. 
    It evaluates whether the strategy's edge is robust or over-fitted to a specific historical sequence.
    
    Parameters
    ----------
    daily_pnl : pd.Series
        Array or Series of historical daily P&L.
    n_simulations : int
        Number of Monte Carlo simulation paths to generate.
    seed : int
        Random seed for reproducibility.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: sim_id, total_return, sharpe_ratio, max_drawdown
    """
    log.info("Starting Monte Carlo Bootstrap Stress Test (%d simulations, seed=%d)…", n_simulations, seed)

    pnl_array = daily_pnl.values
    n_days = len(pnl_array)

    if n_days < MIN_RECOMMENDED_DAYS:
        log.warning(
            "Data contains only %d days. Bootstrap statistics are more reliable with >%d days of history.",
            n_days, MIN_RECOMMENDED_DAYS,
        )
        
    np.random.seed(seed)
    
    # Vectorized bootstrapping for massive speedup
    # Draw indices: shape (n_simulations, n_days)
    indices = np.random.randint(0, n_days, size=(n_simulations, n_days))
    
    # Resampled paths
    paths = pnl_array[indices]  # Shape: (n_simulations, n_days)
    
    # Calculate metrics via vectorization across axis=1
    total_returns = paths.sum(axis=1)

    means = paths.mean(axis=1)
    # ddof=1 matches pandas .std() used in metrics.py for consistency
    stds = paths.std(axis=1, ddof=1)
    stds[stds == 0] = np.nan  # Prevent division by zero
    sharpe_ratios = (means / stds) * np.sqrt(TRADING_DAYS_PER_YEAR)
    
    # Max Drawdown
    cum_pnls = paths.cumsum(axis=1)
    peaks = np.maximum.accumulate(cum_pnls, axis=1)
    drawdowns = cum_pnls - peaks
    max_drawdowns = drawdowns.min(axis=1)
    
    # Construct results DataFrame
    results_df = pd.DataFrame({
        "sim_id": np.arange(1, n_simulations + 1),
        "total_return": total_returns,
        "sharpe_ratio": sharpe_ratios,
        "max_drawdown": max_drawdowns
    })
    
    log.info("Monte Carlo Bootstrap complete — %d simulations over %d days.", n_simulations, n_days)
    return results_df


if __name__ == "__main__":
    from src.backtest.strategy import generate_signals
    from src.backtest.pnl import calculate_pnl

    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", default=0.05, type=float)
    args = parser.parse_args()

    forecasts_path = RESULTS_DIR / "forecasts_dk1.parquet"
    if not forecasts_path.exists():
        raise FileNotFoundError(f"Missing {forecasts_path}")
        
    df = pd.read_parquet(forecasts_path)
    
    logging.getLogger("src.backtest.strategy").setLevel(logging.WARNING)
    
    # Generate backtest Daily P&L
    signals_df = generate_signals(df, threshold=args.threshold)
    trade_log, daily_pnl = calculate_pnl(signals_df)
    
    # Run simulation
    n_sims = 1000
    bootstrap_results = bootstrap_strategy(daily_pnl, n_simulations=n_sims, seed=42)
    
    # Save results
    out_path = RESULTS_DIR / "bootstrap_results.parquet"
    bootstrap_results.to_parquet(out_path, index=False)
    log.info("Saved %d simulations to %s", n_sims, out_path)
    
    # Summary Calculations
    sharpes = bootstrap_results["sharpe_ratio"].dropna()
    
    p05 = np.percentile(sharpes, 5)
    p50 = np.percentile(sharpes, 50)
    p95 = np.percentile(sharpes, 95)
    
    pct_gt_0 = (sharpes > 0).mean() * 100
    pct_gt_1 = (sharpes > 1.0).mean() * 100
    
    log.info(
        "\n  %s\n  --- Monte Carlo Bootstrap Stress Test Summary (DK1) ---\n  %s\n"
        "  5th  pct Sharpe (worst-case) : %8.4f\n"
        "  50th pct Sharpe (median)     : %8.4f\n"
        "  95th pct Sharpe (best-case)  : %8.4f\n"
        "  %s\n"
        "  %% of sims with Sharpe > 0   : %8.2f%%\n"
        "  %% of sims with Sharpe > 1.0 : %8.2f%%\n  %s",
        "="*60, "="*60, p05, p50, p95, "-"*60, pct_gt_0, pct_gt_1, "="*60,
    )

    # Insight block evaluating the edge
    if p05 > 0:
        log.info("Insight: The strategy has a robust positive expected edge (5th percentile > 0).")
    else:
        log.info("Insight: The strategy exhibits fragility; worst-case paths yield a negative Sharpe.")
