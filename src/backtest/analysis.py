import logging
from pathlib import Path

import pandas as pd

from src.backtest.strategy import generate_signals
from src.backtest.pnl import calculate_pnl
from src.backtest.metrics import compute_metrics

# ---------------------------------------------------------------------------
# Logging & Paths
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "data" / "results"

# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def run_threshold_sensitivity(forecasts_df: pd.DataFrame, zone: str):
    thresholds = [0.0, 0.02, 0.05, 0.10]
    results = []

    log.info(f"Running Threshold Sensitivity for {zone}...")
    
    for t in thresholds:
        signals_df = generate_signals(forecasts_df, threshold=t)
        trade_log, daily_pnl = calculate_pnl(signals_df)
        m = compute_metrics(trade_log, daily_pnl, zone, t)
        
        results.append({
            "threshold": t,
            "total_return": m["total_return"],
            "sharpe_ratio": m["sharpe_ratio"],
            "win_rate": m["win_rate"],
            "trade_count": m["trade_count"]
        })

    # Save and print
    df_results = pd.DataFrame(results)
    out_csv = RESULTS_DIR / "threshold_sensitivity.csv"
    df_results.to_csv(out_csv, index=False)
    
    print("\n" + "="*60)
    print(f"--- Threshold Sensitivity Comparison ({zone}) ---")
    print("="*60)
    print(df_results.to_string(index=False))
    print("="*60 + "\n")
    log.info(f"Saved threshold sensitivity results to {out_csv}")


def run_regime_analysis(forecasts_df: pd.DataFrame, zone: str):
    regimes = {
        "Energy crisis": ("2022-01-01", "2022-06-30"),
        "Post-crisis": ("2022-07-01", "2023-06-30"),
        "Stabilization": ("2023-07-01", "2024-12-31")
    }
    
    results = []
    threshold = 0.05
    
    log.info(f"Running Regime Analysis for {zone} (threshold={threshold})...")
    
    # Generate full signals and P&L first, then slice
    signals_df = generate_signals(forecasts_df, threshold=threshold)
    trade_log, daily_pnl = calculate_pnl(signals_df)
    
    for regime_name, (start_dt, end_dt) in regimes.items():
        # Mask trade_log
        mask_log = (trade_log["timestamp_utc"] >= pd.to_datetime(start_dt)) & \
                   (trade_log["timestamp_utc"] <= pd.to_datetime(end_dt + " 23:59:59"))
        regime_trade_log = trade_log[mask_log].copy()
        
        # Mask daily_pnl (index is python date)
        start_date = pd.to_datetime(start_dt).date()
        end_date = pd.to_datetime(end_dt).date()
        mask_daily = (daily_pnl.index >= start_date) & (daily_pnl.index <= end_date)
        regime_daily_pnl = daily_pnl[mask_daily].copy()
        
        if len(regime_daily_pnl) == 0:
            log.warning(f"No data for regime '{regime_name}' ({start_dt} to {end_dt})")
            results.append({
                "regime": regime_name,
                "total_return": 0.0,
                "sharpe_ratio": 0.0,
                "win_rate": 0.0,
                "trade_count": 0
            })
            continue

        m = compute_metrics(regime_trade_log, regime_daily_pnl, zone, threshold)
        
        results.append({
            "regime": regime_name,
            "total_return": m["total_return"],
            "sharpe_ratio": m["sharpe_ratio"],
            "win_rate": m["win_rate"],
            "trade_count": m["trade_count"]
        })

    # Save and print
    df_results = pd.DataFrame(results)
    out_csv = RESULTS_DIR / "regime_analysis.csv"
    df_results.to_csv(out_csv, index=False)
    
    print("\n" + "="*60)
    print(f"--- Regime Analysis ({zone}) ---")
    print("="*60)
    print(df_results.to_string(index=False))
    print("="*60 + "\n")
    log.info(f"Saved regime analysis results to {out_csv}")


if __name__ == "__main__":
    zone = "DK1"
    forecasts_path = RESULTS_DIR / f"forecasts_{zone.lower()}.parquet"
    
    if not forecasts_path.exists():
        raise FileNotFoundError(f"Forecasts path does not exist: {forecasts_path}")
        
    df = pd.read_parquet(forecasts_path)
    
    # Mute intermediate script logs for cleaner output
    logging.getLogger("src.backtest.strategy").setLevel(logging.WARNING)
    logging.getLogger("src.backtest.pnl").setLevel(logging.WARNING)
    
    run_threshold_sensitivity(df, zone)
    run_regime_analysis(df, zone)
