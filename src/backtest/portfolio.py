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

def run_portfolio_analysis(threshold: float = 0.05):
    log.info("Starting combined portfolio analysis...")
    
    # Load and process DK1
    dk1_path = RESULTS_DIR / "forecasts_dk1.parquet"
    df_dk1 = pd.read_parquet(dk1_path)
    signals_dk1 = generate_signals(df_dk1, threshold=threshold)
    trade_log_dk1, daily_pnl_dk1 = calculate_pnl(signals_dk1)
    
    # Load and process DK2
    dk2_path = RESULTS_DIR / "forecasts_dk2.parquet"
    df_dk2 = pd.read_parquet(dk2_path)
    signals_dk2 = generate_signals(df_dk2, threshold=threshold)
    trade_log_dk2, daily_pnl_dk2 = calculate_pnl(signals_dk2)
    
    # Compute base metrics
    m_dk1 = compute_metrics(trade_log_dk1, daily_pnl_dk1, "DK1", threshold)
    m_dk2 = compute_metrics(trade_log_dk2, daily_pnl_dk2, "DK2", threshold)
    
    # Combine Daily P&L (50/50 Equal Weight)
    # Reindex to common dates to avoid missing alignments
    combined_index = daily_pnl_dk1.index.union(daily_pnl_dk2.index)
    d1 = daily_pnl_dk1.reindex(combined_index).fillna(0)
    d2 = daily_pnl_dk2.reindex(combined_index).fillna(0)
    
    portfolio_daily_pnl = (d1 + d2) / 2.0
    
    # Save portfolio Daily P&L
    portfolio_daily_pnl.to_frame("portfolio_daily_pnl").to_parquet(
        RESULTS_DIR / "portfolio_pnl.parquet"
    )
    log.info("Saved portfolio daily P&L to data/results/portfolio_pnl.parquet")
    
    # Compute Correlation
    merged_pnl = pd.DataFrame({'DK1': daily_pnl_dk1, 'DK2': daily_pnl_dk2}).dropna()
    correlation = merged_pnl['DK1'].corr(merged_pnl['DK2'])
    
    # Construct combined trade log to feed into compute_metrics
    # We outer join on timestamp_utc. If either model trades, we count it as a portfolio trade.
    t1 = trade_log_dk1[['timestamp_utc', 'signal', 'hourly_pnl']].copy()
    t2 = trade_log_dk2[['timestamp_utc', 'signal', 'hourly_pnl']].copy()
    
    comb_trade_log = pd.merge(t1, t2, on="timestamp_utc", how="outer", suffixes=("_dk1", "_dk2")).fillna(0)
    comb_trade_log["hourly_pnl"] = (comb_trade_log["hourly_pnl_dk1"] + comb_trade_log["hourly_pnl_dk2"]) / 2.0
    
    # Virtual Signal creation: non-zero if either traded
    traded_mask = (comb_trade_log["signal_dk1"] != 0) | (comb_trade_log["signal_dk2"] != 0)
    comb_trade_log["signal"] = 0
    comb_trade_log.loc[traded_mask, "signal"] = 1  # 1 indicates trade active
    
    # Compute Portfolio metrics
    m_port = compute_metrics(comb_trade_log, portfolio_daily_pnl, "Portfolio (50/50)", threshold)
    
    # Print Comparison Table
    print("\n" + "=" * 80)
    print("--- Portfolio Diversification Analysis ---")
    print("=" * 80)
    
    keys = ["total_return", "sharpe_ratio", "max_drawdown", "win_rate", "trade_count", "max_consecutive_losses"]
    
    header = f"{'Metric':<25} | {'DK1':<12} | {'DK2':<12} | {'Portfolio':<12}"
    print(header)
    print("-" * 80)
    for k in keys:
        v1 = f"{m_dk1[k]:.4f}" if isinstance(m_dk1[k], float) else str(m_dk1[k])
        v2 = f"{m_dk2[k]:.4f}" if isinstance(m_dk2[k], float) else str(m_dk2[k])
        vp = f"{m_port[k]:.4f}" if isinstance(m_port[k], float) else str(m_port[k])
        print(f"{k:<25} | {v1:<12} | {v2:<12} | {vp:<12}")
    
    print("=" * 80)
    print(f"Pearson Correlation (DK1 vs DK2 daily P&L): {correlation:.4f}")
    
    # Diversification insight
    best_sharpe = max(m_dk1['sharpe_ratio'], m_dk2['sharpe_ratio'])
    best_dd = max(m_dk1['max_drawdown'], m_dk2['max_drawdown'])  # Max drawdown is negative, closer to 0 is better
    
    improved_sharpe = m_port['sharpe_ratio'] > best_sharpe
    reduced_dd = m_port['max_drawdown'] > best_dd

    insight = []
    if improved_sharpe:
        insight.append("Sharpe Ratio improved compared to the best individual zone.")
    else:
        insight.append("Sharpe Ratio did not strictly improve over the best individual zone (but may be more stable).")
        
    if reduced_dd:
        insight.append("Max Drawdown was meaningfully reduced (diversification smoothed the losses).")
    else:
        insight.append("Max Drawdown was NOT reduced beyond the best individual zone level.")
        
    if correlation < 0.5:
        insight.append("Low correlation confirms a strong diversification benefit between the zones.")
    else:
        insight.append("High correlation suggests the zones move similarly; diversification benefit is limited.")

    print("\nDiversification Insight:")
    for statement in insight:
        print(f"- {statement}")
    print("\n")


if __name__ == "__main__":
    import logging
    # Mute base logs for a cleaner final output
    logging.getLogger("src.backtest.strategy").setLevel(logging.WARNING)
    logging.getLogger("src.backtest.pnl").setLevel(logging.WARNING)
    
    run_portfolio_analysis(threshold=0.05)
