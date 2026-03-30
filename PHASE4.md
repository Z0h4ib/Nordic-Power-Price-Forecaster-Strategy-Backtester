# Phase 4 — Strategy Backtester

## Goal

Build a trading strategy on top of the XGBoost forecasts from Phase 3. Simulate historical P&L, compute risk-adjusted performance metrics, and stress-test the strategy using Monte Carlo simulation on returns. This is the phase that makes the project feel like a real trading desk deliverable.

## Prerequisites

- Phase 3 complete
- `data/results/forecasts_dk1.parquet` and `forecasts_dk2.parquet` exist with columns: timestamp_utc, actual, forecast, fold_id
- `data/results/monte_carlo_dk1.parquet` exists with percentile bands
- `data/results/model_metrics.csv` exists

---

## Trading strategy logic

The strategy is based on a simple signal: if the XGBoost forecast predicts prices will be higher than the current forward price, go long (buy). If it predicts lower, go short (sell).

### Signal generation

```
forward_price = rolling 24-hour average of actual prices (proxy for forward curve)
signal = +1  if forecast > forward_price * (1 + threshold)   → long
signal = -1  if forecast < forward_price * (1 - threshold)   → short
signal =  0  if abs(forecast - forward_price) < threshold    → flat (no trade)
```

- Default threshold: 5% — only trade when forecast diverges meaningfully
- Position size: 1 MW (normalized — P&L is in EUR/MWh equivalent)
- Test threshold values: 0%, 2%, 5%, 10% — compare performance

### P&L calculation

For each hour with a non-zero signal:
```
hourly_pnl = signal * (actual_price - forward_price)
```

- Accumulate hourly P&L into a daily and cumulative P&L series
- Track trades separately: entry timestamp, exit timestamp, signal direction, realized P&L

---

## Performance metrics

These are the metrics that matter to a trading team. Compute all of them.

| Metric | Formula | Target |
|--------|---------|--------|
| Total return | sum(hourly_pnl) | As high as possible |
| Sharpe ratio | mean(daily_pnl) / std(daily_pnl) * sqrt(252) | > 1.0 is good, > 1.5 is strong |
| Sortino ratio | mean(daily_pnl) / std(negative daily_pnl) * sqrt(252) | Better than Sharpe for skewed returns |
| Max drawdown | largest peak-to-trough decline in cumulative P&L | Lower is better |
| Win rate | % of trades with positive P&L | > 50% is baseline |
| Profit factor | gross profit / gross loss | > 1.5 is good |
| Avg winning trade | mean P&L of profitable trades | — |
| Avg losing trade | mean P&L of unprofitable trades | — |
| Trade count | total number of trades taken | — |
| Max consecutive losses | longest losing streak | — |

---

## Rolling performance analysis

Beyond static metrics, compute rolling versions to show how strategy performance evolves:

- **Rolling 30-day Sharpe** — plot over time to show if performance is stable or decaying
- **Rolling 30-day win rate** — does the model's edge persist?
- **Monthly P&L heatmap** — rows = years, columns = months, cell = monthly P&L
- **Drawdown chart** — plot the drawdown series over time, highlight max drawdown period

---

## Regime analysis

Split the backtest period into market regimes and analyse performance separately:

| Regime | Definition | Period (approx) |
|--------|-----------|-----------------|
| Energy crisis | Extreme price volatility, frequent spikes | 2022 H1 |
| Post-crisis normalization | Prices falling from peak, still elevated | 2022 H2 – 2023 H1 |
| Stabilization | Prices normalized, lower volatility | 2023 H2 – 2024 |

For each regime compute: total return, Sharpe, win rate, trade count. This shows interviewers you understand that model performance varies with market conditions.

---

## Monte Carlo stress test on strategy returns

This is different from Phase 3's Monte Carlo (which was on price paths). This one stress-tests the strategy itself.

### Method

1. Take the historical daily P&L series from the backtest
2. Bootstrap resample daily returns with replacement — 1000 simulations, each with same number of days as the original backtest
3. For each simulation compute: total return, Sharpe ratio, max drawdown
4. Plot the distribution of these metrics across 1000 simulations
5. Report: 5th percentile Sharpe (worst-case), median Sharpe, 95th percentile Sharpe (best-case)

### Why this matters

Point-in-time backtest results can be lucky. The Monte Carlo stress test shows whether the edge is robust or fragile. If the 5th percentile Sharpe is still > 0.5, the strategy has a genuine edge. If it's negative, the strategy may be overfitting to the specific historical path.

```python
def bootstrap_strategy(daily_pnl, n_simulations=1000, seed=42):
    """
    Bootstrap resample strategy daily P&L to stress-test robustness.
    
    Args:
        daily_pnl: array of daily P&L values
        n_simulations: number of bootstrap samples
        seed: random seed
    
    Returns:
        DataFrame with columns: sim_id, total_return, sharpe, max_drawdown
    """
```

---

## Threshold sensitivity analysis

Run the full backtest for threshold values: 0%, 2%, 5%, 10%

For each threshold:
- Total return
- Sharpe ratio
- Trade count
- Win rate

Plot as a 2x2 grid or a table. This shows the strategy is not overfitted to a single magic threshold and demonstrates the tradeoff between selectivity and trade frequency.

---

## DK1 vs DK2 comparison

Run the strategy independently on DK1 and DK2. Then simulate a combined portfolio (equal weight, 50/50).

Compare:
- Individual zone performance
- Portfolio performance (does combining zones reduce drawdown via diversification?)
- Correlation of daily P&L between zones — low correlation = genuine diversification benefit

---

## Database additions

### Table: `trades`
```sql
id              SERIAL PRIMARY KEY
timestamp_utc   TIMESTAMP NOT NULL
bidding_zone    VARCHAR(10) NOT NULL
signal          INT          -- +1, -1, or 0
forecast        FLOAT
forward_price   FLOAT
actual_price    FLOAT
hourly_pnl      FLOAT
threshold       FLOAT        -- which threshold setting generated this
```

### Table: `backtest_metrics`
```sql
id              SERIAL PRIMARY KEY
run_timestamp   TIMESTAMP DEFAULT NOW()
zone            VARCHAR(10)
threshold       FLOAT
total_return    FLOAT
sharpe          FLOAT
sortino         FLOAT
max_drawdown    FLOAT
win_rate        FLOAT
profit_factor   FLOAT
trade_count     INT
```

---

## Files to produce

```
src/backtest/strategy.py        — signal generation and forward price calculation
src/backtest/pnl.py             — P&L tracking and trade log
src/backtest/metrics.py         — all performance metrics functions
src/backtest/monte_carlo.py     — bootstrap stress test on returns
notebooks/04_backtest.ipynb     — full backtesting notebook
data/results/trades_dk1.parquet
data/results/trades_dk2.parquet
data/results/backtest_metrics.csv
data/results/bootstrap_results.parquet
db/schema.sql                   — updated with trades and backtest_metrics tables
```

---

## Notes

- The forward price proxy (rolling 24h mean of actuals) is a simplification — in a real system you'd pull actual forward curve data. Acknowledge this limitation in the notebook.
- Never use future actual prices to generate the signal — the signal must be based only on the forecast and the forward proxy available at decision time
- Set random seed=42 everywhere for reproducibility
- Use `logging` not `print` in all src/ modules
- The backtest period should match the walk-forward out-of-sample period from Phase 3: July 2023 to December 2024
- Annualization factor for Sharpe: use 365 * 24 for hourly data OR aggregate to daily first and use 252
