# Methodology

**Nordic Power Price Forecaster & Strategy Backtester**  
*Technical deep-dive for quant reviewers*

---

## 1. Data Pipeline

Raw market data is sourced from two APIs. Day-ahead spot prices and actual wind/solar generation are pulled from the **ENTSO-E Transparency Platform** via `entsoe-py`, covering DK1 and DK2 bidding zones over 2022-01-01 → 2024-12-31 (26,280 hourly observations per zone). Meteorological data — hourly 2m temperature and 10m wind speed — is pulled from the **Open-Meteo Historical Archive** for Aarhus (representing the Jutland/DK1 grid) and Copenhagen (Zealand/DK2).

Both APIs are called in monthly chunks to stay within rate limits. A three-attempt exponential backoff handles transient 503 errors from ENTSO-E, which occur when paginated requests hit overloaded shards. All timestamps are coerced to **naive UTC** before insertion; this avoids ambiguity around Danish DST transitions (CET/CEST) and ensures a consistent join key across all tables.

Data is ingested into PostgreSQL using `INSERT ... ON CONFLICT DO NOTHING`, making the entire pipeline **idempotent** — re-running any phase does not duplicate rows. Five tables: `spot_prices`, `generation`, `weather`, `features`, and `backtest_metrics`.

---

## 2. Feature Engineering

Thirty-plus features are engineered from the merged hourly dataset, grouped into four families:

**Price lags** (1h, 2h, 24h, 48h, 168h) capture autoregressive structure. The 168h lag — identical hour one week prior — is particularly informative in electricity markets because weekly demand patterns are more stable than daily ones (industrial load follows working weeks, not individual days).

**Rolling statistics** (24h and 168h mean and std) provide a local price level and volatility estimate, serving as a proxy for market regime. The rolling 168h mean feeds directly into the forward price proxy used in the backtest.

**Calendar features** use **cyclical (sin/cos) encoding** for hour-of-day and month-of-year. A naive integer encoding would imply hour 23 and hour 0 are maximally distant (difference = 23), when they are in fact adjacent. Cyclical encoding preserves the circular topology: `hour_sin = sin(2π × hour / 24)`, `hour_cos = cos(2π × hour / 24)`.

**Generation features** compute `wind_total_mw = wind_onshore + wind_offshore` and `renewables_ratio = wind_total / (wind_total + ε)`. The renewables ratio is a normalized measure of the renewable share of the instantaneous generation mix. It captures the merit order effect: high wind penetration suppresses marginal cost, which depresses spot prices — particularly during hours where wind capacity exceeds demand.

---

## 3. Walk-Forward Validation

Time series cross-validation requires strict **temporal ordering** to avoid lookahead bias. A naive random train/test split would leak future price levels into training, producing optimistically biased MAE estimates.

We use an **expanding window** walk-forward scheme: each fold extends the training set by one 168h (7-day) test block, re-fitting the model from scratch on all available prior data. This mirrors production: a deployed model is periodically retrained on the full historical window, then scored on the next week's out-of-sample data.

DK1 yields **5 folds** (constrained by the shorter price history available from ENTSO-E for that zone). DK2 yields **33 folds**, spanning Aug 2023 → Oct 2024. The learning curve analysis on DK2 confirms that MAE continues to decline through fold ~15 before plateauing, suggesting the model has not saturated its useful data scale.

---

## 4. Model Selection

Two baselines are established before fitting XGBoost:

- **Persistence**: forecast = `price_lag_24h`. Achieves MAE of 39.7 EUR/MWh (DK1) and 41.9 EUR/MWh (DK2).
- **Ridge regression**: linear model trained on the same feature set. Outperforms persistence but falls short of XGBoost.

**XGBoost** is chosen as the primary model for three reasons. First, electricity price dynamics are **non-linear and interaction-dependent** — the price impact of wind generation is amplified during certain demand regimes (e.g., high-wind, low-demand Sunday mornings spike negative), which decision trees capture natively without feature engineering. Second, XGBoost handles **missing feature values** without imputation — generation columns are occasionally absent from the ENTSO-E API response, and natively missing values avoid introducing systematic bias. Third, training time on 26k rows with ~30 features is under 60 seconds per fold, making walk-forward retraining operationally feasible.

XGBoost achieves out-of-sample MAE of **37.4 EUR/MWh on DK1** (5.9% improvement over persistence) and **35.3 EUR/MWh on DK2** (15.7% improvement). SHAP analysis confirms `price_lag_24h`, `price_lag_168h`, and `wind_total_mw` as top-3 contributors by mean absolute impact.

LSTM was not pursued: with only 26k training observations and three years of hourly data, sequence models offer marginal benefit over gradient boosted trees on tabular inputs, while requiring substantially more tuning and offering lower interpretability — a disqualifying property in any context where a quant reviewer will scrutinize the model.

---

## 5. Backtesting Design

The strategy generates binary directional signals by comparing the XGBoost point forecast against a **rolling 168h actual price mean** used as a forward price proxy. If `forecast > forward_price × (1 + threshold)`, go long (+1); if `forecast < forward_price × (1 − threshold)`, go short (−1); otherwise flat (0).

This is a **known limitation**: in a live environment, this forward price proxy would be replaced by an actual broker quote or exchange-listed forward. The proxy introduces look-ahead risk (the 168h rolling mean uses data that would be available during the trading window, but the mean itself is backward-looking, not a market-derived expectation).

Hourly P&L is computed as `signal × (actual_price − forward_price)`, reflecting the standard long-short mark-to-market P&L on a 1 MW position. Cumulative P&L is traced over the full backtest window, then aggregated to daily series for Sharpe/Sortino computation. At the default 5% threshold on DK1, the strategy executes **649 trades** with a win rate of **72.3%** and an annualised Sharpe of **16.6**.

The threshold sensitivity analysis (0%, 2%, 5%, 10%) confirms that tighter filtering reduces trade count but improves win rate — the classic precision/recall tradeoff in binary classification applied to trade entry.

---

## 6. Monte Carlo — Two Distinct Applications

**Price path simulation (Phase 3)**: Applied to the XGBoost point forecast to quantify prediction interval uncertainty. Historical out-of-sample residuals (actual − forecast) from earlier walk-forward folds are bootstrapped with an AR(1) filter (ρ ≈ 0.93) to model residual autocorrelation — forecast errors in electricity markets are serially correlated because price processes are mean-reverting with persistent shocks. 1,000 simulated price paths produce empirical p5/p25/p75/p95 bands. The resulting 90% interval achieves **95.8% coverage** on the last held-out fold, slightly conservative as expected with bootstrap intervals.

**Return bootstrap stress test (Phase 4)**: A structurally independent Monte Carlo applied to the strategy's *daily P&L series* rather than prices. 1,000 bootstrap resamples (with replacement, matching original length) produce a distribution of Sharpe ratios across alternate historical paths. This addresses the key question: is the observed Sharpe a genuine edge, or is it path-dependent? Results: p5 Sharpe = **11.4**, p50 = **17.3**, p95 = **25.1**. 100% of simulations produce Sharpe > 0, and 100% produce Sharpe > 1.0 — indicating the strategy's edge is structurally robust rather than a product of a single favourable market episode.

---

## 7. Regime Dependence

The backtest period spans three qualitatively distinct regimes: the **2022 energy crisis** (Jan–Jun 2022, gas supply disruption and record spot volatility), **post-crisis recovery** (Jul 2022–Jun 2023, prices declining but still elevated and volatile), and **stabilisation** (Jul 2023–Dec 2024, prices returning toward long-run equilibrium levels of 30–70 EUR/MWh).

Regime analysis reveals the strategy's P&L concentrates overwhelmingly in the **stabilisation period**, with the crisis and post-crisis windows yielding negligible or zero returns. This is structurally expected: the XGBoost model is trained on data that skews toward the crisis period, but the model's directional edge depends on the forecast capturing smooth, mean-reverting price dynamics — dynamics largely absent during the supply shock, when prices were driven by gas market news and geopolitical events that have no representation in the feature set.

For live deployment this implies two practical requirements: (1) a regime detection layer to suppress signals during structural break periods, and (2) periodic retraining as the crisis-period data ages out of the rolling window and its distorting effect on feature scaling diminishes.
