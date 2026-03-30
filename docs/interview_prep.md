# Interview Preparation — Nordic Power Price Forecaster & Strategy Backtester

*Personal prep document. Answers are grounded in what we actually built — use specific numbers to signal depth.*

---

## Technical Questions

**1. Why did you use walk-forward validation instead of a simple train/test split?**

A random or static train/test split causes **lookahead bias** in any time-series problem — the model sees patterns from future periods during training, producing optimistically biased out-of-sample metrics. In electricity price forecasting this is particularly acute because prices in 2024 carry no causal relationship with prices in 2022; using them together without temporal ordering can leak crisis-period volatility regimes backward into the model. Our expanding-window walk-forward scheme re-trains from scratch on all available prior data and tests on the next 168-hour block, exactly mirroring production deployment. DK1 yields 5 folds and DK2 yields 33 folds — both produce genuine out-of-sample MAE estimates. The 33-fold DK2 series also gives us a learning curve: MAE declines through fold ~15 before plateauing, which tells us the model hasn't saturated its information content and would benefit from longer history.

---

**2. What is data leakage and how did you prevent it?**

Data leakage occurs when information from outside the training window contaminate the model — either through target encoding, cross-contamination of feature statistics, or incorrect temporal ordering. We guarded against it at three levels. First, all feature transformations (lags, rolling means) are computed **strictly within each zone's chronological order**, grouped by `bidding_zone` using pandas `groupby`, so no future prices bleed into lag columns. Second, the walk-forward framework never allows a test fold's timestamps to appear in its corresponding training set. Third, the forward price proxy used in the backtest is a 168-hour **trailing** mean of actual prices — it uses only information available before the trading decision point. One residual risk: hyperparameters were tuned on the full DK2 series using `best_params.json` rather than inside the fold loop. In production I'd move hyperparameter search inside the walk-forward, though for a research project of this scope the bias is small.

---

**3. Why did you choose XGBoost over ARIMA or a linear model?**

Three reasons specific to our problem. First, electricity prices exhibit **non-linear interactions** between features that a linear model cannot capture — for instance, the price-suppressing effect of wind generation is amplified at certain demand regimes (low-demand winter nights with high wind routinely send prices negative). XGBoost's tree structure learns these conditional interactions automatically. Second, the ENTSO-E API occasionally returns missing generation columns for certain zones — XGBoost handles missing values natively by learning the optimal split direction for null inputs, avoiding imputation bias. Third, training time is under 60 seconds per fold on 26k rows with 30 features, making the full 33-fold DK2 retraining tractable. ARIMA would require separate modelling of trend and seasonality at hourly granularity (an ARIMAX with 24 seasonal periods and exogenous generation variables), with higher calibration overhead and lower interpretability. LSTM was also considered but requires substantially more data than three years of hourly observations to outperform gradient boosting on tabular inputs, and it produces no equivalent of SHAP for feature attribution.

---

**4. What does SHAP tell you and what were the top features in your model?**

SHAP (SHapley Additive exPlanations) decomposes each prediction into additive contributions from individual features, satisfying local accuracy, missingness, and consistency axioms that simpler importance metrics (gain, coverage) do not. In our DK1 model, the three highest mean-absolute SHAP contributors were `price_lag_24h`, `price_lag_168h`, and `wind_total_mw`. The 24h and 168h lags together reflect the strong autoregressive and weekly-periodic structure of day-ahead prices — this is expected and reassuring. `wind_total_mw` appearing as the third predictor aligns directly with the merit order: high wind displaces gas peakers and compresses the marginal cost, pulling prices down. Calendar features (`hour_sin`, `is_weekend`) ranked lower, which suggests the weekly and daily demand patterns are largely captured by the price lags themselves. The SHAP summary plot (`data/results/shap_summary_dk1.png`) also shows no spurious features — features like `temp_cph` rank near the bottom, consistent with the intuition that temperature effects on Danish power prices are secondary to wind and price momentum.

---

**5. How did you validate the Monte Carlo simulation?**

We use Monte Carlo in two distinct ways, so validation differs. For **price path simulation** (Phase 3), we bootstrapped historical out-of-sample residuals with an AR(1) filter (ρ ≈ 0.93) to preserve serial autocorrelation — residuals in electricity forecasting are serially correlated because price reverts slowly after a shock. Validation: the empirical 90% prediction interval achieved **95.8% coverage** on the last held-out fold, slightly conservative as expected with bootstrap intervals (they tend to be wide relative to parametric intervals). For the **return bootstrap** (Phase 4), 1,000 resamples of the daily P&L series test whether the observed Sharpe is path-dependent. With p5 Sharpe = 11.4 and 100% of simulations producing Sharpe > 1.0, the strategy's edge appears structurally robust. However I'd note that bootstrap resamples don't preserve the temporal structure of the returns — they treat daily P&L as i.i.d. — which overstates robustness if there is meaningful serial correlation in the returns. A block bootstrap would be more conservative.

---

**6. What is the Sortino ratio and why did you report it alongside Sharpe?**

Sharpe penalises all return volatility symmetrically — upside volatility is treated as equivalent to downside risk. In a trading strategy where large right-tail wins are desirable, this is a misspecification. The Sortino ratio replaces the denominator with **downside deviation**: only returns below zero enter the volatility calculation, so the ratio is insensitive to the size or frequency of profitable periods. For DK1 at the default 5% threshold, we report Sharpe = 16.6 and Sortino = 21.0 — the higher Sortino confirms the strategy has a meaningful positive skew in the P&L distribution, generating losses that are smaller and less frequent than gains. For DK2, Sharpe = 14.3 vs Sortino = 19.5. In energy trading with clustered negative-price episodes, Sortino is arguably the more honest single-number performance descriptor.

---

**7. How does your forward price proxy work and what are its limitations?**

The proxy is the 168-hour trailing mean of actual spot prices — a backward-looking price level used as a stand-in for the expected future spot price. Signal: if `XGBoost_forecast > proxy × (1 + threshold)`, go long (we expect prices to be above "fair value"). This has two known limitations. First, a real forward price reflects market consensus and risk premium in addition to the expected spot — our proxy conflates expected value with recent history and will produce systematically wrong signals during sustained trend regimes. Second, using a rolling mean of *actual* prices introduces minimal but non-zero lookahead: the 168h window includes prices from the same week as the decision point. In production this would be replaced by a live M+1 or Q+1 forward quote from EPEX Spot or Nasdaq Commodities. I'd surface this limitation proactively in any live deployment conversation because the Sharpe estimate is directly sensitive to how well the proxy approximates a real forward.

---

## Domain Questions

**8. Why do electricity prices go negative?**

Electricity cannot be economically stored at grid scale, so supply and demand must be balanced each hour. When renewable generation — particularly wind, which has near-zero marginal cost — exceeds demand and inflexible baseload generation (nuclear, run-of-river hydro) cannot curtail quickly enough, generators face a binary choice: pay to export or shut down and incur restart costs. For generators with high restart costs (e.g., combined heat and power plants with heating obligations), paying negative prices to continue operating is economically rational. In DK1 we observe negative prices predominantly on winter weekends overnight — high wind, low industrial demand — which is encoded in our `is_weekend` and `wind_total_mw` features. Negative prices also create a genuine P&L opportunity in our backtest because they are somewhat predictable by the model under calm conditions, even if the magnitude is not.

---

**9. What is the merit order and why does it matter for forecasting?**

The merit order is the dispatch sequence for generators, ranked by increasing short-run marginal cost: renewables (zero fuel cost) are dispatched first, then nuclear, then coal, then gas, then peakers. The marginal price in the day-ahead market is set by the highest-cost generator required to meet demand — the so-called "price-setting technology." As renewables penetrate, they push expensive generators off the margin and depress the clearing price for the entire stack, disproportionately affecting hours when demand is low and wind is high. This is why `wind_total_mw` and `renewables_ratio` are among our highest-SHAP features: they serve as direct proxies for whether expensive generators are setting the marginal price. Any electricity price forecasting model that omits generation data is implicitly ignoring the most important structural driver of intraday price variation.

---

**10. What is the difference between DK1 and DK2?**

DK1 covers Jutland and Funen (western Denmark), electrically interconnected with Germany and the Netherlands. DK2 covers Zealand and Bornholm (eastern Denmark), connected to Sweden and Germany via sea cables. The two zones are in separate Nordic synchronous system areas and can exhibit significant price divergence when interconnector capacity is constrained — during high wind in Jutland, DK1 prices can collapse while DK2 remains anchored to Swedish nuclear prices. In practice across our 2022–2024 sample, the zones are correlated but not identical: the 50/50 DK1+DK2 combined portfolio produces a daily P&L correlation that reduces max drawdown from -266 EUR/MWh (DK1) and -768 EUR/MWh (DK2) to a combined figure, demonstrating the diversification benefit of cross-zone exposure. DK2 also has significantly more forecast folds (33 vs 5) because DK1's ENTSO-E history in our pull is shorter — a modelling artifact worth mentioning in any presentation.

---

**11. What drove the 2022 energy crisis in Scandinavian power markets?**

The immediate trigger was Russia's curtailment of gas flows to Europe following the Ukraine invasion, which sent TTF gas prices from ~25 EUR/MWh in late 2021 to over 300 EUR/MWh by August 2022. Because gas-fired generation sets the marginal price across much of the European power market, day-ahead electricity prices followed — DK1 prices peaked above 700 EUR/MWh during August 2022. Simultaneously, a severe drought reduced Norwegian and Swedish hydro reservoir levels to multi-decade lows, eliminating the primary seasonal buffer that normally caps Nordic power prices. The combined supply shock created an environment where our model's assumptions broke down: the XGBoost model trained on pre-crisis data could not anticipate price dynamics driven by gas market fundamentals and geopolitical events not represented in the feature set. Our regime analysis confirms this — the strategy generates negligible P&L during the crisis and post-crisis windows (Jan 2022 – Jun 2023), with all returns concentrated in the stabilisation period.

---

**12. How would you improve this model for production use?**

The most impactful improvement would be replacing the rolling-mean forward price proxy with **actual forward market data** — EPEX Day-Ahead or Nasdaq Q+1 quotes. This alone would eliminate the most structurally concerning limitation. Second, I'd add **gas price (TTF) and CO₂ permit price (ETS)** as exogenous features: both have high short-term correlation with power prices and are publicly available. Third, the current model uses fixed hyperparameters across all folds — a production system would retune periodically (monthly or quarterly) using a nested walk-forward to avoid the hyperparameter lookahead bias I noted earlier. Fourth, I'd implement a **regime detection layer** — a changepoint detection algorithm or a hidden Markov model on price volatility — to automatically suppress signals during market dislocations. Finally, the constant 1 MW position size ignores the signal's confidence — a Kelly sizing framework or a forecast-confidence-weighted position would likely improve the Sharpe/drawdown ratio without changing the signal logic.

---

## Project Judgment Questions

**13. What was the hardest part of building this?**

The hardest part was getting the **ENTSO-E data pull to be robust**. The entsoe-py client returns 503 errors unpredictably — sometimes at the first request, sometimes mid-page for a multi-month pull. The naive implementation failed silently: the client would return a partial DataFrame without raising an exception, so I wouldn't detect that 3 months of data was missing until the feature merger produced unexpected NaN gaps. The final solution implements per-chunk try/except with three-attempt exponential backoff, explicit date-range validation after each successful pull, and a final row-count assertion before writing to CSV. Getting this to run idempotently — so re-running the pipeline after a 503 doesn't create partial duplicates — required the `ON CONFLICT DO NOTHING` PostgreSQL pattern throughout. It wasn't intellectually exciting, but it's the kind of production-grade data engineering hygiene that separates a working research prototype from something deployable.

---

**14. What would you do differently?**

The biggest design decision I'd revisit is the **forward price proxy**. I knew at the time it was a stand-in, but I underestimated how much it inflates the apparent Sharpe. The strategy is essentially betting the XGBoost forecast against its own recent average — and since the model is reasonably good, the strategy almost always wins. A real backtest would use exchange-published forward quotes, which reflect market consensus and would produce a tighter (and more honest) signal. I'd also use **block bootstrap** instead of i.i.d. bootstrap for the return stress test — the current implementation breaks the serial structure of daily P&L, which overstates portfolio robustness. Finally, I would have set up CI/CD from day one. The absence of automated testing meant each major module change required manual verification across all dependent scripts — a growing maintenance burden as the codebase scaled to 12 source files.

---

**15. How confident are you in the backtest Sharpe? Could it be overfitted?**

The reported Sharpe of 16.6 on DK1 is almost certainly inflated by the forward price proxy issue I already acknowledged. A strategy that compares a model's own forecast to the model's own trailing average will naturally show a high hit rate, because both numbers are correlated with the same underlying price moment. That said, the bootstrap stress test provides partial reassurance: p5 Sharpe = 11.4 means that even in the worst 5% of alternate historical paths, the strategy comfortably outperforms passive holding. What I cannot rule out is **regime overfitting** — the strategy's parameters (particularly the 5% threshold) were selected after observing the full backtest period, which means they may be optimal for 2022–2024 but suboptimal out-of-sample. A proper production system would select threshold using only pre-stabilisation data and validate on the stabilisation period held out. I'm confident the directional edge is real; I'm less confident the quantitative Sharpe estimate accurately represents what a live trading system would achieve.

---

**16. What happens to the strategy if XGBoost model performance degrades?**

The strategy's P&L is directly tied to the model's directional accuracy — win rate and Sharpe both collapse if the model's forecasts degrade toward noise. We can infer the sensitivity from the threshold sensitivity table: at threshold = 0% (no filter, all hours traded), Sharpe = 17.2 with 720 trades; at threshold = 10%, Sharpe = 15.6 with 581 trades. The higher threshold filters out the model's less-confident signals, which improves win rate (73.5% vs 70.8%) but reduces total return. If model quality degrades, the first visible symptom would be a sharp rise in losing trade frequency — monitorable with a rolling 30-day win rate dashboard. The portfolio-level mitigation is that DK1 and DK2 are sourced from separate models, so a degradation in one zone doesn't automatically impair the other. In production I'd define a hard kill-switch: if the rolling 30-day Sharpe drops below 2.0 (from a 16+ baseline), suspend live trading and trigger a model review.

---

## Additional Likely Questions (Based on PowerMart JD)

**17. How would you handle real-time data ingestion and model scoring at scale if this were deployed in production?**

The current architecture is batch-oriented: the pipeline fetches yesterday's ENTSO-E data and recomputes features overnight. For live deployment, I'd restructure around a **streaming ingestion layer** — ENTSO-E provides an API for intraday generation updates, and the feature computation is lightweight enough to run as a streaming transform (lags and rolling means recompute incrementally in O(1) with a fixed-size buffer). The XGBoost model would publish hourly directional signals 24 hours ahead of delivery via a simple REST endpoint backed by the pre-trained joblib file. The database layer would shift from idempotent batch inserts to an append-only time-series store (TimescaleDB or InfluxDB) for the streaming inputs, retaining PostgreSQL for the slower-moving configuration and historical audit tables. Model scoring latency on a single joblib artifact is under 1ms per prediction, so real-time scoring is not the bottleneck — data ingestion reliability is.

---

**18. How would you quantify model uncertainty in a live trading context?**

The Phase 3 Monte Carlo provides a natural framework: the empirical bootstrap residual distribution, conditioned on the AR(1) residual autocorrelation structure, gives a calibrated prediction interval around each point forecast. In production, I'd report signal confidence as the percentile rank of the forecast within the 90% interval: signals where `forecast > p95` are high-confidence longs; signals near the interval centre are low-confidence and should either be filtered out or sized smaller. The AR(1) filter parameter (ρ ≈ 0.93) would be re-estimated monthly from the rolling residual series to track any drift in forecast error autocorrelation — a useful early indicator of model drift before MAE itself degrades significantly.

---

**19. What is your view on using machine learning for price forecasting in energy markets compared to fundamental models?**

They solve different problems. A fundamental model (stochastic gas + hydro + demand balance) is **interpretable under stress scenarios** — you can run it with Russia cutting off gas and observe the transmission to power prices. An ML model cannot extrapolate beyond its training distribution; it has no mechanism to price information about events that have no historical precedent in the feature set. Our regime analysis makes this concrete: the strategy earns nothing during the 2022 crisis precisely because the model was trained on pre-crisis price dynamics. A production system at an energy trading desk would likely combine both: a fundamental model for long-run position sizing and seasonal structure, and ML for intraday residual signal extraction and short-horizon price direction. The XGBoost model in this project is best understood as a **systematic short-term alpha tool**, not a fundamental price model.

---

**20. Walk me through the P&L calculation — could you verify a single row manually?**

Yes, and I designed the trade log specifically to make this possible. Each row in `data/results/trades_dk1.parquet` contains: `timestamp_utc`, `signal` (+1/−1/0), `forecast` (EUR/MWh from XGBoost), `forward_price` (168h trailing mean at that timestamp), `actual_price` (realised spot), and `hourly_pnl`. The calculation is: `hourly_pnl = signal × (actual_price − forward_price)`. To verify: pick any row where `signal = +1`. If `actual_price = 85.0` and `forward_price = 72.0`, then `hourly_pnl = 1 × (85 − 72) = 13 EUR/MWh` on a 1 MW position. If the model over-predicted and `actual = 65.0`, then `pnl = 1 × (65 − 72) = −7 EUR/MWh`. Daily P&L is the sum of all hourly P&L values within each calendar day. Cumulative P&L is the running sum of daily P&L, and Sharpe is computed as `(mean(daily_pnl) / std(daily_pnl)) × √252`. Every number in the metrics table can be reproduced from the trade log in four lines of pandas.
