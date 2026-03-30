# Phase 3 — Forecasting Model

## Goal

Build and evaluate three progressively complex forecasting models for DK1 and DK2 day-ahead electricity spot prices. Use walk-forward validation (expanding window) to avoid data leakage. Layer a Monte Carlo price path simulation on top of the best model for uncertainty quantification.

## Prerequisites

- Phase 1 complete: data loaded in PostgreSQL (spot_prices, generation, weather tables)
- Phase 2 complete: feature-engineered dataset saved as `data/processed/features_dk1.parquet` and `data/processed/features_dk2.parquet`
- Features available: lag features (t-1, t-24, t-168), calendar features (hour, weekday, month, is_weekend, is_holiday), weather (temperature, wind speed), generation (wind onshore/offshore, solar)

## Deliverables checklist

- [ ] `src/models/validation.py` — walk-forward validation framework
- [ ] `src/models/baseline.py` — persistence and linear regression baselines
- [ ] `src/models/forecaster.py` — XGBoost forecasting model with hyperparameter tuning
- [ ] `src/models/monte_carlo.py` — Monte Carlo price path simulation
- [ ] `notebooks/03_modeling.ipynb` — full modeling notebook with plots and analysis
- [ ] `data/results/model_metrics.csv` — evaluation metrics for all models
- [ ] `data/results/forecasts_dk1.parquet` — out-of-sample forecasts for DK1
- [ ] `data/results/forecasts_dk2.parquet` — out-of-sample forecasts for DK2

## Walk-forward validation framework

This is the most important design decision in the project. Do NOT use random train/test splits.

### How it works

```
Training window (expanding)          Test window (fixed 7 days)
|============================|-------|
|================================|-------|
|====================================|-------|
```

- **Initial training set:** 2022-01-01 to 2023-06-30 (18 months)
- **Test window:** 7 days (168 hours), rolling forward
- **Step size:** 7 days (non-overlapping test windows)
- **Final test window ends:** 2024-12-31
- After each fold, expand the training set to include the previous test window
- This produces ~39 folds covering July 2023 through December 2024

### Implementation spec for `src/models/validation.py`

```python
def walk_forward_split(df, initial_train_end, test_days=7, step_days=7):
    """
    Generator that yields (train_df, test_df) tuples.
    
    Args:
        df: DataFrame with datetime index, sorted chronologically
        initial_train_end: last date of the initial training period
        test_days: number of days in each test window
        step_days: how far to advance between folds
    
    Yields:
        (train_df, test_df) for each fold
    """
```

- Must return a `fold_id` with each split for tracking
- Log the date range of each fold for debugging

## Model 1: Persistence baseline

The simplest possible model. Forecast = same hour yesterday.

```python
def persistence_forecast(df):
    """Forecast price at hour t = actual price at hour t-24."""
```

- This is the floor. Any useful model must beat this.
- Compute MAE, RMSE, MAPE for each fold

## Model 2: Linear regression

A basic statistical model to establish whether features add value over pure persistence.

- Features to use: price_lag_24, price_lag_168, hour_sin, hour_cos, weekday_sin, weekday_cos, month_sin, month_cos, temperature_c, wind_speed_ms, wind_total_mw (onshore + offshore), solar_mw
- Use `sklearn.linear_model.Ridge` (regularized to handle collinearity)
- Fit once per walk-forward fold
- Store coefficients per fold to check for stability

## Model 3: XGBoost

The main model. This is what you will talk about in the interview.

### Features (all of these)

- price_lag_1, price_lag_2, price_lag_3
- price_lag_24, price_lag_48
- price_lag_168 (same hour last week)
- price_rolling_mean_24 (24-hour rolling average)
- price_rolling_std_24 (24-hour rolling std — captures volatility)
- hour_sin, hour_cos (cyclical encoding of hour-of-day)
- weekday_sin, weekday_cos
- month_sin, month_cos
- is_weekend (binary)
- is_holiday (binary — use Danish public holidays)
- temperature_c
- wind_speed_ms
- wind_total_mw (onshore + offshore combined)
- solar_mw

### Hyperparameter tuning

Use `sklearn.model_selection.TimeSeriesSplit` within the training set for inner CV. Do NOT use the walk-forward test set for tuning.

Search space:
```python
param_grid = {
    'n_estimators': [200, 500, 1000],
    'max_depth': [4, 6, 8],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8],
    'colsample_bytree': [0.8],
    'min_child_weight': [3, 5]
}
```

- Use `RandomizedSearchCV` with `n_iter=20` to keep runtime reasonable
- Tune once on the initial training set, then use those hyperparameters for all walk-forward folds
- Save the best hyperparameters to `data/results/best_params.json`

### SHAP values

After the final model is trained:
- Compute SHAP values using `shap.TreeExplainer`
- Generate a SHAP summary plot (beeswarm)
- Save the top 10 feature importances — you will reference these in interviews

## Evaluation metrics

For each model, compute per walk-forward fold:

| Metric | Formula | Why |
|--------|---------|-----|
| MAE | mean(\|y - ŷ\|) | Main metric — interpretable in EUR/MWh |
| RMSE | sqrt(mean((y - ŷ)²)) | Penalizes large errors (spike misses) |
| MAPE | mean(\|y - ŷ\| / \|y\|) * 100 | Relative accuracy — but breaks near zero prices |

Also compute:
- **MAE by hour-of-day** — plot as a 24-bar chart to show when the model struggles
- **MAE by month** — seasonal accuracy patterns
- **Directional accuracy** — % of hours where model correctly predicts price going up vs down

Store all metrics in `data/results/model_metrics.csv` with columns: model, fold_id, mae, rmse, mape, directional_accuracy

## Monte Carlo price path simulation

Layer this on top of the XGBoost model. Purpose: quantify forecast uncertainty.

### Method

1. From each XGBoost forecast point, compute the residual (actual - forecast)
2. Fit the residual distribution — test for normality, if not normal use empirical distribution
3. For each future hour, simulate N=1000 price paths by:
   - Taking the XGBoost point forecast
   - Adding a random draw from the fitted residual distribution
   - Optionally: model residual autocorrelation (AR(1) on residuals) for more realistic paths
4. Output: for each hour, the 5th, 25th, 50th, 75th, and 95th percentile price paths

### Implementation spec for `src/models/monte_carlo.py`

```python
def simulate_price_paths(forecasts, residuals, n_simulations=1000, seed=42):
    """
    Args:
        forecasts: array of XGBoost point forecasts
        residuals: array of historical residuals (actual - forecast)
        n_simulations: number of Monte Carlo paths
        seed: random seed for reproducibility
    
    Returns:
        DataFrame with columns: hour, p5, p25, p50, p75, p95
    """
```

## Plots to generate in `notebooks/03_modeling.ipynb`

1. **Model comparison bar chart** — MAE for persistence vs Ridge vs XGBoost, side by side
2. **Forecast vs actuals time series** — pick one representative test week, overlay XGBoost forecast on actual prices for DK1. Include Monte Carlo confidence bands (5th-95th percentile shaded)
3. **MAE by hour-of-day** — 24-bar chart for XGBoost, showing which hours are hardest to forecast
4. **Residual distribution** — histogram + QQ plot for XGBoost residuals
5. **SHAP beeswarm plot** — feature importance visualization
6. **Monte Carlo fan chart** — 1-week ahead forecast with 1000 simulated paths fading to gray, percentile bands in color
7. **Learning curve** — MAE over walk-forward folds to show if model improves with more data

## Files to produce

```
src/models/validation.py
src/models/baseline.py
src/models/forecaster.py
src/models/monte_carlo.py
notebooks/03_modeling.ipynb
data/results/model_metrics.csv
data/results/best_params.json
data/results/forecasts_dk1.parquet
data/results/forecasts_dk2.parquet
```

## Notes

- All models must be trained separately for DK1 and DK2 — they are different markets
- Never leak future data into training. Double-check that lag features use only past values at prediction time
- XGBoost handles missing values natively, but log any NaN counts per fold
- Use `logging` throughout, not `print`
- Set random seeds everywhere for reproducibility (numpy, xgboost, sklearn)
- Save trained XGBoost models with `joblib.dump()` to `data/results/xgb_dk1.joblib` and `xgb_dk2.joblib` — you may need them in Phase 4