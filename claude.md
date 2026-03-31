# Nordic Power Price Forecaster & Strategy Backtester

## What this project is

A quantitative research project that forecasts day-ahead electricity spot prices for the Danish DK1 and DK2 bidding zones, then backtests a simple trading strategy against historical prices. Built to demonstrate applied quant skills for an energy trading role at PowerMart.

## Why it was built

To bridge the gap between software engineering experience and quantitative energy trading. The project shows:
- Real market data pipeline skills (ENTSO-E, Open-Meteo)
- Time series forecasting with proper walk-forward validation
- Backtesting with P&L tracking, Sharpe ratio, and max drawdown
- Monte Carlo simulation layered on top for risk quantification

## Tech stack

- **Language:** Python 3.11+
- **Database:** PostgreSQL
- **Key libraries:** `entsoe-py`, `pandas`, `sqlalchemy`, `psycopg2`, `scikit-learn`, `xgboost`, `matplotlib`, `plotly`, `statsmodels`, `jupyter`
- **Environment:** `.env` file for secrets (API keys, DB credentials) — never committed

## Repo structure

```
nordic-power-forecaster/
├── CLAUDE.md                  # This file — project context for Claude
├── README.md                  # Public-facing project description
├── .env.example               # Template for environment variables
├── .gitignore
├── requirements.txt
├── data/
│   └── raw/                   # Raw CSVs dumped from API (gitignored)
├── db/
│   └── schema.sql             # PostgreSQL schema definitions
├── notebooks/
│   ├── 01_eda.ipynb           # Exploratory data analysis
│   ├── 02_features.ipynb      # Feature engineering
│   ├── 03_modeling.ipynb      # Model training and evaluation
│   └── 04_backtest.ipynb      # Backtesting and Monte Carlo
├── src/
│   ├── pipeline/
│   │   ├── fetch_entso.py     # ENTSO-E data ingestion
│   │   ├── fetch_weather.py   # Open-Meteo ingestion
│   │   └── load_db.py         # Load raw data into PostgreSQL
│   ├── features/
│   │   └── engineer.py        # Feature engineering functions
│   ├── models/
│   │   ├── baseline.py        # Naive/persistence baselines
│   │   └── forecaster.py      # XGBoost forecasting model
│   └── backtest/
│       ├── strategy.py        # Trading signal logic
│       ├── pnl.py             # P&L tracking
│       └── monte_carlo.py     # Monte Carlo simulation
└── tests/
    └── test_pipeline.py
```

## Data sources

| Source | What we pull | API |
|--------|-------------|-----|
| ENTSO-E Transparency Platform | DK1/DK2 day-ahead prices, wind/solar generation | `entsoe-py` wrapper |
| Open-Meteo | Temperature, wind speed at Danish grid points | REST, no key needed |

## Key domain context

- **DK1** = West Denmark (connected to continental Europe, Jutland)
- **DK2** = East Denmark (connected to Nordic pool, Zealand/Copenhagen)
- **Day-ahead price** = electricity price agreed for delivery the next day, set by auction
- **Negative prices** occur when renewable generation exceeds demand — common in DK1 on windy weekends
- **Merit order** = cheapest generators dispatched first; wind has near-zero marginal cost so high wind = lower prices

## Modeling approach

1. **Baseline:** persistence model (forecast = last observed price)
2. **Statistical:** ARIMA on price series
3. **ML:** XGBoost with lag features, calendar features, weather, and generation data
4. **Validation:** walk-forward (expanding window) — NOT random train/test split, which would cause data leakage in time series

## Backtesting logic

- Signal: if forecast price < current day-ahead forward → buy (go long)
- Track hourly P&L over 2022–2024 window
- Metrics: total return, Sharpe ratio, max drawdown, win rate
- Monte Carlo: simulate 1000 return paths by bootstrapping daily returns to stress-test strategy robustness

## Environment variables needed

```
ENTSO_E_API_KEY=your_key_here
DB_HOST=localhost
DB_PORT=5432
DB_NAME=power_forecaster
DB_USER=your_user
DB_PASSWORD=your_password
```

## Current status

Project complete. All 5 phases built and tested. Ready for CV and interview use.

## Conventions

- All functions must have docstrings
- Use `logging` not `print` in src/ modules
- Notebooks are for exploration and presentation only — reusable logic goes in src/
- Commit after each phase is complete with a descriptive message