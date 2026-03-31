# Phase 5 — Polish, README & Interview Prep

## Goal

Make the project publicly presentable and interview-ready. A recruiter or senior quant at PowerMart should be able to open the GitHub repo, understand what was built and why in under 2 minutes, run the project from scratch using the setup guide, and be impressed by the depth and clarity of the documentation.

This phase has no new modeling. It is entirely about presentation, code quality, and making sure everything holds together end to end.

## Prerequisites

- Phases 1–4 complete
- All four notebooks exist and run cleanly
- All src/ modules have docstrings and use logging
- data/results/ has all output files
- db/schema.sql is up to date with all five tables

---

## Deliverables checklist

- [ ] `README.md` — comprehensive, visually compelling project readme
- [ ] `docs/methodology.md` — deeper technical writeup
- [ ] `docs/interview_prep.md` — Q&A document for interview preparation
- [ ] `Makefile` — one-command setup and pipeline execution
- [ ] `docker-compose.yml` — PostgreSQL setup so anyone can run the project
- [ ] All notebooks cleaned: outputs cleared, re-run top to bottom, no errors
- [ ] All src/ modules reviewed: consistent logging, docstrings, type hints
- [ ] `.github/` — optional but good: a simple CI check that imports all modules
- [ ] `CLAUDE.md` updated to reflect completed project status

---

## README.md structure

This is the most important deliverable of Phase 5. It must be good enough to show in an interview.

### Sections

**1. Header**
- Project title: Nordic Power Price Forecaster & Strategy Backtester
- One-line description
- Badges: Python version, license
- A single hero chart image — the forecast vs actuals chart with Monte Carlo bands from notebook 03

**2. Overview**
- What the project does in 3–4 sentences
- Why DK1/DK2 specifically
- What the final numbers are: XGBoost MAE vs persistence, backtest Sharpe ratio

**3. Architecture diagram**
- Simple ASCII or mermaid diagram showing the data flow:
  ENTSO-E API → fetch_entso.py → PostgreSQL → engineer.py → forecaster.py → strategy.py → results

**4. Results summary**
A clean table showing key outcomes:

| Metric | DK1 | DK2 | Portfolio |
|--------|-----|-----|-----------|
| Forecast MAE vs persistence (% improvement) | | | |
| Backtest Sharpe ratio | | | |
| Max drawdown | | | |
| Win rate | | | |
| Monte Carlo p5 Sharpe | | | |

**5. Project structure**
- Full repo tree with one-line description of each file

**6. Quickstart**
- Prerequisites (Python 3.11+, PostgreSQL, ENTSO-E API key)
- Setup steps: clone → create .env → docker-compose up → make install → make pipeline
- Expected runtime per phase

**7. Methodology**
- Brief explanation of walk-forward validation with the diagram from PHASE_3.md
- Why XGBoost over ARIMA
- Monte Carlo on prices vs Monte Carlo on returns — explain both
- Key domain insight: merit order and why wind generation is the strongest predictor

**8. Limitations**
- Forward price is proxied, not real forward curve data
- No transaction costs or slippage modeled
- Position sizing is constant (1 MW) — no Kelly criterion or dynamic sizing
- Model is retrained on fixed hyperparameters — in production you'd retune periodically

**9. Tech stack**
- Clean table of libraries and what each is used for

**10. Author**
- Name, LinkedIn, GitHub

---

## docs/methodology.md

A deeper technical document (500–800 words) covering:

1. **Data pipeline** — how ENTSO-E and Open-Meteo data is fetched, cleaned, and stored. Note the UTC normalization and idempotent loading.

2. **Feature engineering** — why cyclical encoding matters for hour and month. Why lag-168 (same hour last week) is important for electricity markets. What renewables ratio captures.

3. **Walk-forward validation** — explain expanding window vs sliding window, why random splits cause data leakage in time series, what the fold structure looks like numerically.

4. **Model selection** — why persistence and Ridge baselines first. Why XGBoost for the main model (handles non-linearity, interactions between hour and wind, missing values natively). Why not LSTM (insufficient data, harder to interpret, slower to train).

5. **Backtesting design** — signal generation logic, forward price proxy and its limitations, why threshold matters, how P&L is calculated at hourly granularity.

6. **Monte Carlo** — two separate uses: price path uncertainty (Phase 3) and bootstrap stress test on returns (Phase 4). Explain the difference clearly.

7. **Regime dependence** — how 2022 energy crisis affected model performance and strategy returns, why this matters for live deployment.

---

## docs/interview_prep.md

A Q&A document with 20 questions a senior quant at an energy trading firm might ask, with strong answers. Covers:

**Technical questions**
- Why walk-forward and not train/test split?
- What is data leakage and how did you prevent it?
- Why XGBoost over a linear model?
- What does SHAP tell you and what were the top features?
- How did you validate the Monte Carlo simulation?
- What is the Sortino ratio and why use it alongside Sharpe?
- How does your forward price proxy work and what are its limitations?

**Domain questions**
- Why do electricity prices go negative?
- What is the merit order and why does it matter for forecasting?
- What is the difference between DK1 and DK2?
- What drove the 2022 energy crisis in Scandinavian power markets?
- How would you improve this model for production use?

**Project judgment questions**
- What was the hardest part of building this?
- What would you do differently?
- How confident are you in the backtest Sharpe? Could it be overfitted?
- What happens to the strategy if XGBoost model performance degrades?

---

## Makefile

```makefile
.PHONY: install pipeline phase1 phase2 phase3 phase4 clean

install:
	pip install -r requirements.txt

db:
	docker-compose up -d
	sleep 3
	psql -h localhost -U $${DB_USER} -d $${DB_NAME} -f db/schema.sql

phase1:
	python src/pipeline/fetch_entso.py
	python src/pipeline/fetch_weather.py
	python src/pipeline/load_db.py

phase2:
	python src/features/engineer.py

phase3:
	python src/models/baseline.py
	python src/models/forecaster.py
	python src/models/monte_carlo.py

phase4:
	python src/backtest/strategy.py
	python src/backtest/pnl.py
	python src/backtest/metrics.py
	python src/backtest/monte_carlo.py

pipeline: phase1 phase2 phase3 phase4

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
```

---

## docker-compose.yml

```yaml
version: '3.8'
services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: ${DB_NAME}
      POSTGRES_USER: ${DB_USER}
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    ports:
      - "${DB_PORT}:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./db/schema.sql:/docker-entrypoint-initdb.d/schema.sql

volumes:
  postgres_data:
```

---

## Code quality pass

Go through every file in src/ and verify:

| Check | Standard |
|-------|---------|
| Docstrings | Every function and module has one |
| Type hints | All function signatures have type hints |
| Logging | No bare `print()` statements in src/ |
| Error handling | API calls have try/except with logged errors |
| Constants | No magic numbers — define as named constants at top of file |
| Imports | Sorted: stdlib → third-party → local |

---

## Notebook cleanup

For each of the four notebooks:

1. Restart kernel and run all cells top to bottom — zero errors allowed
2. Clear all outputs before committing (use `jupyter nbconvert --clear-output`)
3. Add a one-paragraph introduction cell at the top explaining what the notebook covers and what inputs it expects
4. Make sure all chart titles include units (EUR/MWh, MW, etc.)
5. Check that every chart has axis labels

---

## Final repo state

After Phase 5 the repo should look like this:

```
nordic-power-forecaster/
├── CLAUDE.md
├── README.md
├── Makefile
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── .gitignore
├── docs/
│   ├── methodology.md
│   └── interview_prep.md
├── db/
│   └── schema.sql
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_features.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_backtest.ipynb
├── src/
│   ├── pipeline/
│   │   ├── fetch_entso.py
│   │   ├── fetch_weather.py
│   │   └── load_db.py
│   ├── features/
│   │   └── engineer.py
│   ├── models/
│   │   ├── validation.py
│   │   ├── baseline.py
│   │   ├── forecaster.py
│   │   └── monte_carlo.py
│   └── backtest/
│       ├── strategy.py
│       ├── pnl.py
│       ├── metrics.py
│       └── monte_carlo.py
├── data/
│   └── results/
│       ├── model_metrics.csv
│       ├── best_params.json
│       ├── backtest_metrics.csv
│       ├── threshold_sensitivity.csv
│       └── regime_analysis.csv
└── tests/
    └── test_pipeline.py
```

---

## Notes

- `.joblib` model files and `.parquet` data files go in `.gitignore` if they are large — link to a release or mention they are generated by running the pipeline
- The hero chart in the README must be an actual saved PNG from notebook 03 — commit it to `docs/images/`
- README tone: professional but direct. No marketing language. A quant should feel like they're reading documentation, not a sales pitch.
- Update CLAUDE.md status section to say: "Project complete — all 5 phases done"
