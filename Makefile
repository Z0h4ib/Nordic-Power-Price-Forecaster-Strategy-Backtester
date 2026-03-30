PYTHON ?= python3

.PHONY: help install db phase1 phase2 phase3 phase4 pipeline clean

help:
	@echo "Nordic Power Price Forecaster & Strategy Backtester"
	@echo ""
	@echo "Targets:"
	@echo "  install   Install Python dependencies from requirements.txt"
	@echo "  db        Start PostgreSQL via docker-compose and apply schema"
	@echo "  phase1    Fetch ENTSO-E + weather data and load into DB"
	@echo "  phase2    Run feature engineering pipeline"
	@echo "  phase3    Train baselines, XGBoost, SHAP analysis, Monte Carlo"
	@echo "  phase4    Run strategy backtest, metrics, and bootstrap stress test"
	@echo "  pipeline  Run phase1 → phase2 → phase3 → phase4 end-to-end"
	@echo "  clean     Remove compiled Python files and __pycache__ directories"

install:
	$(PYTHON) -m pip install -r requirements.txt

db:
	docker-compose up -d
	sleep 3
	psql -h localhost -U $${DB_USER} -d $${DB_NAME} -f db/schema.sql

phase1:
	$(PYTHON) -m src.pipeline.fetch_entso
	$(PYTHON) -m src.pipeline.fetch_weather
	$(PYTHON) -m src.pipeline.load_db

phase2:
	$(PYTHON) -m src.features.engineer

phase3:
	$(PYTHON) -m src.models.baseline --zone DK1
	$(PYTHON) -m src.models.baseline --zone DK2
	$(PYTHON) -m src.models.forecaster --zone both
	$(PYTHON) -m src.models.shap_analysis --zone both
	$(PYTHON) -m src.models.monte_carlo --zone both

phase4:
	$(PYTHON) -m src.backtest.strategy --zone DK1
	$(PYTHON) -m src.backtest.strategy --zone DK2
	$(PYTHON) -m src.backtest.pnl --zone DK1
	$(PYTHON) -m src.backtest.pnl --zone DK2
	$(PYTHON) -m src.backtest.metrics
	$(PYTHON) -m src.backtest.monte_carlo
	$(PYTHON) -m src.backtest.analysis
	$(PYTHON) -m src.backtest.portfolio

pipeline: phase1 phase2 phase3 phase4

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
