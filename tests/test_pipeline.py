"""
tests/test_pipeline.py

Smoke tests for the Nordic Power Forecaster pipeline.

These tests verify that all modules import cleanly, core functions exist,
and the P&L / metrics logic produces correct outputs for known inputs.
They do NOT require a live database or API connection.

Run with::

    python -m pytest tests/test_pipeline.py -v
"""

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# 1. Import smoke tests
# ---------------------------------------------------------------------------


def test_import_fetch_entso():
    import src.pipeline.fetch_entso  # noqa: F401


def test_import_fetch_weather():
    import src.pipeline.fetch_weather  # noqa: F401


def test_import_load_db():
    import src.pipeline.load_db  # noqa: F401


def test_import_engineer():
    import src.features.engineer  # noqa: F401


def test_import_baseline():
    import src.models.baseline  # noqa: F401


def test_import_forecaster():
    import src.models.forecaster  # noqa: F401


def test_import_shap_analysis():
    pytest.importorskip("shap", reason="shap not installed")
    import src.models.shap_analysis  # noqa: F401


def test_import_models_monte_carlo():
    import src.models.monte_carlo  # noqa: F401


def test_import_strategy():
    import src.backtest.strategy  # noqa: F401


def test_import_pnl():
    import src.backtest.pnl  # noqa: F401


def test_import_metrics():
    import src.backtest.metrics  # noqa: F401


def test_import_backtest_monte_carlo():
    import src.backtest.monte_carlo  # noqa: F401


# ---------------------------------------------------------------------------
# 2. Strategy signal logic
# ---------------------------------------------------------------------------


def _make_signals_df(
    last_forecast: float,
    last_actual: float,
    history_price: float = 100.0,
    threshold: float = 0.05,
    n_history: int = 25,
) -> pd.DataFrame:
    """Build a DataFrame with enough history to seed the 24-hour rolling window.

    ``generate_signals`` computes forward_price internally as
    ``actual.shift(1).rolling(24, min_periods=1).mean()``.  With fewer than
    2 rows, shift(1) gives NaN and all signals are forced to flat.  This helper
    prepends ``n_history`` rows at ``history_price`` before the test row so
    the window is fully warmed up and forward_price == history_price exactly.
    """
    from src.backtest.strategy import generate_signals

    n_total = n_history + 1
    dates = pd.date_range("2023-01-01", periods=n_total, freq="h")
    actuals = [history_price] * n_history + [last_actual]
    forecasts = [history_price] * n_history + [last_forecast]
    df = pd.DataFrame({"timestamp_utc": dates, "forecast": forecasts, "actual": actuals})
    return generate_signals(df, threshold=threshold)


def test_signal_long():
    """Forecast 10% above forward (history=100) → signal = +1 (long)."""
    # forward_price ≈ 100; forecast=110 > 100*1.05=105 → long
    signals = _make_signals_df(last_forecast=110.0, last_actual=105.0)
    assert signals["signal"].iloc[-1] == 1


def test_signal_short():
    """Forecast 10% below forward (history=100) → signal = -1 (short)."""
    # forward_price ≈ 100; forecast=90 < 100*0.95=95 → short
    signals = _make_signals_df(last_forecast=90.0, last_actual=95.0)
    assert signals["signal"].iloc[-1] == -1


def test_signal_flat():
    """Forecast within 5% band → signal = 0 (flat)."""
    # forward_price ≈ 100; forecast=101 within [95, 105] → flat
    signals = _make_signals_df(last_forecast=101.0, last_actual=100.0)
    assert signals["signal"].iloc[-1] == 0


# ---------------------------------------------------------------------------
# 3. P&L calculation
# ---------------------------------------------------------------------------


def test_pnl_long_profit():
    """Long trade on last row: actual=115 vs forward≈100 → hourly_pnl ≈ +15."""
    from src.backtest.pnl import calculate_pnl

    # history_price=100 → forward_price of last row = 100.0 exactly (all history rows)
    signals = _make_signals_df(last_forecast=110.0, last_actual=115.0, history_price=100.0)
    trade_log, daily_pnl = calculate_pnl(signals)
    # last row is the trade row with signal=+1; forward_price=100, actual=115 → pnl=15
    last_row = trade_log[trade_log["signal"] != 0].iloc[-1]
    assert last_row["hourly_pnl"] == pytest.approx(15.0)


def test_pnl_flat_zero():
    """Flat position: P&L is zero regardless of actual price move."""
    from src.backtest.pnl import calculate_pnl

    signals = _make_signals_df(last_forecast=101.0, last_actual=200.0)
    trade_log, daily_pnl = calculate_pnl(signals)
    # signal=0 on last row → hourly_pnl = 0 * (200 - forward) = 0
    assert trade_log["hourly_pnl"].iloc[-1] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 4. Metrics — Sharpe ratio edge cases
# ---------------------------------------------------------------------------


def test_metrics_sharpe_zero_std():
    """If daily P&L has zero variance Sharpe should be 0.0, not NaN or error."""
    from src.backtest.metrics import compute_metrics

    n = 50
    dates = pd.date_range("2023-01-01", periods=n * 24, freq="h")
    trade_log = pd.DataFrame(
        {
            "timestamp_utc": dates,
            "signal": [1] * (n * 24),
            "hourly_pnl": [1.0] * (n * 24),
        }
    )
    daily_pnl = pd.Series([24.0] * n)
    m = compute_metrics(trade_log, daily_pnl, zone="DK1", threshold=0.05)
    assert m["sharpe_ratio"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 5. Bootstrap Monte Carlo shape
# ---------------------------------------------------------------------------


def test_bootstrap_shape():
    """bootstrap_strategy should return a DataFrame with the right shape."""
    from src.backtest.monte_carlo import bootstrap_strategy

    rng = np.random.default_rng(0)
    daily_pnl = pd.Series(rng.normal(10, 5, 300))
    results = bootstrap_strategy(daily_pnl, n_simulations=100, seed=0)
    assert results.shape == (100, 4)
    assert list(results.columns) == ["sim_id", "total_return", "sharpe_ratio", "max_drawdown"]
