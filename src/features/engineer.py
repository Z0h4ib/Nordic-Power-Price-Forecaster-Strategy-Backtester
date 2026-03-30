"""
src/features/engineer.py

Reusable feature engineering pipeline for the Nordic Power Price Forecaster.

Loads raw spot_prices, generation, and weather tables from PostgreSQL,
merges them, engineers 30+ features (lags, calendar, generation, weather),
adds the forecast target (price_next_24h), runs quality checks, and
optionally saves the result back to a ``features`` table in PostgreSQL.

All functions accept and return DataFrames so they can be unit-tested
or called independently from notebooks.
"""

import logging
import os
from pathlib import Path

import holidays
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import MetaData, create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.dialects.postgresql import insert as pg_insert

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH  = PROJECT_ROOT / "db" / "schema.sql"

FEATURE_COLUMNS = [
    "timestamp_utc", "bidding_zone",
    # lag
    "price_lag_1h", "price_lag_2h", "price_lag_24h", "price_lag_48h",
    "price_lag_168h", "price_rolling_mean_24h", "price_rolling_std_24h",
    "price_rolling_mean_168h",
    # calendar
    "hour", "hour_sin", "hour_cos", "day_of_week", "is_weekend",
    "month", "month_sin", "month_cos", "is_danish_holiday",
    # generation
    "wind_total_mw", "wind_lag_1h", "wind_lag_24h", "solar_mw",
    "renewables_ratio",
    # weather
    "temp_aarhus", "temp_cph", "wind_speed_aarhus", "wind_speed_cph",
    "temp_mean_dk",
    # target
    "price_next_24h",
]

CHUNK_SIZE = 5_000

#: Small epsilon added to the renewables denominator to prevent division by zero.
RENEWABLES_EPS = 1e-6


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def get_engine() -> Engine:
    """
    Build a SQLAlchemy engine from .env credentials.

    Returns
    -------
    Engine
    """
    load_dotenv(dotenv_path=PROJECT_ROOT / ".env")
    user = os.getenv("DB_USER")
    pw   = os.getenv("DB_PASSWORD", "")
    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "5432")
    name = os.getenv("DB_NAME")
    return create_engine(
        f"postgresql+psycopg2://{user}:{pw}@{host}:{port}/{name}",
        pool_pre_ping=True,
    )


# ---------------------------------------------------------------------------
# Step 1 — Load & merge
# ---------------------------------------------------------------------------

def load_raw_data(engine: Engine) -> pd.DataFrame:
    """
    Load spot_prices, generation, and weather from PostgreSQL and merge
    into a single DataFrame keyed on ``(timestamp_utc, bidding_zone)``.

    Weather (stored per location) is pivoted to wide format so each row
    has columns for both Aarhus and Copenhagen readings.

    Parameters
    ----------
    engine : sqlalchemy.engine.Engine

    Returns
    -------
    pd.DataFrame
        Columns: timestamp_utc, bidding_zone, price_eur_mwh,
        wind_onshore_mw, wind_offshore_mw, solar_mw,
        temp_aarhus, temp_cph, wind_speed_aarhus, wind_speed_cph.
    """
    prices = pd.read_sql(
        "SELECT timestamp_utc, bidding_zone, price_eur_mwh "
        "FROM spot_prices ORDER BY timestamp_utc",
        engine, parse_dates=["timestamp_utc"],
    )
    gen = pd.read_sql(
        "SELECT timestamp_utc, bidding_zone, wind_onshore_mw, wind_offshore_mw, solar_mw "
        "FROM generation ORDER BY timestamp_utc",
        engine, parse_dates=["timestamp_utc"],
    )
    weather = pd.read_sql(
        "SELECT timestamp_utc, location, temperature_c, wind_speed_ms "
        "FROM weather ORDER BY timestamp_utc",
        engine, parse_dates=["timestamp_utc"],
    )

    # Pivot weather to wide
    w_aar = weather[weather.location == "Aarhus"][["timestamp_utc", "temperature_c", "wind_speed_ms"]].rename(
        columns={"temperature_c": "temp_aarhus", "wind_speed_ms": "wind_speed_aarhus"})
    w_cph = weather[weather.location == "Copenhagen"][["timestamp_utc", "temperature_c", "wind_speed_ms"]].rename(
        columns={"temperature_c": "temp_cph", "wind_speed_ms": "wind_speed_cph"})
    weather_wide = w_aar.merge(w_cph, on="timestamp_utc", how="outer")

    # Merge
    df = prices.merge(gen, on=["timestamp_utc", "bidding_zone"], how="left")
    df = df.merge(weather_wide, on="timestamp_utc", how="left")
    df = df.sort_values(["bidding_zone", "timestamp_utc"]).reset_index(drop=True)

    log.info("Loaded and merged raw data: %d rows.", len(df))
    return df


# ---------------------------------------------------------------------------
# Step 2 — Lag features
# ---------------------------------------------------------------------------

def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add price lag and rolling-window features, computed per bidding zone.

    Features added (8):
        price_lag_1h, price_lag_2h, price_lag_24h, price_lag_48h,
        price_lag_168h, price_rolling_mean_24h, price_rolling_std_24h,
        price_rolling_mean_168h.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``price_eur_mwh`` and ``bidding_zone``.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with lag columns appended.
    """
    lags = [
        ("price_lag_1h",   1),
        ("price_lag_2h",   2),
        ("price_lag_24h",  24),
        ("price_lag_48h",  48),
        ("price_lag_168h", 168),
    ]
    for col, n in lags:
        df[col] = df.groupby("bidding_zone")["price_eur_mwh"].shift(n)

    for zone in df.bidding_zone.unique():
        mask = df.bidding_zone == zone
        s = df.loc[mask, "price_eur_mwh"]
        df.loc[mask, "price_rolling_mean_24h"]  = s.rolling(24,  min_periods=1).mean()
        df.loc[mask, "price_rolling_std_24h"]   = s.rolling(24,  min_periods=1).std()
        df.loc[mask, "price_rolling_mean_168h"] = s.rolling(168, min_periods=1).mean()

    log.info("Lag features added (8 columns).")
    return df


# ---------------------------------------------------------------------------
# Step 3 — Calendar features
# ---------------------------------------------------------------------------

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add calendar and cyclical time-encoding features.

    Features added (9):
        hour, hour_sin, hour_cos, day_of_week, is_weekend,
        month, month_sin, month_cos, is_danish_holiday.

    Cyclical encoding uses sin/cos so that adjacent time points
    (e.g. hour 23 and hour 0) remain close in feature space.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``timestamp_utc``.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with calendar columns appended.
    """
    ts = df["timestamp_utc"]

    df["hour"]        = ts.dt.hour
    df["hour_sin"]    = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"]    = np.cos(2 * np.pi * df["hour"] / 24)
    df["day_of_week"] = ts.dt.dayofweek
    df["is_weekend"]  = df["day_of_week"].isin([5, 6])
    df["month"]       = ts.dt.month
    df["month_sin"]   = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]   = np.cos(2 * np.pi * df["month"] / 12)

    years = range(int(ts.dt.year.min()), int(ts.dt.year.max()) + 1)
    dk_hol = holidays.Denmark(years=years)
    df["is_danish_holiday"] = ts.dt.date.map(lambda d: d in dk_hol)

    log.info("Calendar features added (9 columns).")
    return df


# ---------------------------------------------------------------------------
# Step 4 — Generation features
# ---------------------------------------------------------------------------

def add_generation_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add wind/solar generation features.

    Features added (5):
        wind_total_mw, wind_lag_1h, wind_lag_24h,
        solar_mw (already present from merge — validated here),
        renewables_ratio.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``wind_onshore_mw``, ``wind_offshore_mw``,
        ``solar_mw``, ``bidding_zone``.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with generation feature columns appended.
    """
    df["wind_total_mw"] = df["wind_onshore_mw"].fillna(0) + df["wind_offshore_mw"].fillna(0)
    df["wind_lag_1h"]   = df.groupby("bidding_zone")["wind_total_mw"].shift(1)
    df["wind_lag_24h"]  = df.groupby("bidding_zone")["wind_total_mw"].shift(24)

    total_ren = df["wind_total_mw"] + df["solar_mw"].fillna(0)
    df["renewables_ratio"] = total_ren / (total_ren + RENEWABLES_EPS)

    log.info("Generation features added (5 columns).")
    return df


# ---------------------------------------------------------------------------
# Step 5 — Weather features
# ---------------------------------------------------------------------------

def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add weather-derived features.

    Features added (1 new):
        temp_mean_dk — average of Aarhus and Copenhagen temperatures.

    The four raw weather columns (temp_aarhus, temp_cph, wind_speed_aarhus,
    wind_speed_cph) are already present from the merge step.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``temp_aarhus`` and ``temp_cph``.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with ``temp_mean_dk`` appended.
    """
    df["temp_mean_dk"] = (df["temp_aarhus"].fillna(0) + df["temp_cph"].fillna(0)) / 2
    log.info("Weather features added (1 new column: temp_mean_dk).")
    return df


# ---------------------------------------------------------------------------
# Step 6 — Target
# ---------------------------------------------------------------------------

def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add the forecast target: price 24 hours ahead.

    The last 24 rows per zone will have a null target (no future data
    to predict). These are removed in :func:`run_quality_checks`.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``price_eur_mwh`` and ``bidding_zone``.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with ``price_next_24h`` appended.
    """
    df["price_next_24h"] = df.groupby("bidding_zone")["price_eur_mwh"].shift(-24)
    log.info("Target added (price_next_24h).")
    return df


# ---------------------------------------------------------------------------
# Step 7 — Quality checks
# ---------------------------------------------------------------------------

def run_quality_checks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate the feature DataFrame and drop un-trainable rows.

    Checks performed:
    - Warn if any row has ALL price-lag features null.
    - Warn about rolling-feature outliers (range sanity).
    - **Drop** rows where ``price_next_24h`` is null (can't train on them).

    Parameters
    ----------
    df : pd.DataFrame
        Full feature DataFrame including target.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with null-target rows removed.
    """
    lag_cols = [c for c in df.columns if "lag" in c and "price" in c]
    all_lag_null = df[lag_cols].isna().all(axis=1).sum()
    if all_lag_null:
        log.warning("Rows with ALL price-lag features null: %d (expected for first 168h per zone).", all_lag_null)

    for col in ["price_rolling_mean_24h", "price_rolling_std_24h", "price_rolling_mean_168h"]:
        if col in df.columns:
            col_range = df[col].max() - df[col].min()
            log.info("Rolling check | %s | range: %.1f", col, col_range)

    rows_before = len(df)
    df = df.dropna(subset=["price_next_24h"]).reset_index(drop=True)
    dropped = rows_before - len(df)
    log.info("Dropped %d rows with null target. Remaining: %d.", dropped, len(df))

    return df


# ---------------------------------------------------------------------------
# Step 8 — Master pipeline
# ---------------------------------------------------------------------------

def build_features(engine: Engine) -> pd.DataFrame:
    """
    End-to-end feature engineering pipeline.

    Calls load → lag → calendar → generation → weather → target →
    quality checks, returning a model-ready DataFrame.

    Parameters
    ----------
    engine : sqlalchemy.engine.Engine

    Returns
    -------
    pd.DataFrame
        Model-ready dataset with all features and target.
        Columns are ordered as defined in ``FEATURE_COLUMNS``.
    """
    df = load_raw_data(engine)
    df = add_lag_features(df)
    df = add_calendar_features(df)
    df = add_generation_features(df)
    df = add_weather_features(df)
    df = add_target(df)
    df = run_quality_checks(df)

    # Select and order final columns
    available = [c for c in FEATURE_COLUMNS if c in df.columns]
    df = df[available]

    log.info(
        "Feature build complete | %d rows | %d columns | %s → %s",
        len(df), len(df.columns),
        df["timestamp_utc"].min().date(), df["timestamp_utc"].max().date(),
    )
    return df


# ---------------------------------------------------------------------------
# Save to PostgreSQL
# ---------------------------------------------------------------------------

def save_features(df: pd.DataFrame, engine: Engine) -> int:
    """
    Save the feature DataFrame to the ``features`` table in PostgreSQL.

    Creates the table if it does not exist (applies schema.sql with
    IF NOT EXISTS). Uses INSERT ... ON CONFLICT DO NOTHING for
    idempotent reruns.

    Parameters
    ----------
    df : pd.DataFrame
        Output of :func:`build_features`.
    engine : sqlalchemy.engine.Engine

    Returns
    -------
    int
        Number of rows inserted (excluding duplicates).
    """
    # Apply schema (creates features table if missing)
    raw_sql = SCHEMA_PATH.read_text()
    with engine.begin() as conn:
        for stmt in raw_sql.split(";"):
            # Strip leading comment lines so we can inspect the actual SQL
            lines = [l for l in stmt.split("\n") if not l.strip().startswith("--")]
            stmt = "\n".join(lines).strip()
            if not stmt:
                continue
            if stmt.upper().startswith("DROP"):
                continue
            stmt = stmt.replace("CREATE TABLE ", "CREATE TABLE IF NOT EXISTS ", 1)
            stmt = stmt.replace("CREATE INDEX ", "CREATE INDEX IF NOT EXISTS ", 1)
            conn.execute(text(stmt))

    # Reflect the features table
    metadata = MetaData()
    metadata.reflect(bind=engine, only=["features"])
    table = metadata.tables["features"]

    records = df.to_dict(orient="records")
    inserted = 0

    with engine.begin() as conn:
        for i in range(0, len(records), CHUNK_SIZE):
            chunk = records[i : i + CHUNK_SIZE]
            stmt = pg_insert(table).values(chunk).on_conflict_do_nothing()
            result = conn.execute(stmt)
            inserted += result.rowcount

    log.info("Inserted %d / %d rows into features table.", inserted, len(df))
    return inserted


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Run the full feature pipeline and save to PostgreSQL.
    """
    engine = get_engine()
    df = build_features(engine)
    save_features(df, engine)

    # Confirmation query
    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT bidding_zone, COUNT(*), MIN(timestamp_utc), MAX(timestamp_utc) "
            "FROM features GROUP BY bidding_zone ORDER BY bidding_zone"
        )).fetchall()
    for zone, count, mn, mx in rows:
        log.info("features | %s | %d rows | %s → %s", zone, count, mn, mx)


if __name__ == "__main__":
    main()
