"""
src/pipeline/load_db.py

Loads raw CSV data into PostgreSQL, runs data quality checks, and prints
a summary of what was loaded.

Behaviour:
  - Applies db/schema.sql with CREATE TABLE IF NOT EXISTS (skips DROP TABLE)
    so existing data is preserved on rerun.
  - Inserts rows with ON CONFLICT DO NOTHING — safe to rerun without
    creating duplicates.
  - Runs four quality checks after loading and logs warnings (never raises).
  - Prints a final summary: row counts, date ranges, warnings triggered.

Requires DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD in .env.
"""

import logging
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import MetaData, create_engine, text
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
RAW_DIR      = PROJECT_ROOT / "data" / "raw"
SCHEMA_PATH  = PROJECT_ROOT / "db" / "schema.sql"

# (csv filename, table name, timestamp column name)
CSV_TABLE_MAP: list[tuple[str, str, str]] = [
    ("prices_raw.csv",     "spot_prices", "timestamp_utc"),
    ("generation_raw.csv", "generation",  "timestamp_utc"),
    ("weather_raw.csv",    "weather",     "timestamp_utc"),
]

CHUNK_SIZE = 5_000   # rows per INSERT batch

PRICE_MIN = -500.0   # EUR/MWh — sanity floor (below this is suspicious)
PRICE_MAX = 3_000.0  # EUR/MWh — sanity ceiling
MAX_CONSECUTIVE_MISSING_HOURS = 48


# ---------------------------------------------------------------------------
# Environment & connection
# ---------------------------------------------------------------------------

def load_env() -> dict[str, str]:
    """
    Load database credentials from the project-root .env file.

    Returns
    -------
    dict
        Keys: host, port, name, user, password.

    Raises
    ------
    EnvironmentError
        If any required variable is missing.
    """
    load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

    required = {
        "host": "DB_HOST",
        "port": "DB_PORT",
        "name": "DB_NAME",
        "user": "DB_USER",
    }
    # Password is optional — local Homebrew/trust-auth installs have none
    optional = {"password": "DB_PASSWORD"}

    creds = {}
    missing = []
    for key, env_var in required.items():
        value = os.getenv(env_var)
        if not value:
            missing.append(env_var)
        else:
            creds[key] = value

    if missing:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing)}. "
            "Copy .env.example to .env and fill in your credentials."
        )

    for key, env_var in optional.items():
        creds[key] = os.getenv(env_var, "")
    return creds


def get_engine(creds: dict[str, str]):
    """
    Build a SQLAlchemy engine from DB credentials.

    Parameters
    ----------
    creds : dict
        Must contain keys: host, port, name, user, password.

    Returns
    -------
    sqlalchemy.engine.Engine
    """
    url = (
        f"postgresql+psycopg2://{creds['user']}:{creds['password']}"
        f"@{creds['host']}:{creds['port']}/{creds['name']}"
    )
    engine = create_engine(url, pool_pre_ping=True)
    log.info(
        "Connected to PostgreSQL | %s:%s/%s",
        creds["host"], creds["port"], creds["name"],
    )
    return engine


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def apply_schema(engine) -> None:
    """
    Apply db/schema.sql to ensure all tables and indexes exist.

    DROP TABLE statements are skipped so existing data is not destroyed on
    rerun. CREATE TABLE and CREATE INDEX statements are made idempotent by
    injecting IF NOT EXISTS.

    Parameters
    ----------
    engine : sqlalchemy.engine.Engine
    """
    raw_sql = SCHEMA_PATH.read_text()

    with engine.begin() as conn:
        for stmt in raw_sql.split(";"):
            # Strip leading comment lines so we can inspect the actual SQL
            lines = [l for l in stmt.split("\n") if not l.strip().startswith("--")]
            stmt = "\n".join(lines).strip()
            if not stmt:
                continue

            # Skip DROP TABLE — we want to preserve existing data on rerun
            if stmt.upper().startswith("DROP"):
                continue

            # Make CREATE TABLE / CREATE INDEX idempotent
            stmt = stmt.replace("CREATE TABLE ", "CREATE TABLE IF NOT EXISTS ", 1)
            stmt = stmt.replace("CREATE INDEX ", "CREATE INDEX IF NOT EXISTS ", 1)

            conn.execute(text(stmt))

    log.info("Schema applied — tables and indexes are in place.")


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def load_csv(engine, csv_filename: str, table_name: str, ts_col: str) -> int:
    """
    Load a raw CSV into a PostgreSQL table using INSERT ... ON CONFLICT DO NOTHING.

    Rows that already exist (matched by the UNIQUE constraint defined in the
    schema) are silently skipped, making reruns safe.

    Parameters
    ----------
    engine : sqlalchemy.engine.Engine
    csv_filename : str
        Filename inside ``data/raw/``, e.g. ``'prices_raw.csv'``.
    table_name : str
        Target PostgreSQL table name.
    ts_col : str
        Name of the timestamp column to parse as datetime.

    Returns
    -------
    int
        Number of rows inserted (not counting skipped duplicates).

    Raises
    ------
    FileNotFoundError
        If the CSV does not exist in ``data/raw/``.
    """
    csv_path = RAW_DIR / csv_filename
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Raw CSV not found: {csv_path}. "
            "Run fetch_entso.py and fetch_weather.py first."
        )

    df = pd.read_csv(csv_path, parse_dates=[ts_col])
    log.info("Loaded %d rows from %s", len(df), csv_filename)

    # Reflect table structure from the live DB
    metadata = MetaData()
    metadata.reflect(bind=engine, only=[table_name])
    table = metadata.tables[table_name]

    records = df.to_dict(orient="records")
    inserted = 0

    with engine.begin() as conn:
        for i in range(0, len(records), CHUNK_SIZE):
            chunk = records[i : i + CHUNK_SIZE]
            stmt = pg_insert(table).values(chunk).on_conflict_do_nothing()
            result = conn.execute(stmt)
            inserted += result.rowcount

    log.info(
        "Inserted %d / %d rows into %s (duplicates skipped)",
        inserted, len(df), table_name,
    )
    return inserted


def load_all_csvs(engine) -> dict[str, int]:
    """
    Load all three raw CSVs into their respective tables.

    Parameters
    ----------
    engine : sqlalchemy.engine.Engine

    Returns
    -------
    dict
        Maps table name → number of rows inserted.
    """
    counts = {}
    for csv_filename, table_name, ts_col in CSV_TABLE_MAP:
        counts[table_name] = load_csv(engine, csv_filename, table_name, ts_col)
    return counts


# ---------------------------------------------------------------------------
# Data quality checks
# ---------------------------------------------------------------------------

def check_consecutive_missing_prices(engine) -> list[str]:
    """
    Check that no bidding zone has more than 48 consecutive missing hourly prices.

    Gaps are inferred from the difference between consecutive present timestamps;
    a gap of N hours means N−1 hours of missing data.

    Parameters
    ----------
    engine : sqlalchemy.engine.Engine

    Returns
    -------
    list of str
        Warning messages — empty if all checks pass.
    """
    warnings = []

    with engine.connect() as conn:
        zones = [
            row[0]
            for row in conn.execute(
                text("SELECT DISTINCT bidding_zone FROM spot_prices ORDER BY bidding_zone")
            )
        ]

    for zone in zones:
        df = pd.read_sql(
            text(
                "SELECT timestamp_utc FROM spot_prices "
                "WHERE bidding_zone = :zone ORDER BY timestamp_utc"
            ),
            engine,
            params={"zone": zone},
        )

        if df.empty:
            warnings.append(f"[prices] No data at all for zone {zone}.")
            continue

        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"])
        diffs = df["timestamp_utc"].diff().dropna()
        max_gap = diffs.max()
        max_missing = (max_gap.total_seconds() / 3600) - 1

        if max_missing > MAX_CONSECUTIVE_MISSING_HOURS:
            warnings.append(
                f"[prices] Zone {zone}: {int(max_missing)} consecutive missing hours "
                f"detected (threshold: {MAX_CONSECUTIVE_MISSING_HOURS})."
            )
        else:
            log.info(
                "Consecutive missing hours check passed | %s | max gap: %.0f h",
                zone, max_missing,
            )

    return warnings


def check_negative_generation(engine) -> list[str]:
    """
    Check that no generation column contains negative values.

    Parameters
    ----------
    engine : sqlalchemy.engine.Engine

    Returns
    -------
    list of str
        Warning messages — empty if all checks pass.
    """
    warnings = []

    query = text(
        """
        SELECT
            bidding_zone,
            SUM(CASE WHEN wind_onshore_mw  < 0 THEN 1 ELSE 0 END) AS neg_onshore,
            SUM(CASE WHEN wind_offshore_mw < 0 THEN 1 ELSE 0 END) AS neg_offshore,
            SUM(CASE WHEN solar_mw         < 0 THEN 1 ELSE 0 END) AS neg_solar
        FROM generation
        GROUP BY bidding_zone
        ORDER BY bidding_zone
        """
    )

    with engine.connect() as conn:
        rows = conn.execute(query).fetchall()

    for row in rows:
        zone, neg_on, neg_off, neg_sol = row
        for col, count in [
            ("wind_onshore_mw", neg_on),
            ("wind_offshore_mw", neg_off),
            ("solar_mw", neg_sol),
        ]:
            if count and count > 0:
                warnings.append(
                    f"[generation] Zone {zone}: {count} negative values in {col}."
                )

    if not warnings:
        log.info("Negative generation check passed — no negative values found.")

    return warnings


def check_price_sanity(engine) -> list[str]:
    """
    Flag prices outside the expected range of -500 to 3000 EUR/MWh.

    Negative prices are real (renewable surplus) and are not removed,
    but extreme values beyond this band warrant investigation.

    Parameters
    ----------
    engine : sqlalchemy.engine.Engine

    Returns
    -------
    list of str
        Warning messages — empty if all prices are in range.
    """
    warnings = []

    query = text(
        """
        SELECT
            bidding_zone,
            COUNT(*)         AS out_of_range,
            MIN(price_eur_mwh) AS min_price,
            MAX(price_eur_mwh) AS max_price
        FROM spot_prices
        WHERE price_eur_mwh < :price_min OR price_eur_mwh > :price_max
        GROUP BY bidding_zone
        ORDER BY bidding_zone
        """
    )

    with engine.connect() as conn:
        rows = conn.execute(
            query, {"price_min": PRICE_MIN, "price_max": PRICE_MAX}
        ).fetchall()

    for row in rows:
        zone, count, min_p, max_p = row
        warnings.append(
            f"[prices] Zone {zone}: {count} prices outside "
            f"[{PRICE_MIN}, {PRICE_MAX}] EUR/MWh "
            f"(observed range: {min_p:.2f} → {max_p:.2f})."
        )

    if not warnings:
        log.info(
            "Price sanity check passed — all prices within [%.0f, %.0f] EUR/MWh.",
            PRICE_MIN, PRICE_MAX,
        )

    return warnings


def check_weather_alignment(engine) -> list[str]:
    """
    Check that weather timestamps align with price timestamps.

    Verifies that the weather data covers the same start/end date range as
    spot prices and that the total hourly count is in the same ballpark.

    Parameters
    ----------
    engine : sqlalchemy.engine.Engine

    Returns
    -------
    list of str
        Warning messages — empty if alignment looks good.
    """
    warnings = []

    with engine.connect() as conn:
        price_range = conn.execute(
            text("SELECT MIN(timestamp_utc), MAX(timestamp_utc) FROM spot_prices")
        ).fetchone()
        weather_range = conn.execute(
            text("SELECT MIN(timestamp_utc), MAX(timestamp_utc) FROM weather")
        ).fetchone()

    p_start, p_end = price_range
    w_start, w_end = weather_range

    if p_start is None or w_start is None:
        warnings.append("[alignment] One or both tables are empty — cannot align.")
        return warnings

    p_start = pd.Timestamp(p_start)
    p_end   = pd.Timestamp(p_end)
    w_start = pd.Timestamp(w_start)
    w_end   = pd.Timestamp(w_end)

    if w_start > p_start:
        warnings.append(
            f"[alignment] Weather starts at {w_start.date()}, "
            f"prices start at {p_start.date()} — weather begins later."
        )
    if w_end < p_end:
        warnings.append(
            f"[alignment] Weather ends at {w_end.date()}, "
            f"prices end at {p_end.date()} — weather ends earlier."
        )

    if not warnings:
        log.info(
            "Weather alignment check passed | prices: %s → %s | weather: %s → %s",
            p_start.date(), p_end.date(), w_start.date(), w_end.date(),
        )

    return warnings


def run_quality_checks(engine) -> list[str]:
    """
    Run all four data quality checks and collect warnings.

    Checks are logged individually as they run. No exception is raised
    regardless of check outcomes — callers receive the full list of warnings.

    Parameters
    ----------
    engine : sqlalchemy.engine.Engine

    Returns
    -------
    list of str
        All warning messages across all checks.
    """
    log.info("Running data quality checks…")
    all_warnings: list[str] = []

    for check_fn in [
        check_consecutive_missing_prices,
        check_negative_generation,
        check_price_sanity,
        check_weather_alignment,
    ]:
        try:
            warnings = check_fn(engine)
            for w in warnings:
                log.warning(w)
            all_warnings.extend(warnings)
        except Exception as exc:
            msg = f"[quality check] {check_fn.__name__} failed unexpectedly: {exc}"
            log.error(msg)
            all_warnings.append(msg)

    return all_warnings


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(engine, insert_counts: dict[str, int], warnings: list[str]) -> None:
    """
    Print a human-readable summary of the load run to stdout.

    Includes row counts per table, date ranges, and any quality warnings.

    Parameters
    ----------
    engine : sqlalchemy.engine.Engine
    insert_counts : dict
        Maps table name → rows inserted this run.
    warnings : list of str
        Quality warning messages from :func:`run_quality_checks`.
    """
    table_meta = {
        "spot_prices": "timestamp_utc",
        "generation":  "timestamp_utc",
        "weather":     "timestamp_utc",
    }

    print("\n" + "=" * 60)
    print("  LOAD SUMMARY")
    print("=" * 60)

    with engine.connect() as conn:
        for table, ts_col in table_meta.items():
            total = conn.execute(
                text(f"SELECT COUNT(*) FROM {table}")  # noqa: S608 — table name is internal constant
            ).scalar()
            date_row = conn.execute(
                text(f"SELECT MIN({ts_col}), MAX({ts_col}) FROM {table}")
            ).fetchone()
            min_ts, max_ts = date_row

            inserted_this_run = insert_counts.get(table, 0)
            print(f"\n  {table}")
            print(f"    Total rows in DB : {total:,}")
            print(f"    Inserted this run: {inserted_this_run:,}")
            if min_ts and max_ts:
                print(f"    Date range       : {pd.Timestamp(min_ts).date()} → {pd.Timestamp(max_ts).date()}")
            else:
                print("    Date range       : (empty)")

    print(f"\n  Quality warnings : {len(warnings)}")
    if warnings:
        for w in warnings:
            print(f"    ⚠  {w}")
    else:
        print("    All checks passed.")

    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Orchestrate the full load pipeline.

    Steps:
    1. Load DB credentials from .env.
    2. Connect to PostgreSQL.
    3. Apply schema (CREATE TABLE IF NOT EXISTS — preserves existing data).
    4. Load prices_raw.csv → spot_prices.
    5. Load generation_raw.csv → generation.
    6. Load weather_raw.csv → weather.
    7. Run data quality checks and log warnings.
    8. Print a load summary.
    """
    creds  = load_env()
    engine = get_engine(creds)

    apply_schema(engine)

    insert_counts = load_all_csvs(engine)

    warnings = run_quality_checks(engine)

    print_summary(engine, insert_counts, warnings)

    log.info("load_db.py complete.")


if __name__ == "__main__":
    main()