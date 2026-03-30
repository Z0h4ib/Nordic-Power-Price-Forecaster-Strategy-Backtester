"""
src/pipeline/fetch_entso.py

Fetches day-ahead electricity spot prices and actual wind/solar generation
data for DK1 and DK2 bidding zones from the ENTSO-E Transparency Platform.

Data range: 2022-01-01 to 2024-12-31 (fetched in monthly chunks).
All timestamps are stored as naive UTC.

Requires ENTSO_E_API_KEY in a .env file at the project root.
"""

import logging
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from entsoe import EntsoePandasClient

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

ZONES = {
    "DK1": "DK_1",
    "DK2": "DK_2",
}

# Inclusive start, exclusive end — covers full calendar years 2022-2024
DATA_START = pd.Timestamp("2022-01-01", tz="UTC")
DATA_END   = pd.Timestamp("2025-01-01", tz="UTC")

# Project root is two levels above this file: src/pipeline/fetch_entso.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR      = PROJECT_ROOT / "data" / "raw"

# Map ENTSO-E generation column names → our DB column names
GENERATION_COLUMNS = {
    "Wind Onshore":  "wind_onshore_mw",
    "Wind Offshore": "wind_offshore_mw",
    "Solar":         "solar_mw",
}

# Seconds to wait before each retry attempt on a 503 (exponential-ish backoff)
RETRY_WAIT_SECONDS = [10, 30, 60]

# Polite pause between every API call — reduces rate-limit pressure
INTER_REQUEST_SLEEP = 2.0


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

def load_env() -> str:
    """
    Load environment variables from the project-root .env file.

    Returns
    -------
    str
        The ENTSO-E API key.

    Raises
    ------
    EnvironmentError
        If ENTSO_E_API_KEY is missing from the environment.
    """
    env_path = PROJECT_ROOT / ".env"
    load_dotenv(dotenv_path=env_path)
    api_key = os.getenv("ENTSO_E_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ENTSO_E_API_KEY not found. "
            "Copy .env.example to .env and add your token."
        )
    return api_key


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------

def monthly_ranges(
    start: pd.Timestamp, end: pd.Timestamp
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Split a date range into consecutive monthly (chunk_start, chunk_end) pairs.

    The final chunk ends at *end*, not necessarily at a month boundary.
    This keeps each API call to at most one month, avoiding ENTSO-E timeouts.

    Parameters
    ----------
    start : pd.Timestamp
        Range start (tz-aware, inclusive).
    end : pd.Timestamp
        Range end (tz-aware, exclusive).

    Returns
    -------
    list of (pd.Timestamp, pd.Timestamp)
        Each tuple is (chunk_start, chunk_end) with the same timezone as input.
    """
    ranges = []
    cursor = start
    while cursor < end:
        # Advance to the first day of the next calendar month
        next_month_year  = cursor.year + (cursor.month // 12)
        next_month_month = (cursor.month % 12) + 1
        next_month = pd.Timestamp(
            year=next_month_year, month=next_month_month, day=1, tz=cursor.tzinfo
        )
        chunk_end = min(next_month, end)
        ranges.append((cursor, chunk_end))
        cursor = chunk_end
    return ranges


# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------

def _fetch_with_retry(call_fn: Callable[[], Any], description: str) -> Any:
    """
    Call ``call_fn()`` with exponential backoff retries on HTTP 503 errors.

    ENTSO-E returns 503 under two conditions observed in practice:
    - Rate limiting (requests fired too close together)
    - Paginated requests (offset > 0) hitting an overloaded shard

    The inter-request sleep (``INTER_REQUEST_SLEEP``) reduces the first cause;
    this retry loop recovers from the second.

    Parameters
    ----------
    call_fn : callable
        Zero-argument callable that performs one ENTSO-E API call.
    description : str
        Human-readable label used in log messages (e.g. "prices DK1 2023-01").

    Returns
    -------
    Any
        Whatever ``call_fn()`` returns on success.

    Raises
    ------
    Exception
        The last exception raised after all retry attempts are exhausted,
        or immediately for non-503 errors.
    """
    last_exc: Exception = Exception("Max retries exceeded without an exception")

    for attempt, wait in enumerate([0] + RETRY_WAIT_SECONDS, start=1):
        if wait:
            log.info(
                "Retrying %s (attempt %d/%d) — waiting %ds after 503…",
                description, attempt, len(RETRY_WAIT_SECONDS) + 1, wait,
            )
            time.sleep(wait)
        try:
            return call_fn()
        except Exception as exc:
            last_exc = exc
            if "503" in str(exc) and attempt <= len(RETRY_WAIT_SECONDS):
                log.warning("503 on attempt %d for %s", attempt, description)
                continue
            raise  # non-503, or retries exhausted — propagate immediately

    raise last_exc  # all 503 retries exhausted


# ---------------------------------------------------------------------------
# Price fetching
# ---------------------------------------------------------------------------

def fetch_prices_zone(
    client: EntsoePandasClient,
    label: str,
    area_code: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """
    Fetch day-ahead spot prices for one bidding zone over the full date range.

    Data is pulled in monthly chunks to avoid ENTSO-E API timeouts.
    Timestamps are converted to naive UTC before return.

    Parameters
    ----------
    client : EntsoePandasClient
        Authenticated ENTSO-E client.
    label : str
        Human-readable zone label, e.g. ``'DK1'``.
    area_code : str
        ENTSO-E area code, e.g. ``'DK_1'``.
    start : pd.Timestamp
        Fetch window start (UTC, inclusive).
    end : pd.Timestamp
        Fetch window end (UTC, exclusive).

    Returns
    -------
    pd.DataFrame
        Columns: ``timestamp_utc`` (naive UTC), ``bidding_zone``,
        ``price_eur_mwh``.
    """
    chunks = []

    for chunk_start, chunk_end in monthly_ranges(start, end):
        log.info(
            "Fetching prices | %s | %s → %s",
            label, chunk_start.date(), chunk_end.date(),
        )
        desc = f"prices {label} {chunk_start.date()}→{chunk_end.date()}"
        try:
            series = _fetch_with_retry(
                lambda s=chunk_start, e=chunk_end: client.query_day_ahead_prices(
                    area_code, start=s, end=e
                ),
                desc,
            )
            chunks.append(series)
        except Exception as exc:
            log.warning(
                "Skipping prices for %s %s → %s after all retries: %s",
                label, chunk_start.date(), chunk_end.date(), exc,
            )
        finally:
            time.sleep(INTER_REQUEST_SLEEP)

    if not chunks:
        log.warning("No price data retrieved for %s over full range.", label)
        return pd.DataFrame(
            columns=["timestamp_utc", "bidding_zone", "price_eur_mwh"]
        )

    combined: pd.Series = pd.concat(chunks)
    combined = combined[~combined.index.duplicated(keep="first")].sort_index()

    # Convert to UTC then strip timezone (store as naive UTC in the DB)
    combined.index = combined.index.tz_convert("UTC").tz_localize(None)

    df = (
        combined
        .rename_axis("timestamp_utc")
        .reset_index(name="price_eur_mwh")
    )
    df["bidding_zone"] = label
    df = df[["timestamp_utc", "bidding_zone", "price_eur_mwh"]]

    log.info(
        "Prices fetched | %s | %d rows | %s → %s",
        label, len(df), df["timestamp_utc"].min(), df["timestamp_utc"].max(),
    )
    return df


def fetch_all_prices(client: EntsoePandasClient) -> pd.DataFrame:
    """
    Fetch day-ahead prices for all zones and combine into one DataFrame.

    Parameters
    ----------
    client : EntsoePandasClient
        Authenticated ENTSO-E client.

    Returns
    -------
    pd.DataFrame
        Columns: ``timestamp_utc``, ``bidding_zone``, ``price_eur_mwh``.
        Sorted by (timestamp_utc, bidding_zone).
    """
    frames = [
        fetch_prices_zone(client, label, area_code, DATA_START, DATA_END)
        for label, area_code in ZONES.items()
    ]
    df = (
        pd.concat(frames, ignore_index=True)
        .sort_values(["timestamp_utc", "bidding_zone"])
        .reset_index(drop=True)
    )
    log.info("Total price rows (both zones): %d", len(df))
    return df


# ---------------------------------------------------------------------------
# Generation fetching
# ---------------------------------------------------------------------------

def fetch_generation_zone(
    client: EntsoePandasClient,
    label: str,
    area_code: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """
    Fetch actual wind and solar generation for one bidding zone.

    Pulls Wind Onshore, Wind Offshore, and Solar columns from the raw
    ENTSO-E generation response. Generation types absent from the API
    response (e.g. Wind Offshore in DK2) are filled with ``NaN``.

    Data is pulled in monthly chunks to avoid ENTSO-E API timeouts.
    Timestamps are converted to naive UTC before return.

    Parameters
    ----------
    client : EntsoePandasClient
        Authenticated ENTSO-E client.
    label : str
        Human-readable zone label, e.g. ``'DK1'``.
    area_code : str
        ENTSO-E area code, e.g. ``'DK_1'``.
    start : pd.Timestamp
        Fetch window start (UTC, inclusive).
    end : pd.Timestamp
        Fetch window end (UTC, exclusive).

    Returns
    -------
    pd.DataFrame
        Columns: ``timestamp_utc`` (naive UTC), ``bidding_zone``,
        ``wind_onshore_mw``, ``wind_offshore_mw``, ``solar_mw``.
    """
    chunks = []

    for chunk_start, chunk_end in monthly_ranges(start, end):
        log.info(
            "Fetching generation | %s | %s → %s",
            label, chunk_start.date(), chunk_end.date(),
        )
        desc = f"generation {label} {chunk_start.date()}→{chunk_end.date()}"
        try:
            raw = _fetch_with_retry(
                lambda s=chunk_start, e=chunk_end: client.query_generation(
                    area_code, start=s, end=e
                ),
                desc,
            )
            chunks.append(raw)
        except Exception as exc:
            log.warning(
                "Skipping generation for %s %s → %s after all retries: %s",
                label, chunk_start.date(), chunk_end.date(), exc,
            )
        finally:
            time.sleep(INTER_REQUEST_SLEEP)

    if not chunks:
        log.warning("No generation data retrieved for %s over full range.", label)
        return pd.DataFrame(
            columns=[
                "timestamp_utc", "bidding_zone",
                "wind_onshore_mw", "wind_offshore_mw", "solar_mw",
            ]
        )

    combined: pd.DataFrame = pd.concat(chunks)
    combined = combined[~combined.index.duplicated(keep="first")].sort_index()

    # entsoe-py may return MultiIndex columns: (type, 'Actual Aggregated')
    # Flatten to the first level so we can match by generation type name
    if isinstance(combined.columns, pd.MultiIndex):
        combined.columns = combined.columns.get_level_values(0)

    # Extract only the three generation types we need
    result = pd.DataFrame(index=combined.index)
    for raw_col, dest_col in GENERATION_COLUMNS.items():
        if raw_col in combined.columns:
            result[dest_col] = combined[raw_col]
        else:
            log.warning(
                "Column '%s' not present in generation response for %s "
                "— filling with NaN.",
                raw_col, label,
            )
            result[dest_col] = float("nan")

    # Convert to UTC then strip timezone
    result.index = result.index.tz_convert("UTC").tz_localize(None)
    result = result.rename_axis("timestamp_utc").reset_index()
    result["bidding_zone"] = label
    result = result[
        ["timestamp_utc", "bidding_zone", "wind_onshore_mw", "wind_offshore_mw", "solar_mw"]
    ]

    log.info(
        "Generation fetched | %s | %d rows | %s → %s",
        label, len(result),
        result["timestamp_utc"].min(), result["timestamp_utc"].max(),
    )
    return result


def fetch_all_generation(client: EntsoePandasClient) -> pd.DataFrame:
    """
    Fetch wind and solar generation for all zones and combine into one DataFrame.

    Parameters
    ----------
    client : EntsoePandasClient
        Authenticated ENTSO-E client.

    Returns
    -------
    pd.DataFrame
        Columns: ``timestamp_utc``, ``bidding_zone``, ``wind_onshore_mw``,
        ``wind_offshore_mw``, ``solar_mw``.
        Sorted by (timestamp_utc, bidding_zone).
    """
    frames = [
        fetch_generation_zone(client, label, area_code, DATA_START, DATA_END)
        for label, area_code in ZONES.items()
    ]
    df = (
        pd.concat(frames, ignore_index=True)
        .sort_values(["timestamp_utc", "bidding_zone"])
        .reset_index(drop=True)
    )
    log.info("Total generation rows (both zones): %d", len(df))
    return df


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_raw(df: pd.DataFrame, filename: str) -> None:
    """
    Save a DataFrame to ``data/raw/`` as a CSV backup.

    Creates the directory if it does not exist. The CSV is written without
    the DataFrame index.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to persist.
    filename : str
        Target filename, e.g. ``'prices_raw.csv'``.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / filename
    df.to_csv(path, index=False)
    log.info("Saved %d rows → %s", len(df), path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch all ENTSO-E data, save CSV backups, and return the two DataFrames.

    Orchestrates the full pipeline:
    1. Load API key from .env
    2. Fetch day-ahead prices for DK1 and DK2
    3. Fetch wind/solar generation for DK1 and DK2
    4. Save raw CSVs to data/raw/
    5. Return (prices_df, generation_df) for downstream use in load_db.py

    Returns
    -------
    tuple of pd.DataFrame
        ``(prices_df, generation_df)`` — both shaped and ready for
        PostgreSQL insertion via load_db.py.
    """
    api_key = load_env()
    client = EntsoePandasClient(api_key=api_key)

    log.info(
        "Starting ENTSO-E fetch | %s → %s",
        DATA_START.date(), (DATA_END - pd.Timedelta(days=1)).date(),
    )

    prices_df     = fetch_all_prices(client)
    generation_df = fetch_all_generation(client)

    save_raw(prices_df,     "prices_raw.csv")
    save_raw(generation_df, "generation_raw.csv")

    log.info("ENTSO-E fetch complete.")
    return prices_df, generation_df


if __name__ == "__main__":
    main()
