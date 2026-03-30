"""
src/pipeline/fetch_weather.py

Fetches hourly historical weather observations for two Danish grid points
(Aarhus and Copenhagen) from the Open-Meteo Historical Weather API.

No API key required.
Data range: 2022-01-01 to 2024-12-31, hourly, UTC.

Output columns: timestamp_utc, location, temperature_c, wind_speed_ms.
"""

import logging
from pathlib import Path

import pandas as pd
import requests

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

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

LOCATIONS: dict[str, dict[str, float]] = {
    "Aarhus":     {"lat": 56.15, "lon": 10.21},
    "Copenhagen": {"lat": 55.67, "lon": 12.57},
}

DATA_START = "2022-01-01"
DATA_END   = "2024-12-31"

# Open-Meteo variable names → our column names
VARIABLE_MAP = {
    "temperature_2m":  "temperature_c",
    "wind_speed_10m":  "wind_speed_ms",
}

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR      = PROJECT_ROOT / "data" / "raw"

# Retry budget for transient HTTP errors
MAX_RETRIES = 3


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------

def fetch_weather_location(
    location: str,
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Fetch hourly temperature and wind speed for one location from Open-Meteo.

    Makes a single HTTP GET request covering the full date range; Open-Meteo
    archive handles multi-year windows without chunking. Timestamps are
    requested in UTC and returned as naive UTC datetimes.

    Parameters
    ----------
    location : str
        Human-readable location name stored in the ``location`` column,
        e.g. ``'Aarhus'``.
    lat : float
        Latitude of the grid point.
    lon : float
        Longitude of the grid point.
    start_date : str
        Fetch window start in ``YYYY-MM-DD`` format (inclusive).
    end_date : str
        Fetch window end in ``YYYY-MM-DD`` format (inclusive).

    Returns
    -------
    pd.DataFrame
        Columns: ``timestamp_utc`` (naive UTC), ``location``,
        ``temperature_c``, ``wind_speed_ms``.

    Raises
    ------
    requests.HTTPError
        If the API returns a non-2xx status code after all retries.
    KeyError
        If the expected variables are missing from the API response.
    """
    params = {
        "latitude":    lat,
        "longitude":   lon,
        "start_date":  start_date,
        "end_date":    end_date,
        "hourly":      ",".join(VARIABLE_MAP.keys()),
        "timezone":    "UTC",
        "wind_speed_unit": "ms",   # metres per second, not km/h (default)
    }

    log.info(
        "Fetching weather | %s (lat=%.2f, lon=%.2f) | %s → %s",
        location, lat, lon, start_date, end_date,
    )

    response = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(ARCHIVE_URL, params=params, timeout=60)
            response.raise_for_status()
            break
        except requests.HTTPError as exc:
            log.warning(
                "Attempt %d/%d failed for %s: %s", attempt, MAX_RETRIES, location, exc
            )
            if attempt == MAX_RETRIES:
                raise
        except requests.RequestException as exc:
            log.warning(
                "Attempt %d/%d — request error for %s: %s",
                attempt, MAX_RETRIES, location, exc,
            )
            if attempt == MAX_RETRIES:
                raise

    payload = response.json()
    hourly  = payload["hourly"]

    # Parse timestamps — Open-Meteo returns ISO 8601 strings in UTC
    # e.g. "2022-01-01T00:00" (no 'Z' suffix, but requested as UTC)
    timestamps = pd.to_datetime(hourly["time"], format="%Y-%m-%dT%H:%M", utc=False)

    df = pd.DataFrame({"timestamp_utc": timestamps})
    for api_col, dest_col in VARIABLE_MAP.items():
        df[dest_col] = hourly[api_col]

    df["location"] = location
    df = df[["timestamp_utc", "location", "temperature_c", "wind_speed_ms"]]

    log.info(
        "Weather fetched | %s | %d rows | %s → %s",
        location, len(df), df["timestamp_utc"].min(), df["timestamp_utc"].max(),
    )
    return df


# ---------------------------------------------------------------------------
# Combine all locations
# ---------------------------------------------------------------------------

def fetch_all_weather() -> pd.DataFrame:
    """
    Fetch weather for all configured locations and combine into one DataFrame.

    Iterates over ``LOCATIONS``, calls :func:`fetch_weather_location` for
    each, and concatenates the results. Failed locations are logged and
    skipped rather than aborting the run.

    Returns
    -------
    pd.DataFrame
        Columns: ``timestamp_utc``, ``location``, ``temperature_c``,
        ``wind_speed_ms``. Sorted by (timestamp_utc, location).
    """
    frames = []

    for name, coords in LOCATIONS.items():
        try:
            df = fetch_weather_location(
                location=name,
                lat=coords["lat"],
                lon=coords["lon"],
                start_date=DATA_START,
                end_date=DATA_END,
            )
            frames.append(df)
        except Exception as exc:
            log.error("Failed to fetch weather for %s: %s", name, exc)

    if not frames:
        raise RuntimeError("No weather data retrieved for any location.")

    combined = (
        pd.concat(frames, ignore_index=True)
        .sort_values(["timestamp_utc", "location"])
        .reset_index(drop=True)
    )
    log.info("Total weather rows (all locations): %d", len(combined))
    return combined


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_raw(df: pd.DataFrame, filename: str) -> None:
    """
    Save a DataFrame to ``data/raw/`` as a CSV backup.

    Creates the directory if it does not exist. Written without the
    DataFrame index.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to persist.
    filename : str
        Target filename, e.g. ``'weather_raw.csv'``.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / filename
    df.to_csv(path, index=False)
    log.info("Saved %d rows → %s", len(df), path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> pd.DataFrame:
    """
    Fetch all weather data, save a CSV backup, and return the DataFrame.

    Orchestrates the full weather pipeline:

    1. Fetch hourly temperature and wind speed for Aarhus and Copenhagen
       over 2022-01-01 → 2024-12-31.
    2. Save raw CSV to ``data/raw/weather_raw.csv``.
    3. Return the combined DataFrame for downstream use in ``load_db.py``.

    Returns
    -------
    pd.DataFrame
        Columns: ``timestamp_utc``, ``location``, ``temperature_c``,
        ``wind_speed_ms``. Ready for PostgreSQL insertion.
    """
    log.info(
        "Starting Open-Meteo fetch | %s → %s | locations: %s",
        DATA_START, DATA_END, ", ".join(LOCATIONS),
    )

    weather_df = fetch_all_weather()
    save_raw(weather_df, "weather_raw.csv")

    log.info("Open-Meteo fetch complete.")
    return weather_df


if __name__ == "__main__":
    main()