# Phase 1 — Data Pipeline

## Goal

Pull raw electricity price and weather data from two public APIs and store it in a local PostgreSQL database. At the end of this phase the DB should have clean, queryable tables ready for EDA in Phase 2.

## Deliverables checklist

- [ ] Repo structure initialized (folders, .gitignore, requirements.txt, .env.example)
- [ ] `db/schema.sql` — PostgreSQL schema created and applied
- [ ] `src/pipeline/fetch_entso.py` — pulls DK1/DK2 day-ahead prices + wind/solar generation
- [ ] `src/pipeline/fetch_weather.py` — pulls hourly temp + wind speed for Denmark
- [ ] `src/pipeline/load_db.py` — loads all raw data into PostgreSQL
- [ ] Data loaded for full range: **2022-01-01 to 2024-12-31**
- [ ] Basic data quality checks pass (no full missing days, dtypes correct)

## Database schema

### Table: `spot_prices`
```sql
id              SERIAL PRIMARY KEY
timestamp_utc   TIMESTAMP NOT NULL
bidding_zone    VARCHAR(10) NOT NULL   -- 'DK1' or 'DK2'
price_eur_mwh   FLOAT                  -- can be negative
source          VARCHAR(50) DEFAULT 'ENTSO-E'
```

### Table: `generation`
```sql
id              SERIAL PRIMARY KEY
timestamp_utc   TIMESTAMP NOT NULL
bidding_zone    VARCHAR(10) NOT NULL
wind_onshore_mw FLOAT
wind_offshore_mw FLOAT
solar_mw        FLOAT
```

### Table: `weather`
```sql
id              SERIAL PRIMARY KEY
timestamp_utc   TIMESTAMP NOT NULL
location        VARCHAR(50)           -- e.g. 'Aarhus', 'Copenhagen'
temperature_c   FLOAT
wind_speed_ms   FLOAT
```

## API details

### ENTSO-E
- Register at: https://transparency.entsoe.eu → My Account → Security Token
- Python wrapper: `entsoe-py`
- Area codes: `DK_1` and `DK_2`
- Pull in monthly chunks to avoid timeouts

### Open-Meteo
- No API key needed
- Endpoint: `https://archive-api.open-meteo.com/v1/archive`
- Variables: `temperature_2m`, `wind_speed_10m`
- Pull for two coordinates:
  - Aarhus: lat=56.15, lon=10.21
  - Copenhagen: lat=55.67, lon=12.57

## Data quality checks to implement

1. No bidding zone should have more than 48 consecutive missing hours
2. Price range sanity: flag any price outside -500 to 3000 EUR/MWh as a warning (don't drop — negative prices are real)
3. Generation values should never be negative
4. Weather timestamps should align with price timestamps (hourly, UTC)

## Files to produce

```
src/pipeline/fetch_entso.py
src/pipeline/fetch_weather.py
src/pipeline/load_db.py
db/schema.sql
requirements.txt
.env.example
.gitignore
```

## Notes

- All timestamps must be stored in **UTC** — ENTSO-E returns local time by default, convert explicitly
- Pull data in monthly loops to avoid API timeouts
- Log every fetch with start/end date so reruns are debuggable
- `load_db.py` should be idempotent — safe to rerun without creating duplicate rows (use INSERT ... ON CONFLICT DO NOTHING)