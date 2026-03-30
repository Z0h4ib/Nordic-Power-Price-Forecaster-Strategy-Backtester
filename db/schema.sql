-- Nordic Power Price Forecaster — PostgreSQL schema
-- Safe to rerun: drops and recreates all tables

DROP TABLE IF EXISTS backtest_metrics;
DROP TABLE IF EXISTS trades;
DROP TABLE IF EXISTS features;
DROP TABLE IF EXISTS weather;
DROP TABLE IF EXISTS generation;
DROP TABLE IF EXISTS spot_prices;


-- Day-ahead spot prices for DK1 and DK2
CREATE TABLE spot_prices (
    id              SERIAL PRIMARY KEY,
    timestamp_utc   TIMESTAMP NOT NULL,
    bidding_zone    VARCHAR(10) NOT NULL,   -- 'DK1' or 'DK2'
    price_eur_mwh   FLOAT,                  -- can be negative
    source          VARCHAR(50) DEFAULT 'ENTSO-E',
    UNIQUE (timestamp_utc, bidding_zone)
);

CREATE INDEX idx_spot_prices_timestamp ON spot_prices (timestamp_utc);


-- Wind and solar generation by bidding zone
CREATE TABLE generation (
    id               SERIAL PRIMARY KEY,
    timestamp_utc    TIMESTAMP NOT NULL,
    bidding_zone     VARCHAR(10) NOT NULL,
    wind_onshore_mw  FLOAT,
    wind_offshore_mw FLOAT,
    solar_mw         FLOAT,
    UNIQUE (timestamp_utc, bidding_zone)
);

CREATE INDEX idx_generation_timestamp ON generation (timestamp_utc);


-- Hourly weather observations at Danish grid points
CREATE TABLE weather (
    id              SERIAL PRIMARY KEY,
    timestamp_utc   TIMESTAMP NOT NULL,
    location        VARCHAR(50),            -- e.g. 'Aarhus', 'Copenhagen'
    temperature_c   FLOAT,
    wind_speed_ms   FLOAT,
    UNIQUE (timestamp_utc, location)
);

CREATE INDEX idx_weather_timestamp ON weather (timestamp_utc);


-- Model-ready feature dataset (Phase 2 output)
CREATE TABLE features (
    id                      SERIAL PRIMARY KEY,
    timestamp_utc           TIMESTAMP NOT NULL,
    bidding_zone            VARCHAR(10) NOT NULL,
    -- lag features
    price_lag_1h            FLOAT,
    price_lag_2h            FLOAT,
    price_lag_24h           FLOAT,
    price_lag_48h           FLOAT,
    price_lag_168h          FLOAT,
    price_rolling_mean_24h  FLOAT,
    price_rolling_std_24h   FLOAT,
    price_rolling_mean_168h FLOAT,
    -- calendar features
    hour                    INT,
    hour_sin                FLOAT,
    hour_cos                FLOAT,
    day_of_week             INT,
    is_weekend              BOOLEAN,
    month                   INT,
    month_sin               FLOAT,
    month_cos               FLOAT,
    is_danish_holiday       BOOLEAN,
    -- generation features
    wind_total_mw           FLOAT,
    wind_lag_1h             FLOAT,
    wind_lag_24h            FLOAT,
    solar_mw                FLOAT,
    renewables_ratio        FLOAT,
    -- weather features
    temp_aarhus             FLOAT,
    temp_cph                FLOAT,
    wind_speed_aarhus       FLOAT,
    wind_speed_cph          FLOAT,
    temp_mean_dk            FLOAT,
    -- target
    price_next_24h          FLOAT,
    UNIQUE (timestamp_utc, bidding_zone)
);

CREATE INDEX idx_features_timestamp ON features (timestamp_utc);

-- Backtest trade log (Phase 4 output)
CREATE TABLE trades (
    id              SERIAL PRIMARY KEY,
    timestamp_utc   TIMESTAMP NOT NULL,
    bidding_zone    VARCHAR(10) NOT NULL,
    signal          INT,
    forecast        FLOAT,
    forward_price   FLOAT,
    actual_price    FLOAT,
    hourly_pnl      FLOAT,
    threshold       FLOAT,            -- which threshold setting generated this trade
    UNIQUE (timestamp_utc, bidding_zone, threshold)
);
CREATE INDEX idx_trades_timestamp ON trades (timestamp_utc);

-- Backtest metrics (Phase 4 output)
CREATE TABLE backtest_metrics (
    id                     SERIAL PRIMARY KEY,
    run_timestamp          TIMESTAMP DEFAULT NOW(),
    zone                   VARCHAR(50) NOT NULL,
    threshold              FLOAT NOT NULL,
    total_return           FLOAT,
    sharpe_ratio           FLOAT,
    sortino_ratio          FLOAT,
    max_drawdown           FLOAT,
    trade_count            INT,
    win_rate               FLOAT,
    profit_factor          FLOAT,
    avg_win                FLOAT,
    avg_loss               FLOAT,
    max_consecutive_losses INT,
    UNIQUE (zone, threshold)
);
