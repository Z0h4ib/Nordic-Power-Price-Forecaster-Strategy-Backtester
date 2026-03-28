-- Nordic Power Price Forecaster — PostgreSQL schema
-- Safe to rerun: drops and recreates all tables

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
