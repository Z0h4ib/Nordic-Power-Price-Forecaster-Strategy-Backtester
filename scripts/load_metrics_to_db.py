import logging
from pathlib import Path

import pandas as pd
from sqlalchemy import MetaData
from sqlalchemy.dialects.postgresql import insert as pg_insert

from src.pipeline.load_db import apply_schema, get_engine, load_env

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

def main():
    creds = load_env()
    engine = get_engine(creds)
    
    # Apply schema to create the new tables robustly
    apply_schema(engine)
    
    metadata = MetaData()
    metadata.reflect(bind=engine, only=['trades', 'backtest_metrics'])
    trades_table = metadata.tables['trades']
    metrics_table = metadata.tables['backtest_metrics']
    
    RESULTS_DIR = Path("data/results")
    
    def load_trades(zone: str):
        path = RESULTS_DIR / f"trades_{zone.lower()}.parquet"
        if not path.exists():
            log.warning(f"No trade log found for {zone} at {path}")
            return
            
        df = pd.read_parquet(path)
        df['bidding_zone'] = zone
        df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'])
        
        # Replace NaN with None for robust insertion
        df = df.where(pd.notnull(df), None)
        
        cols = ['timestamp_utc', 'bidding_zone', 'signal', 'forecast', 'forward_price', 'actual_price', 'hourly_pnl']
        records = df[cols].to_dict(orient="records")
        
        inserted = 0
        with engine.begin() as conn:
            for i in range(0, len(records), 5000):
                chunk = records[i:i+5000]
                stmt = pg_insert(trades_table).values(chunk).on_conflict_do_nothing()
                result = conn.execute(stmt)
                inserted += result.rowcount
                
        log.info(f"[{zone}] Inserted {inserted} / {len(records)} trades (duplicates skipped).")

    load_trades("DK1")
    load_trades("DK2")
    
    # Load Metrics
    metrics_path = RESULTS_DIR / "backtest_metrics.csv"
    if metrics_path.exists():
        df_metrics = pd.read_csv(metrics_path)
        df_metrics = df_metrics.where(pd.notnull(df_metrics), None)
        records = df_metrics.to_dict(orient="records")
        
        inserted = 0
        with engine.begin() as conn:
            for i in range(0, len(records), 100):
                chunk = records[i:i+100]
                stmt = pg_insert(metrics_table).values(chunk).on_conflict_do_nothing()
                result = conn.execute(stmt)
                inserted += result.rowcount
        log.info(f"Inserted {inserted} rows into backtest_metrics table.")
    else:
        log.warning("No backtest_metrics.csv found.")

if __name__ == "__main__":
    main()
