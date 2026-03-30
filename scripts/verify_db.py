import pandas as pd
from sqlalchemy import text
from src.pipeline.load_db import load_env, get_engine
import logging

logging.basicConfig(level=logging.WARNING)

def main():
    creds = load_env()
    engine = get_engine(creds)
    query = """
    SELECT zone, threshold, sharpe_ratio, max_drawdown, win_rate, trade_count
    FROM backtest_metrics
    ORDER BY zone, threshold;
    """
    
    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn)
        print("\n" + "="*80)
        print("--- Verified Database Extraction ---")
        print(df.to_string(index=False))
        print("="*80 + "\n")

if __name__ == "__main__":
    main()
