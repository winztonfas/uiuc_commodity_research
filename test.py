import pandas as pd 
import duckdb as db
from configs.paths import TUSHARE_DATABASE_PATH

def connect_duckdb(dir):
    return db.connect(str(dir), read_only=True)

def tushare_continuous(conn):
    rollover_df = conn.sql("""
            SELECT 
            ts_code, trade_date,
            open, close, pre_close, high, low,
            amount, oi,
            product, exchange
            FROM rolling_futures
            Where trade_date > DATE '2020-09-29' and trade_date < DATE '2026-10-02'
            """).fetchdf() 

    return rollover_df

def main():
    # tushare data 
    conn_tushare = connect_duckdb(TUSHARE_DATABASE_PATH)
    rollover_df = tushare_continuous(conn_tushare)
    conn_tushare.close()
    tushare_df = rollover_df.sort_values(
        ["product","trade_date"]).reset_index(drop=True)  

    print(rollover_df)

if __name__ == "__main__":
    main()


