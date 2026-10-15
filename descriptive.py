import pandas as pd
import duckdb as db

from configs.paths import TUSHARE_DATABASE_PATH, BENTO_DATABASE_DIR
from configs.mapping import us_roots, china_roots
from plot import compare_plot, zoomin_sector, histogram, sector_return_boxplot, rolling_volatility_plot, market_correlation_heatmap

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
            Where trade_date > DATE '2021-10-01' and trade_date < DATE '2026-10-02'
            """).fetchdf() 

    return rollover_df

def bento_continuous(conn):
    continueous_df = conn.sql("""
            SELECT * 
            FROM rolling_futures
            """).fetchdf() 

    return continueous_df

def metal_sector(root):
    metal = root['Metal'] 

    return metal

def energy_sector(root):
    engery = root['Energy'] 

    return engery

def agri_sector(root):
    agri = root['Agriculture'] 

    return agri

def cal_ret(df):
    df = df.copy()
    
    if "contract_return" in df.columns:
        df["pre_close"] = df["previous_contract_close"]
        df["ret"] = df["contract_return"]

    elif 'pre_close' in df.columns:
        previous = df['pre_close'].where(df['pre_close'] != 0)
        df['ret'] = (df['close'] - previous) / previous

    return df 

def main():
    # tushare data --- start from Oct 8th, because of national day
    conn_tushare = connect_duckdb(TUSHARE_DATABASE_PATH)
    rollover_df = tushare_continuous(conn_tushare)
    conn_tushare.close()
    tushare_df = rollover_df.sort_values(
        ["product","trade_date"]).reset_index(drop=True)  

    # bento data
    conn_bento = connect_duckdb(BENTO_DATABASE_DIR)
    continueous_df = bento_continuous(conn_bento)
    conn_bento.close()
    continueous_df = continueous_df.sort_values(
        ["product","trade_date"]).reset_index(drop=True)  

    # drop the value of uneeded columns
    bento_df = continueous_df.copy().drop(columns=['source_dataset', 'source_request_id',
                                                   'contract_sort', 'delist_date', 'switch_reason',
                                                   ])
    bento_df["bar_start_utc"] = pd.to_datetime(bento_df["ts_event_ns"], unit="ns", utc=True,)
    
    # data description 
    print("Chinese futures market selected dataset:", tushare_df.shape)
    print("United States futures market corresponding:", bento_df.shape)
    print(tushare_df.info())
    print(bento_df.info())

    # get metal sector list
    metal_china_list = metal_sector(china_roots)
    metal_us_list = metal_sector(us_roots)

    metal_cn = tushare_df.loc[tushare_df['product'].isin(metal_china_list)].reset_index(drop=True)
    metal_us = bento_df.loc[bento_df['product'].isin(metal_us_list)]

    # get energy sector list
    energy_china_list = energy_sector(china_roots)
    energy_us_list = energy_sector(us_roots)

    energy_cn = tushare_df[tushare_df["product"].isin(energy_china_list)]
    energy_us = bento_df[bento_df["product"].isin(energy_us_list)]

    # get argriculture sector list
    agri_china_list = agri_sector(china_roots)
    agri_us_list = agri_sector(us_roots)

    agri_cn = tushare_df[tushare_df["product"].isin(agri_china_list)]
    agri_us = bento_df[bento_df["product"].isin(agri_us_list)]

    # ALL 
    all_three_list_cn = metal_china_list + energy_china_list + agri_china_list
    all_three_list_us = metal_us_list + energy_us_list + agri_us_list

    all_cn = tushare_df[tushare_df["product"].isin(all_three_list_cn)]
    all_us = bento_df[bento_df["product"].isin(all_three_list_us)]

    # ret for both market
    metal_cn = cal_ret(metal_cn)
    metal_us = cal_ret(metal_us)

    energy_cn = cal_ret(energy_cn)
    energy_us = cal_ret(energy_us)

    agri_cn = cal_ret(agri_cn)
    agri_us = cal_ret(agri_us)

    all_cn = cal_ret(all_cn)
    all_us = cal_ret(all_us)

    # grouby one day and calculate averge return of a sector
    metal_sector_cn = (metal_cn.groupby("trade_date").agg(
        sector_return=("ret", "mean"),
        available_products=("ret", "count"),
    ).sort_index())

    metal_sector_us = (metal_us.groupby("trade_date").agg(
        sector_return = ("ret", "mean"),
        available_products = ("ret", "count")
    )).sort_index()

    energy_sector_cn = (energy_cn.groupby("trade_date").agg(
        sector_return=("ret", "mean"),
        available_products=("ret", "count"),
    ).sort_index())

    energy_sector_us = (energy_us.groupby("trade_date").agg(
        sector_return = ("ret", "mean"),
        available_products = ("ret", "count")
    )).sort_index()

    agri_sector_cn = (agri_cn.groupby("trade_date").agg(
        sector_return=("ret", "mean"),
        available_products=("ret", "count"),
    ).sort_index())

    agri_sector_us = (agri_us.groupby("trade_date").agg(
        sector_return = ("ret", "mean"),
        available_products = ("ret", "count")
    )).sort_index()

    all_sector_cn = (all_cn.groupby("trade_date").agg(
        sector_return = ("ret", "mean"),
        available_products = ("ret", "count")
    )).sort_index()

    all_sector_us = (all_us.groupby("trade_date").agg(
        sector_return = ("ret", "mean"),
        available_products = ("ret", "count")
    )).sort_index()

    for sector_frame, constituent_frame in [
        (metal_sector_cn, metal_cn), (metal_sector_us, metal_us),
        (energy_sector_cn, energy_cn), (energy_sector_us, energy_us),
        (agri_sector_cn, agri_cn), (agri_sector_us, agri_us),
    ]:
        sector_frame.attrs["constituent_returns"] = constituent_frame[
            ["trade_date", "product", "ret"]
        ].to_dict("records")

    # plot
    for sector, cn, us in [
        ("Metals", metal_sector_cn, metal_sector_us),
        ("Energy", energy_sector_cn, energy_sector_us),
        ("Agriculture", agri_sector_cn, agri_sector_us),
        ("All Sectors", all_sector_cn, all_sector_us)
    ]:
        compare_plot(cn, us, sector=sector)
        zoomin_sector(cn, us, sector=sector)
        print(f"Saved {sector} comparison and zoom-in figures.")
    
    histogram(metal_sector_cn, metal_sector_us, sector="Metal")
    histogram(energy_sector_cn, energy_sector_us, sector="Energy")
    histogram(agri_sector_cn, agri_sector_us, sector="Agriculure")
    histogram(all_sector_cn, all_sector_us, sector="All")

    boxplot_path = sector_return_boxplot({
        "Metals": (metal_sector_cn, metal_sector_us),
        "Energy": (energy_sector_cn, energy_sector_us),
        "Agriculture": (agri_sector_cn, agri_sector_us),
    })
    print(f"Saved boxplot: {boxplot_path}")

    volatility_path = rolling_volatility_plot({
        "Metals": (metal_sector_cn, metal_sector_us),
        "Energy": (energy_sector_cn, energy_sector_us),
        "Agriculture": (agri_sector_cn, agri_sector_us),
        "All Sectors": (all_sector_cn, all_sector_us),
    }, window=20)
    print(f"Saved rolling volatility: {volatility_path}")

    for market, constituents, mapping in [
        ("China", all_cn, china_roots),
        ("US", all_us, us_roots),
    ]:
        heatmap_path = market_correlation_heatmap(constituents, market, mapping)
        print(f"Saved {market} heatmap: {heatmap_path}")


if __name__ == "__main__":
    main()
