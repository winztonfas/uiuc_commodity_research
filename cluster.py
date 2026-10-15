from narwhals import corr
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

import pandas as pd
import duckdb as db

from configs.paths import TUSHARE_DATABASE_PATH, BENTO_DATABASE_DIR

from descriptive import cal_ret, connect_duckdb, tushare_continuous, bento_continuous
from configs.mapping import china_roots, us_roots
from plot import plot_dendrograms

def hrp_1(ret):
    # correlation matrix
    corr = ret.corr()

    if not np.isfinite(corr.values).all():
        raise ValueError("Correlation matrix contains non-finite values")

    corr_values = corr.to_numpy(copy=True)
    np.fill_diagonal(corr_values, 1)
    corr = pd.DataFrame(corr_values, index=corr.index, columns=corr.columns)
    
    # distance matrix
    d_corr = np.sqrt(0.5*(1-corr))

    dist = squareform(d_corr.values, checks=False)
    link = linkage(dist, method="average")

    return d_corr, link

def cal_weekly_ret(ret):
    weekly_ret = (1+ret).resample("W-SAT").prod(min_count = 1) -1 
    return weekly_ret

def get_all_clusters(d_corr, link, distance_threshold=0.43):
    labels = fcluster(
        link,
        t=distance_threshold,
        criterion="distance",
    )

    memberships = pd.Series(labels, index=d_corr.index)

    return [
        members.tolist()
        for _, members in memberships.groupby(memberships).groups.items()
        if len(members) >= 2
    ]

def main():
    china_list = [product for products in china_roots.values() for product in products]
    us_list = [product for products in us_roots.values() for product in products]

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

    all_cn = tushare_df[tushare_df["product"].isin(china_list)]
    all_us = bento_df[bento_df["product"].isin(us_list)]

    all_cn = cal_ret(all_cn)
    all_us = cal_ret(all_us)

    cn_returns = all_cn.pivot(
        index="trade_date",
        columns="product",
        values="ret",).sort_index()

    us_returns = all_us.pivot(
        index="trade_date",
        columns="product",
        values="ret",
    ).sort_index()

    cn_returns.index = pd.to_datetime(cn_returns.index)
    us_returns.index = pd.to_datetime(us_returns.index)

    # first example of looking back
    cutoff1 = pd.Timestamp("2026-01-01")
    cutoff2 = pd.Timestamp("2023-01-01")
    # 52 weeks
    lookback = 52
    
    # turing daily to weekly because the struture difference 
    cn_weekly_ret = cal_weekly_ret(cn_returns)
    us_weekly_ret = cal_weekly_ret(us_returns)
    
    cn_window1 = cn_weekly_ret.loc[cn_weekly_ret.index < cutoff1].tail(lookback)
    us_window1 = us_weekly_ret.loc[us_weekly_ret.index < cutoff1].tail(lookback)
    cn_window2 = cn_weekly_ret.loc[cn_weekly_ret.index < cutoff2].tail(lookback)
    us_window2 = us_weekly_ret.loc[us_weekly_ret.index < cutoff2].tail(lookback)

    na_cn1 = cn_window1.isna().sum()
    na_us1 = us_window1.isna().sum()
    na_cn2 = cn_window2.isna().sum()
    na_us2 = us_window2.isna().sum()

    # drropping missing value > 10, only china some contract have liquidity problem
    missing_count1 = na_cn1
    return_std1 = cn_window1.std()
    missing_count2 = na_cn2
    return_std2 = cn_window2.std()

    keep1 = (missing_count1 <= 10) & (return_std1 > 0)
    keep2 = (missing_count2 <= 10) & (return_std2 > 0)

    cn_window1 = cn_window1.loc[:,keep1]
    us_window1 = us_window1.loc[:,na_us1 <= 10]
    cn_window2 = cn_window2.loc[:,keep2]
    us_window2 = us_window2.loc[:,na_us2 <= 10]

    cn_d_corr1, cn_link1 = hrp_1(cn_window1)
    us_d_corr1, us_link1 = hrp_1(us_window1)
    cn_d_corr2, cn_link2 = hrp_1(cn_window2)
    us_d_corr2, us_link2 = hrp_1(us_window2)

    cn_clusters1 = get_all_clusters(cn_d_corr1,cn_link1,distance_threshold=0.46)
    us_clusters1 = get_all_clusters(us_d_corr1,us_link1, distance_threshold=0.46)
    cn_clusters2 = get_all_clusters(cn_d_corr2,cn_link2,distance_threshold=0.46)
    us_clusters2 = get_all_clusters(us_d_corr2,us_link2, distance_threshold=0.46)

    print("2026 cn clusters", cn_clusters1)
    print("2026 us clusters", us_clusters1)
    print("2023 cn clusters", cn_clusters2)
    print("2023 us clusters", us_clusters2)


    plot_dendrograms(
        cn_d_corr1, cn_link1,
        us_d_corr1, us_link1,
        distance_threshold=0.46,
        cutoff=cutoff1
    )

    plot_dendrograms(
        cn_d_corr2, cn_link2,
        us_d_corr2, us_link2,
        distance_threshold=0.46,
        cutoff=cutoff2
    )

if __name__ == "__main__":
    main()