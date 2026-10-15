"""Calculate candidate features and descriptive heatmaps; no ML fitting.

Run: python calculate_features.py
All input databases are read-only. Results go to data/processed/features.duckdb.
Local features include their current completed observation; no prediction cutoff
is assumed. Calendar-date cross-market features are exploratory only, NOT aligned
predictors. Availability alignment must precede the train/validation/test dataset.
"""
import argparse
import json
from pathlib import Path
import duckdb
import numpy as np
import pandas as pd
from configs.paths import DATA_DIR, BENTO_DATABASE_DIR, TUSHARE_DATABASE_PATH
from configs.mapping import china_roots, us_roots
from features import product_features, portfolio_features, cross_market_features
from plot import feature_correlation_heatmap


DIAGNOSTICS = ["fallback", "daily_exclusion", "effective_constituents_changed",
               "available_weight", "interval_bars", "roll_flag"]
IDENTIFIERS = ["market", "anchor", "trade_date"]


def read_inputs():
    """Use the same configured product universe as the cluster portfolios."""
    panels = []
    for market, path, mapping in [("China", TUSHARE_DATABASE_PATH, china_roots),
                                  ("US", BENTO_DATABASE_DIR, us_roots)]:
        with duckdb.connect(str(path), read_only=True) as conn:
            data = conn.sql("SELECT * FROM rolling_futures").df()
        roots = list(dict.fromkeys(p for members in mapping.values() for p in members))
        data = data.loc[data["product"].isin(roots)]
        panels.append(product_features(data, market))
    with duckdb.connect(str(DATA_DIR / "cluster_indexes.duckdb"), read_only=True) as conn:
        anchors = conn.sql("SELECT * FROM anchor_indexes").df()
        contributions = conn.sql("SELECT * FROM anchor_contributions").df()
    return pd.concat(panels, ignore_index=True), anchors, contributions


def calendar_cross_features(portfolios):
    """Calendar-date EDA only; preserve unmatched dates as missing, never fill.

    These dates do not establish concurrent economic return intervals or forecast
    availability. Keep this table separate from eventual predictive features.
    """
    results = []
    for anchor, group in portfolios.groupby("anchor"):
        if not all(group.loc[group.market.eq(m), "return_lag0"].notna().any()
                   for m in ("China", "US")):
            continue
        wide = group.pivot(index="trade_date", columns="market", values="return_lag0").sort_index()
        # Union dates retain missing observations rather than compressing time gaps.
        values = cross_market_features(wide["China"], wide["US"], min_observations=30)
        values["paired_observations_60"] = (wide["China"].notna() & wide["US"].notna()).rolling(60, min_periods=1).sum()
        cn = group.loc[group.market.eq("China")].set_index("trade_date")
        us = group.loc[group.market.eq("US")].set_index("trade_date")
        for market, frame in [("china", cn), ("us", us)]:
            for feature in ("return_lag0", "momentum_20", "volatility_20"):
                values[f"{market}_{feature}"] = frame[feature].reindex(values.index)
        # Momentum uses each local observation calendar, not a union grid with
        # manufactured gaps. Historical scales exclude the current observation.
        cn_scale = cn.return_lag0.shift(1).rolling(20).std() * np.sqrt(20)
        us_scale = us.return_lag0.shift(1).rolling(20).std() * np.sqrt(20)
        values["momentum_divergence_20"] = (
            cn.momentum_20.div(cn_scale.where(cn_scale.gt(0))).reindex(values.index)
            - us.momentum_20.div(us_scale.where(us_scale.gt(0))).reindex(values.index))
        values["anchor"] = anchor
        values["alignment"] = "calendar_date_eda_only"
        results.append(values.reset_index())
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


def feature_coverage(portfolios):
    rows = []
    for (market, anchor), group in portfolios.groupby(["market", "anchor"]):
        for column in group.columns:
            if column in IDENTIFIERS:
                continue
            value = group[column]
            rows.append(dict(market=market, anchor=anchor, feature=column,
                role="diagnostic" if column in DIAGNOSTICS else "candidate_predictor",
                rows=len(group), available=int(value.notna().sum()),
                missing_fraction=float(value.isna().mean()), unique_values=int(value.nunique()),
                first_valid_date=group.loc[value.notna(), "trade_date"].min(),
                last_valid_date=group.loc[value.notna(), "trade_date"].max()))
    return pd.DataFrame(rows)


def correlation_records(correlations, counts, market, anchor, scope):
    if correlations.empty:
        return pd.DataFrame()
    output = correlations.rename_axis("feature_x").reset_index().melt(
        id_vars="feature_x", var_name="feature_y", value_name="correlation")
    observed = counts.rename_axis("feature_x").reset_index().melt(
        id_vars="feature_x", var_name="feature_y", value_name="paired_observations")
    output = output.merge(observed, on=["feature_x", "feature_y"], validate="one_to_one")
    output["market"], output["anchor"], output["scope"] = market, anchor, scope
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DATA_DIR / "features.duckdb")
    parser.add_argument("--correlation", choices=["spearman", "pearson"], default="spearman")
    parser.add_argument("--min-observations", type=int, default=60)
    args = parser.parse_args()
    sources = [TUSHARE_DATABASE_PATH, BENTO_DATABASE_DIR, DATA_DIR / "cluster_indexes.duckdb"]
    if args.output.resolve() in [Path(p).resolve() for p in sources]:
        parser.error("Output must differ from source databases")
    if args.min_observations < 2:
        parser.error("min-observations must be at least two")
    print("Calculating product and portfolio features...", flush=True)
    products, anchors, contributions = read_inputs()
    portfolios = portfolio_features(anchors, products, contributions)
    calendar = calendar_cross_features(portfolios)
    for name, frame in [("products", products), ("portfolios", portfolios), ("calendar", calendar)]:
        if np.isinf(frame.select_dtypes(include="number").to_numpy()).any():
            raise ValueError(f"{name}: infinite feature values")
    coverage = feature_coverage(portfolios)
    heatmaps, matrices = [], []
    for (market, anchor), group in portfolios.groupby(["market", "anchor"]):
        columns = [c for c in portfolios if c not in IDENTIFIERS + DIAGNOSTICS]
        output, corr, counts = feature_correlation_heatmap(group,
            title=f"{market}: {anchor.title()} Portfolio Feature Correlations",
            filename=f"{market.lower()}_{anchor}_features.png", columns=columns,
            min_observations=args.min_observations, method=args.correlation)
        if output is not None:
            heatmaps.append(dict(market=market, anchor=anchor, scope="local_features", path=str(output)))
            matrices.append(correlation_records(corr, counts, market, anchor, "local_features"))
    if not calendar.empty:
        for anchor, group in calendar.groupby("anchor"):
            columns = [c for c in group if c not in ["anchor", "trade_date", "alignment", "paired_observations_60"]]
            output, corr, counts = feature_correlation_heatmap(group,
                title=f"{anchor.title()}: Cross-Market Calendar-Date EDA (Not Forecast Aligned)",
                filename=f"cross_market_{anchor}_calendar_eda.png", columns=columns,
                min_observations=args.min_observations, method=args.correlation)
            if output is not None:
                heatmaps.append(dict(market="China-US", anchor=anchor, scope="calendar_eda", path=str(output)))
                matrices.append(correlation_records(corr, counts, "China-US", anchor, "calendar_eda"))
    tables = {"product_features": products, "portfolio_features": portfolios,
              "calendar_cross_features_eda": calendar, "feature_coverage": coverage,
              "heatmap_files": pd.DataFrame(heatmaps),
              "feature_correlations": pd.concat(matrices, ignore_index=True) if matrices else pd.DataFrame(),
              "run_config": pd.DataFrame([dict(created_at_utc=pd.Timestamp.now(tz="UTC"),
                  correlation_method=args.correlation, min_observations=args.min_observations,
                  activity_scope="selected contract, not summed across expiries",
                  feature_timing="current completed observation; availability alignment pending",
                  calendar_cross_scope="EDA only; 60 union-date observations, minimum 30 paired returns; local 20-observation momentum; not a predictive dataset",
                  configuration=json.dumps({"china_roots": china_roots, "us_roots": us_roots}))])}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(args.output)) as conn:
        conn.execute("BEGIN TRANSACTION")
        try:
            for name, frame in tables.items():
                if not len(frame.columns):
                    continue
                conn.register("incoming", frame)
                conn.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM incoming")
                conn.unregister("incoming")
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise
    print(f"Saved {len(products):,} product rows and {len(portfolios):,} portfolio rows to {args.output}")
    print(f"Saved {len(heatmaps)} heatmaps; paths and paired sample counts are in the output database.")
    print("China wheat remains empty. Warmup/unsupported features remain NaN; no rows imputed or dropped.")
    print("Full-history EDA only. Define forecast cutoff/target intervals before building the ML dataset.")


if __name__ == "__main__":
    main()
