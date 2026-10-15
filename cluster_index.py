"""Monthly adaptive commodity clusters with pure HRP, estimated from weekly returns.

Run: python cluster_index.py
Daily portfolio returns use fixed daily target weights during each monthly vintage.
US copper excludes missing-return members and renormalizes daily available weights;
this observation-based convention is descriptive, not an executable trading rule.
This is a research return index, not a self-financing buy-and-hold backtest.
US metals weekend returns are compounded into Monday; other market dates are retained.
Weekly W-SAT labels are used. Cross-market forecast
cutoffs still require separate information-availability alignment.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import duckdb
from scipy.cluster.hierarchy import linkage, fcluster, leaves_list
from scipy.spatial.distance import squareform
from descriptive import cal_ret, tushare_continuous, bento_continuous
from configs.paths import TUSHARE_DATABASE_PATH, BENTO_DATABASE_DIR, DATA_DIR
from configs.mapping import china_roots, us_roots

ANCHORS = {
    "gold": {"China": "AU", "US": "GC"},
    "copper": {"China": "CU", "US": "HG"},
    "crude": {"China": "SC", "US": "CL"},
    "soybeans": {"China": "B", "US": "ZS"},
    "wheat": {"China": "WH", "US": "ZW"},
}

def hierarchy(returns, method="average"):
    corr = returns.corr().clip(-1, 1)
   
    if returns.shape[1] < 2 or not np.isfinite(corr.to_numpy()).all():
        raise ValueError("Need at least two varying products with finite correlations")
   
    distance = np.sqrt(np.maximum(0, (1 - corr.to_numpy()) / 2))
    distance = (distance + distance.T) / 2
    np.fill_diagonal(distance, 0)
    
    return linkage(squareform(distance), method=method)


def hrp_weights(returns, method="average"):
    
    """Standard inverse-variance cluster risks and recursive bisection."""
    
    if returns.isna().any().any():
        raise ValueError("HRP expects complete historical weekly observations")
    
    tree = hierarchy(returns, method)
    
    ordered = returns.columns[leaves_list(tree)].tolist()
    cov = returns.cov().loc[ordered, ordered]
    
    if (np.diag(cov) <= 0).any():
        raise ValueError("HRP covariance has nonpositive variance")
    weights = pd.Series(1.0, index=ordered)

    def variance(members):
        sub = cov.loc[members, members].to_numpy()
        inverse = 1 / np.diag(sub)
        inverse /= inverse.sum()
        return float(inverse @ sub @ inverse)

    pending = [ordered]
    
    while pending:
        members = pending.pop()
        
        if len(members) < 2:
            continue
        middle = len(members) // 2
        
        left, right = members[:middle], members[middle:]
        left_var, right_var = variance(left), variance(right)
        
        if left_var + right_var <= 0:
            raise ValueError("Invalid cluster variance")
        
        alpha = right_var / (left_var + right_var)
        weights.loc[left] *= alpha
        weights.loc[right] *= 1 - alpha
        pending.extend([left, right])
    
    return weights / weights.sum()


def metals_weekday_returns(daily, products):
    """Compound weekend returns into Monday, requiring an observed Monday return.

    Weekday gaps stay missing. Source databases are untouched. The combined
    return is available only once the destination weekday bar is complete.
    """
    metals = daily.loc[:, [p for p in products if p in daily]].copy()
    weekdays = metals.index.dayofweek
    offsets = np.where(weekdays >= 5, 7 - weekdays, 0)
    labels = metals.index + pd.to_timedelta(offsets, unit="D")
    grouped = (1 + metals).groupby(labels).prod(min_count=1) - 1
    # A weekend-only observation must not masquerade as a completed Monday bar.
    destination = metals.reindex(grouped.index)
    grouped = grouped.where(destination.notna())
    # The source panel records absent product dates as NaN; no zero returns are inserted.
    other = daily.drop(columns=metals.columns)
    result = pd.concat([other, grouped], axis=1).sort_index()
    result = result.reindex(columns=daily.columns)
    return result.dropna(how="all")


def load_market(market):
    path, loader, mapping = ((TUSHARE_DATABASE_PATH, tushare_continuous, china_roots)
                            if market == "China" else
                            (BENTO_DATABASE_DIR, bento_continuous, us_roots))
    
    with duckdb.connect(str(path), read_only=True) as conn:
        frame = cal_ret(loader(conn))
    
    roots = list(dict.fromkeys(r for members in mapping.values() for r in members))
    frame = frame.loc[frame["product"].isin(roots)]
    daily = frame.pivot(index="trade_date", columns="product", values="ret").sort_index()
    daily.index = pd.to_datetime(daily.index)
    
    if np.isinf(daily.to_numpy(dtype=float)).any():
        raise ValueError(f"{market}: infinite daily returns")
    
    if (daily < -1).any().any():
        raise ValueError(f"{market}: return below -100%; audit data before compounding")
    
    if market == "US":
        daily = metals_weekday_returns(daily, us_roots["Metal"])
    return daily

def weekly_panel(daily):
    
    weekly = (1 + daily).resample("W-SAT").prod(min_count=1) - 1
    # Exclude the truncated first calendar week; partial-product weeks remain flagged by coverage.
    
    return weekly.loc[weekly.index - pd.Timedelta(days=6) >= daily.index.min().normalize()]


def build_market(daily, market, lookback=52, max_missing=10, threshold=0.46,
                 method="average", min_size=2):
    weekly = weekly_panel(daily)
    
    starts = pd.date_range(daily.index.min().to_period("M").start_time,
                           daily.index.max(), freq="MS")
    rows = {"cluster_membership": [], "cluster_weights": [], "cluster_daily": [],
            "return_contributions": [], "eligibility": [], "rebalance_diagnostics": []}
    
    for cutoff in starts:
        next_cutoff = cutoff + pd.offsets.MonthBegin(1)
        history = weekly.loc[weekly.index < cutoff].tail(lookback)
        
        if len(history) != lookback:
            continue
        
        missing = history.isna().sum()
        std = history.std()
        eligible = (missing <= max_missing) & std.gt(0) & std.notna()
        
        for product in history.columns:
            reason = ("too_many_missing_weeks" if missing[product] > max_missing else
                      "constant_or_undefined_returns" if not eligible[product] else "eligible")
            rows["eligibility"].append(dict(market=market, rebalance_date=cutoff,
                product=product, missing_weeks=int(missing[product]), reason=reason))
        
        clean = history.loc[:, eligible].dropna(how="any")
        
        # A common sample gives HRP a coherent covariance matrix; no zero filling.
        clean = clean.loc[:, clean.std().gt(0)]
        sufficient = len(clean) >= lookback - max_missing and clean.shape[1] >= 2
        
        rows["rebalance_diagnostics"].append(dict(market=market, rebalance_date=cutoff,
            common_weeks=len(clean), eligible_products=clean.shape[1],
            status="ok" if sufficient else "insufficient_common_history"))
        
        if not sufficient:
            continue
        
        tree = hierarchy(clean, method)
        labels = pd.Series(fcluster(tree, t=threshold, criterion="distance"), index=clean.columns)
        holding = daily.loc[(daily.index >= cutoff) & (daily.index < next_cutoff)]
        
        if holding.empty:
            continue
        
        for _, members_index in labels.groupby(labels).groups.items():
            members = sorted(members_index.tolist())
            membership_key = "|".join(members)
            
            identity = hashlib.sha256(membership_key.encode()).hexdigest()[:12]
            vintage = f"{market}_{cutoff:%Y%m%d}_{identity}"
            retained = len(members) >= min_size
            
            anchors = [name for name, mapping in ANCHORS.items() if mapping[market] in members]
        
            for product in members:
                rows["cluster_membership"].append(dict(market=market, rebalance_date=cutoff,
                    next_rebalance_date=next_cutoff, cluster_id=vintage,
                    membership_key=membership_key, product=product, retained=retained,
                    anchors="|".join(anchors)))
        
            if not retained:
                continue
        
            weights = hrp_weights(clean[members], method)
            assert np.isclose(weights.sum(), 1) and weights.ge(0).all()
        
            for product, weight in weights.items():
                rows["cluster_weights"].append(dict(market=market, rebalance_date=cutoff,
                    cluster_id=vintage, product=product, weight=float(weight),
                    estimation_start=clean.index.min(), estimation_end=clean.index.max(),
                    common_weeks=len(clean)))
        
            values = holding.reindex(columns=weights.index)
            # No cluster observation when every constituent market lacks a bar (e.g. agriculture Sundays).
            values = values.dropna(how="all")
            
            contributions = values.mul(weights, axis=1)
            complete = values.notna().all(axis=1)
            portfolio = contributions.sum(axis=1, min_count=len(weights))
            
            index_values = 100 * (1 + portfolio).cumprod(skipna=False)
        
            for date in values.index:
                available = values.loc[date].notna()
                
                rows["cluster_daily"].append(dict(market=market, trade_date=date,
                    rebalance_date=cutoff, cluster_id=vintage, membership_key=membership_key,
                    
                    anchors="|".join(anchors), cluster_return=portfolio.loc[date],
                    
                    vintage_index=index_values.loc[date], complete=bool(complete.loc[date]),
                    available_weight=float(weights.loc[available].sum()),
                    
                    missing_products="|".join(values.columns[~available]),
                    constituent_count=len(members)))
                
                for product in weights.index:
                    rows["return_contributions"].append(dict(market=market, trade_date=date,
                        
                        rebalance_date=cutoff, cluster_id=vintage, product=product,
                        weight=float(weights[product]), product_return=values.loc[date, product],
                        
                        contribution=contributions.loc[date, product]))
        
        print(f"{market} {cutoff:%Y-%m}: {clean.shape[1]} products, {len(clean)} common weeks", flush=True)
    return {name: pd.DataFrame(values) for name, values in rows.items()}

def make_anchor_pairs(membership):
    output = []
    
    for cutoff, group in membership.groupby("rebalance_date"):
        for anchor, codes in ANCHORS.items():
            row = dict(rebalance_date=cutoff, anchor=anchor)
            
            for market, code in codes.items():
                match = group.loc[group.market.eq(market) & group["product"].eq(code) & group.retained]
                row[f"{market.lower()}_cluster_id"] = None if match.empty else match.cluster_id.iloc[0]
            
            row["available"] = all(row[f"{m}_cluster_id"] is not None for m in ["china", "us"])
            
            output.append(row)
    
    result = pd.DataFrame(output)
    if not result.empty:
        result["duplicate_portfolio_pair"] = result.duplicated(
            ["rebalance_date", "china_cluster_id", "us_cluster_id"], keep=False) & result.available
    return result


def build_anchor_indexes(market_daily, membership, diagnostics, lookback=52,
                         max_missing=10, method="average", start_value=100):
    """Use current eligible clusters, or retain previous members for singleton anchors.

    Raw cluster tables remain unchanged. Fallback portfolios and their newly
    estimated weights are stored separately so they cannot masquerade as clusters.
    """
    parts, rebalances, weight_rows, contribution_rows = [], [], [], []
    
    for market, daily in market_daily.items():
        weekly = weekly_panel(daily)
        mapping = china_roots if market == "China" else us_roots
        cutoffs = diagnostics.loc[diagnostics.market.eq(market), "rebalance_date"]
        
        for anchor, codes in ANCHORS.items():
            code = codes[market]
            sector = next(p for p in mapping.values() if code in p)
            calendar = daily[[p for p in sector if p in daily]].dropna(how="all").index
            previous, previous_fallback = None, False
           
            for cutoff in cutoffs:
                next_cutoff = cutoff + pd.offsets.MonthBegin(1)
                dates = calendar[(calendar >= cutoff) & (calendar < next_cutoff)]
                history = weekly.loc[weekly.index < cutoff].tail(lookback)
                group = membership.loc[membership.market.eq(market)
                    & membership.rebalance_date.eq(cutoff)]
                match = group.loc[group["product"].eq(code)]
                current = None
                
                if not match.empty and bool(match.retained.iloc[0]):
                    current = sorted(group.loc[group.cluster_id.eq(match.cluster_id.iloc[0]), "product"])
                singleton = not match.empty and not bool(match.retained.iloc[0])
                
                fallback = current is None and singleton and previous is not None
                members = current if current is not None else previous if fallback else None
                
                status = "current_cluster" if current is not None else "singleton_fallback" if fallback else "no_eligible_anchor_cluster"
                weights = None
                
                if members is not None:
                    sample = history.reindex(columns=members).dropna(how="any")
                    if len(sample) >= lookback - max_missing and sample.std().gt(0).all():
                        weights = hrp_weights(sample, method)
                    else:
                        status = "insufficient_fallback_history" if fallback else "insufficient_cluster_history"
                
                active = weights is not None
                changed = active and previous is not None and members != previous
                
                fallback_start = active and fallback and not previous_fallback
                
                key = "|".join(members) if members is not None else None
                
                rebalances.append(dict(market=market, anchor=anchor, rebalance_date=cutoff,
                    membership_key=key, status=status, fallback=bool(fallback),
                    fallback_start=bool(fallback_start), constituents_changed=bool(changed)))
                
                frame = pd.DataFrame({"trade_date": dates})
                frame["market"], frame["anchor"], frame["anchor_product"] = market, anchor, code
                frame["rebalance_date"], frame["membership_key"] = cutoff, key
                
                frame["construction_status"], frame["fallback"] = status, bool(fallback)
                frame["fallback_start"], frame["constituents_changed"] = False, False
                frame["cluster_return"], frame["available_weight"] = np.nan, np.nan
                frame["missing_products"] = None
                
                if active:
                    values = daily.reindex(index=dates, columns=weights.index)
                    available_weight = values.notna().mul(weights, axis=1).sum(axis=1)
                    effective_weights = pd.DataFrame(
                        np.tile(weights.to_numpy(), (len(values), 1)),
                        index=values.index, columns=weights.index)
                    exclude_missing = market == "US" and anchor == "copper"
                    if exclude_missing:
                        # Descriptive available-member index: missing products get zero
                        # exposure; remaining monthly HRP weights are renormalized.
                        effective_weights = effective_weights.where(values.notna(), 0).div(
                            available_weight.where(available_weight.gt(0)), axis=0)
                    contributions = values.mul(effective_weights)
                    portfolio = contributions.sum(axis=1, min_count=1 if exclude_missing else len(weights))
                    frame["cluster_return"] = portfolio.to_numpy()
                    frame["available_weight"] = available_weight.to_numpy()
                    frame["daily_exclusion"] = (values.isna().any(axis=1) & available_weight.gt(0)).to_numpy() if exclude_missing else False
                    frame["missing_products"] = values.apply(lambda row: "|".join(row.index[row.isna()]), axis=1).to_numpy()
                    
                    for product, weight in weights.items():
                        weight_rows.append(dict(market=market, anchor=anchor, rebalance_date=cutoff,
                            product=product, weight=float(weight), fallback=bool(fallback),
                            estimation_start=sample.index.min(), estimation_end=sample.index.max()))
                        contribution_rows.extend(dict(market=market, anchor=anchor, trade_date=date,
                            rebalance_date=cutoff, product=product, weight=float(weight),
                            effective_weight=effective_weights.at[date, product],
                            product_return=values.at[date, product], contribution=contributions.at[date, product])
                            for date in dates)
                    
                    if len(frame):
                        frame.loc[frame.index[0], "fallback_start"] = fallback_start
                        frame.loc[frame.index[0], "constituents_changed"] = changed
                    
                    previous = members.copy()
                
                previous_fallback = bool(active and fallback)
                frame["status"] = np.where(frame.cluster_return.notna(), "ok",
                    "missing_constituent_return" if active else status)
                
                parts.append(frame)
    
    result = pd.concat(parts, ignore_index=True).sort_values(["market", "anchor", "trade_date"])
    # Track actual daily exposure separately from monthly cluster membership.
    result["effective_membership_key"] = None
    result["effective_constituents_changed"] = False
    for _, group in result.groupby(["market", "anchor"], sort=False):
        previous_effective = None
        for index, row in group.iterrows():
            if pd.isna(row.cluster_return):
                continue
            members = row.membership_key.split("|")
            if row.daily_exclusion:
                missing = set((row.missing_products or "").split("|"))
                members = [product for product in members if product not in missing]
            effective_key = "|".join(sorted(members))
            result.at[index, "effective_membership_key"] = effective_key
            result.at[index, "effective_constituents_changed"] = (
                previous_effective is not None and effective_key != previous_effective)
            previous_effective = effective_key
    result["index_value"], result["segment_index"] = np.nan, np.nan
    result["segment_id"] = pd.Series(pd.NA, index=result.index, dtype="Int64")
    
    for _, group in result.groupby(["market", "anchor"], sort=False):
        begun, broken, active = False, False, False
        value, segment_value, segment = start_value, start_value, 0
        
        for index, row in group.iterrows():
            if pd.isna(row.cluster_return):
                if begun:
                    broken = True
                active = False
                continue
            begun = True
            
            if not active:
                segment += 1
                segment_value, active = start_value, True
            segment_value *= 1 + row.cluster_return
            result.at[index, "segment_index"] = segment_value
            result.at[index, "segment_id"] = segment
            
            if not broken:
                value *= 1 + row.cluster_return
                result.at[index, "index_value"] = value
    
    return {"anchor_indexes": result.reset_index(drop=True),
            "anchor_rebalances": pd.DataFrame(rebalances),
            "anchor_weights": pd.DataFrame(weight_rows),
            "anchor_contributions": pd.DataFrame(contribution_rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lookback", type=int, default=52)
    parser.add_argument("--max-missing", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.46)
    parser.add_argument("--linkage", choices=["single", "average", "complete"], default="average")
    parser.add_argument("--min-size", type=int, default=2)
    parser.add_argument("--output", type=Path, default=DATA_DIR / "cluster_indexes.duckdb")
    
    args = parser.parse_args()
    
    if args.lookback < 4 or not 0 <= args.max_missing < args.lookback - 1 or args.min_size < 2:
        parser.error("Invalid lookback, missingness, or cluster-size setting")
    
    market_daily = {market: load_market(market) for market in ["China", "US"]}
    results = [build_market(market_daily[market], market, args.lookback, args.max_missing,
               args.threshold, args.linkage, args.min_size) for market in ["China", "US"]]
    
    combined = {name: pd.concat([r[name] for r in results], ignore_index=True)
                for name in results[0]}
    
    if combined["cluster_daily"].empty:
        raise ValueError("No cluster indices built; inspect historical coverage")
    
    combined["anchor_pairs"] = make_anchor_pairs(combined["cluster_membership"])
    combined.update(build_anchor_indexes(
        market_daily, combined["cluster_membership"], combined["rebalance_diagnostics"],
        args.lookback, args.max_missing, args.linkage,
    ))
    config = vars(args).copy()
    config["output"] = str(config["output"])
    combined["run_config"] = pd.DataFrame([{"created_at_utc": pd.Timestamp.now(tz="UTC"),
        "config_json": json.dumps(config), "weight_rule": "monthly HRP target weights; US copper renormalizes across observed members on missing-return dates",
        "calendar_rule": "US metals weekend returns compounded into Monday; destination weekday required; no weekday filling",
        "fallback_rule": "singleton anchor retains previous eligible members; historical HRP weights reestimated",
        "missing_rule": "US copper excludes missing members and renormalizes available weights; all-missing dates remain undefined; other anchors require complete returns"}])
    
    if args.output.resolve() in {Path(TUSHARE_DATABASE_PATH).resolve(), Path(BENTO_DATABASE_DIR).resolve()}:
        parser.error("Output must differ from the source databases")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    with duckdb.connect(str(args.output)) as conn:
        conn.execute("BEGIN TRANSACTION")
        try:
            for name, frame in combined.items():
                if frame.empty and not len(frame.columns):
                    continue
                conn.register("incoming", frame)
                conn.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM incoming")
                conn.unregister("incoming")
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise
    
    print("Saved", args.output)
    
    for name, frame in combined.items():
        print(name, len(frame))
    
    print("Vintage indices restart at 100 each month; cluster IDs are vintage-specific, not permanent sector labels.")
    print("anchor_indexes stitches monthly portfolios by anchor without resetting at rebalance.")
    print("index_value stays undefined after gaps; segment_index restarts explicitly with a new segment_id.")

if __name__ == "__main__":
    main()
