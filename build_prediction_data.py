"""Availability-aligned China return targets, features, and chronological splits.

Run: python build_prediction_data.py
No fitting, tuning, or imputation. Reads existing databases; writes prediction_data.duckdb.
Cutoff/start: previous observed Chinese trading date at 15:00 Asia/Shanghai.
End: target Chinese trading date at 15:00. US UTC bars complete the following UTC
midnight, including weekend-folded metal observations (not available early Monday).
Publication delays are conservative ASSUMPTIONS, not measured arrival timestamps.
The default 24-hour China delay means latest China features may be two target rows
back. Override only with defensible publication evidence, not to improve scores.
Target HRP weights and selected contracts are frozen before the cutoff, avoiding
same-day Chinese contract-selection leakage. Targets may therefore differ from the
previous descriptive index on rebalance/roll dates. No settlement prices are used.
"""
import argparse
import json
from pathlib import Path
import duckdb
import numpy as np
import pandas as pd
from configs.paths import DATA_DIR, TUSHARE_DATABASE_PATH, BENTO_DATABASE_DIR
from configs.mapping import china_roots
from features import align_features_asof


CORE_FEATURES = ["return_lag0", "return_lag1", "absolute_return", "momentum_5",
    "volatility_5", "momentum_20", "volatility_20", "shock_20",
    "volume_relative_20", "volume_breakout_ratio_20", "volume_breakout_20",
    "positive_return_breadth", "constituent_count", "largest_weight", "shock_volume_interaction"]
CHINA_EXTRA = ["turnover_relative_20", "turnover_breakout_ratio_20", "turnover_breakout_20",
               "oi_change_1", "oi_change_5", "return_oi_interaction"]
DIAGNOSTICS = ["fallback", "daily_exclusion", "effective_constituents_changed",
               "available_weight", "interval_bars", "roll_flag"]


def china_close(dates):
    dates = pd.DatetimeIndex(pd.to_datetime(dates)).normalize()
    if dates.tz is not None:
        raise ValueError("Expected naive Chinese trading-date labels")
    return (dates + pd.Timedelta(hours=15)).tz_localize("Asia/Shanghai").tz_convert("UTC").as_unit("ns")


def add_availability(portfolios, cn_delay_hours=24, us_delay_minutes=5):
    """Attach explicit assumed publication times; never use bar starts as availability."""
    output = {}
    for market, group in portfolios.groupby("market"):
        group = group.copy()
        columns = CORE_FEATURES + (CHINA_EXTRA if market == "China" else [])
        if market == "China":
            group["available_at"] = china_close(group.trade_date) + pd.Timedelta(hours=cn_delay_hours)
        else:
            group["available_at"] = (pd.DatetimeIndex(group.trade_date).tz_localize("UTC")
                + pd.Timedelta(days=1, minutes=us_delay_minutes))
        group = group.rename(columns={"trade_date": "source_trade_date"})
        output[market] = group[["anchor", "source_trade_date", "available_at"] + columns
                            + [c for c in DIAGNOSTICS if c in group]]
    return output


def load_inputs(feature_path):
    with duckdb.connect(str(feature_path), read_only=True) as conn:
        features = conn.sql("SELECT * FROM portfolio_features").df()
    with duckdb.connect(str(DATA_DIR / "cluster_indexes.duckdb"), read_only=True) as conn:
        weights = conn.sql("SELECT * FROM anchor_weights WHERE market='China'").df()
    roots = list(dict.fromkeys(p for values in china_roots.values() for p in values))
    with duckdb.connect(str(TUSHARE_DATABASE_PATH), read_only=True) as conn:
        selected = conn.sql("SELECT trade_date,product,ts_code FROM rolling_futures").df()
        selected = selected.loc[selected["product"].isin(roots)]
        first = features.trade_date.min() - pd.Timedelta(days=60)
        selected = selected.loc[selected.trade_date.between(first, features.trade_date.max())]
        wanted = selected[["ts_code"]].drop_duplicates()
        conn.register("wanted", wanted)
        raw = conn.execute("""SELECT d.ts_code, d.trade_date, d.close, d.pre_close
            FROM future_daily d SEMI JOIN wanted w USING(ts_code)
            WHERE d.trade_date BETWEEN ? AND ?""",
            [first.strftime("%Y%m%d"), features.trade_date.max().strftime("%Y%m%d")]).df()
    raw["trade_date"] = pd.to_datetime(raw.trade_date, format="%Y%m%d")
    if raw.duplicated(["ts_code", "trade_date"]).any():
        raise ValueError("Duplicate raw contract/date rows")
    return features, weights, selected, raw


def freeze_targets(portfolios, weights, selected, raw, cn_delay_hours=24):
    """Construct same-contract close-to-close targets with cutoff-known exposure."""
    calendar = pd.DatetimeIndex(sorted(selected.trade_date.unique()))
    prior = dict(zip(calendar[1:], calendar[:-1]))
    cn = portfolios.loc[portfolios.market.eq("China") & portfolios.return_lag0.notna()]
    close_lookup = raw.set_index(["trade_date", "ts_code"])["close"].to_dict()
    selection = {}
    for product, group in selected.groupby("product"):
        group = group.sort_values("trade_date").copy()
        group["available_at"] = china_close(group.trade_date) + pd.Timedelta(hours=cn_delay_hours)
        selection[product] = group
    # Weekly labels are Saturdays; find the last actual Chinese session in each
    # estimation window before assigning its publication time.
    weights = weights.copy()
    last_session_pos = calendar.searchsorted(pd.DatetimeIndex(weights.estimation_end), side="right") - 1
    if (last_session_pos < 0).any():
        raise ValueError("Source calendar does not cover weight estimation end")
    estimate_available = china_close(calendar[last_session_pos]) + pd.Timedelta(hours=cn_delay_hours)
    schedule_available = pd.DatetimeIndex(weights.rebalance_date).tz_localize("Asia/Shanghai").tz_convert("UTC").as_unit("ns")
    weights["available_at"] = pd.to_datetime(np.maximum(estimate_available.asi8, schedule_available.asi8), utc=True)
    portfolios_by_anchor = {anchor: group for anchor, group in weights.groupby("anchor")}
    rows, audit = [], []
    for anchor, group in cn.groupby("anchor"):
        anchor_weights = portfolios_by_anchor.get(anchor, weights.iloc[:0])
        vintages = [(time, g) for time, g in anchor_weights.groupby("available_at", sort=True)]
        vintage_times = pd.DatetimeIndex([time for time, _ in vintages])
        for date in sorted(group.trade_date):
            if date not in prior:
                continue
            previous_date = prior[date]
            cutoff = china_close([previous_date])[0]
            end = china_close([date])[0]
            row = dict(anchor=anchor, target_trade_date=date, previous_trade_date=previous_date,
                prediction_cutoff=cutoff, target_start=cutoff, target_end=end,
                label_available_at=end + pd.Timedelta(hours=cn_delay_hours),
                target_return=np.nan, target_absolute_return=np.nan, target_status="no_cutoff_known_weights")
            position = vintage_times.searchsorted(cutoff, side="right") - 1 if len(vintages) else -1
            if position >= 0:
                weight_time, exposure = vintages[position]
                row["target_weight_available_at"] = weight_time
                row["target_rebalance_date"] = exposure.rebalance_date.iloc[0]
                total, valid = 0., True
                for item in exposure.itertuples():
                    candidates = selection.get(item.product)
                    chosen = None
                    if candidates is not None:
                        j = candidates.available_at.searchsorted(cutoff, side="right") - 1
                        if j >= 0:
                            chosen = candidates.iloc[j]
                    contract = chosen.ts_code if chosen is not None else None
                    start_price = close_lookup.get((previous_date, contract), np.nan)
                    end_price = close_lookup.get((date, contract), np.nan)
                    good = np.isfinite(start_price) and np.isfinite(end_price) and start_price > 0 and end_price > 0
                    ret = end_price / start_price - 1 if good else np.nan
                    total += float(item.weight) * ret
                    valid &= good
                    audit.append(dict(anchor=anchor, target_trade_date=date, product=item.product,
                        ts_code=contract, weight=float(item.weight), contract_return=ret,
                        contribution=float(item.weight)*ret, start_close=start_price, end_close=end_price,
                        contract_selection_date=chosen.trade_date if chosen is not None else pd.NaT,
                        contract_available_at=chosen.available_at if chosen is not None else pd.NaT,
                        weight_available_at=weight_time, prediction_cutoff=cutoff))
                row["target_status"] = "ok" if valid else "missing_frozen_contract_quote"
                if valid:
                    row["target_return"], row["target_absolute_return"] = total, abs(total)
            rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(audit)


def aligned_cross_features(dataset):
    """Historical lead-lag associations on forecast rows, not calendar-date EDA.

    Correlation/sensitivity compare cutoff-known Chinese returns to cutoff-known
    US returns; these can have different endpoints and are labelled accordingly.
    Repeated US observations are retained and documented by source-date/age fields.
    """
    output = []
    for anchor, group in dataset.groupby("anchor", sort=False):
        group = group.sort_values("prediction_cutoff").copy()
        cn, us = group.cn_return_lag0, group.us_return_lag0
        paired = cn.notna() & us.notna()
        x, y = cn.where(paired), us.where(paired)
        group["cross_correlation_60"] = x.rolling(60, min_periods=30).corr(y)
        var = y.rolling(60, min_periods=30).var()
        group["cross_sensitivity_60"] = x.rolling(60, min_periods=30).cov(y).div(var.where(var.gt(0)))
        group["cross_paired_count_60"] = paired.rolling(60, min_periods=1).sum()
        cn_scale = group.cn_volatility_20 * np.sqrt(20)
        us_scale = group.us_volatility_20 * np.sqrt(20)
        group["cross_momentum_divergence_20"] = (
            group.cn_momentum_20.div(cn_scale.where(cn_scale.gt(0)))
            - group.us_momentum_20.div(us_scale.where(us_scale.gt(0))))
        output.append(group)
    return pd.concat(output, ignore_index=True)


def split_and_cv(dataset, test_start="2025-10-01"):
    """Date-based expanding folds; purge labels not known at evaluation cutoff."""
    data = dataset.copy()
    boundary = pd.Timestamp(test_start)
    data["split"] = np.where(data.target_trade_date.ge(boundary), "test", "train")
    ready = data.target_return.notna() & data.cn_return_lag0.notna() & data.us_return_lag0.notna()
    data["model_eligible"] = ready
    # Same dates for domestic-only and cross-market comparisons. Retain partial
    # predictor NaNs for future training-only imputation, never impute here.
    for anchor, group in data.groupby("anchor"):
        test = group.loc[group.split.eq("test") & group.model_eligible]
        if not test.empty:
            too_late = group.split.eq("train") & group.label_available_at.gt(test.prediction_cutoff.min())
            data.loc[group.index[too_late], "split"] = "purged_boundary"
    fold_rows = []
    for anchor, group in data.loc[data.split.eq("train") & data.model_eligible].groupby("anchor"):
        for fold, (start, end) in enumerate([
            ("2024-04-01", "2024-10-01"), ("2024-10-01", "2025-04-01"),
            ("2025-04-01", "2025-10-01")], 1):
            validation = group.loc[group.target_trade_date.ge(start) & group.target_trade_date.lt(end)]
            if validation.empty:
                continue
            cutoff = validation.prediction_cutoff.min()
            training = group.loc[group.target_trade_date.lt(start) & group.label_available_at.le(cutoff)]
            for role, subset in [("train", training), ("validation", validation)]:
                fold_rows.extend(dict(anchor=anchor, fold=fold, role=role,
                    target_trade_date=date, validation_first_cutoff=cutoff) for date in subset.target_trade_date)
    return data, pd.DataFrame(fold_rows)


def validate_alignment(data, audit):
    for prefix in ("cn", "us"):
        present = data[f"{prefix}_available_at"].notna()
        if not (data.loc[present, f"{prefix}_available_at"] <= data.loc[present, "prediction_cutoff"]).all():
            raise ValueError(f"Future {prefix} feature joined")
    if not (data.prediction_cutoff <= data.target_start).all() or not (data.target_start < data.target_end).all():
        raise ValueError("Invalid forecast/target intervals")
    known = audit.contract_available_at.notna()
    if not (audit.loc[known, "contract_available_at"] <= audit.loc[known, "prediction_cutoff"]).all():
        raise ValueError("Future contract selection")
    if not (audit.weight_available_at <= audit.prediction_cutoff).all():
        raise ValueError("Future target weights")
    if not np.allclose(audit.groupby(["anchor", "target_trade_date"]).weight.sum(), 1):
        raise ValueError("Target weights do not sum to one")
    if np.isinf(data.select_dtypes(include="number").to_numpy()).any():
        raise ValueError("Infinite dataset values")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=DATA_DIR / "features.duckdb")
    parser.add_argument("--output", type=Path, default=DATA_DIR / "prediction_data.duckdb")
    parser.add_argument("--test-start", default="2025-10-01")
    parser.add_argument("--cn-delay-hours", type=float, default=24)
    parser.add_argument("--us-delay-minutes", type=float, default=5)
    parser.add_argument("--max-feature-age", default="30D")
    args = parser.parse_args()
    sources = [args.features, DATA_DIR / "cluster_indexes.duckdb", TUSHARE_DATABASE_PATH, BENTO_DATABASE_DIR]
    if args.output.resolve() in [Path(p).resolve() for p in sources]:
        parser.error("Output cannot overwrite source databases")
    if args.cn_delay_hours < 0 or args.us_delay_minutes < 0:
        parser.error("Publication delays cannot be negative")
    print("Reading features and constructing cutoff-frozen targets...", flush=True)
    features, weights, selected, raw = load_inputs(args.features)
    targets, audit = freeze_targets(features, weights, selected, raw, args.cn_delay_hours)
    available = add_availability(features, args.cn_delay_hours, args.us_delay_minutes)
    data = align_features_asof(available["China"], targets, "cn", args.max_feature_age)
    data = align_features_asof(available["US"], data, "us", args.max_feature_age)
    data = aligned_cross_features(data)
    data, folds = split_and_cv(data, args.test_start)
    validate_alignment(data, audit)
    # Explicit allowlist: targets, audit timestamps and diagnostics never enter X.
    domestic = ["cn_" + c for c in CORE_FEATURES + CHINA_EXTRA]
    us = ["us_" + c for c in CORE_FEATURES]
    cross = ["cross_correlation_60", "cross_sensitivity_60", "cross_momentum_divergence_20"]
    columns = pd.DataFrame([dict(feature=c, feature_group=g) for g, names in
        [("domestic", domestic), ("us", us), ("cross_market", cross)] for c in names])
    summary = data.groupby(["anchor", "split"]).agg(rows=("target_trade_date", "size"),
        eligible_rows=("model_eligible", "sum"), start=("target_trade_date", "min"), end=("target_trade_date", "max")).reset_index()
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    tables = {"prediction_dataset": data, "target_contributions": audit, "cv_membership": folds,
        "feature_columns": columns, "split_summary": summary,
        "run_config": pd.DataFrame([dict(created_at_utc=pd.Timestamp.now(tz="UTC"),
            config_json=json.dumps(config), availability_status="assumed publication delays; no historical arrival timestamps",
            target_rule="same-contract previous-close to target-close; HRP weights/contracts frozen at cutoff",
            cross_rule="rolling cutoff-known lead-lag associations; not concurrent market correlations")])}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(args.output)) as conn:
        conn.execute("BEGIN TRANSACTION")
        try:
            for name, frame in tables.items():
                conn.register("incoming", frame)
                conn.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM incoming")
                conn.unregister("incoming")
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise
    print(summary.to_string(index=False))
    print(f"Saved {args.output}; {len(domestic)} domestic + {len(us)} US + {len(cross)} cross-market candidates.")
    print("Timing and frozen-target checks passed. No models fitted; preprocessing remains training-only.")
    print("Publication delays are assumptions. China features default to a conservative 24-hour delay.")


if __name__ == "__main__":
    main()
