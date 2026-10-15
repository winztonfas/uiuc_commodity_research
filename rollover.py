"""Build US continuous-contract tables from raw Databento daily OHLCV.

Reference: Internship/system/data/preprocess/rollover.py.
Volume leader, two-day confirmation, forward-only rolls, optional expiry rule.
Raw UTC bars are unchanged. Forecast timing must still be aligned downstream.
"""
import argparse
import re
from pathlib import Path
import duckdb
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
RAW_DB = PROJECT_DIR / "data/raw/databento/commodities.duckdb"
OUT_DB = PROJECT_DIR / "data/processed/us_continuous.duckdb"
MONTHS = dict(zip("FGHJKMNQUVXZ", range(1, 13)))


def parse_month(symbol, root, trade_date):
    match = re.fullmatch(re.escape(root) + r"([FGHJKMNQUVXZ])(\d{1,4})", str(symbol))
    if not match:
        return None
    year_text = match.group(2)
    year = int(year_text)
    if len(year_text) == 1:
        # CME one-digit years: use the nearest non-stale decade around the bar.
        minimum = trade_date.year - 1
        year = (minimum // 10) * 10 + year
        if year < minimum:
            year += 10
    elif len(year_text) == 2:
        year += 2000 if year < 90 else 1900
    return year * 100 + MONTHS[match.group(1)]


def prepare_bars(frame, expiry=None):
    frame = frame.copy()
    frame["trade_date"] = pd.to_datetime(frame["ts_event_ns"], unit="ns", utc=True).dt.tz_localize(None)
    frame = frame.rename(columns={"product_root": "product", "symbol": "ts_code", "volume": "vol"})
    frame["contract_sort"] = [parse_month(s, r, d) for s, r, d in zip(frame["ts_code"], frame["product"], frame["trade_date"])]
    rejected = int(frame.contract_sort.isna().sum())
    frame = frame.dropna(subset=["contract_sort"]).copy()
    frame["contract_sort"] = frame.contract_sort.astype("int64")
    for col in ("open", "high", "low", "close"):
        frame[col] = frame[col].mask(frame[col] == 9223372036854775807) / 1_000_000_000
    frame = frame.dropna(subset=["close"])
    frame = frame[frame["close"] != 0].copy()
    keys = ["product", "trade_date", "ts_code"]
    if frame.duplicated(keys).any():
        raise ValueError("Duplicate outright bars; inspect instrument IDs before continuing.")
    if expiry is not None:
        required = {"product", "ts_code", "delist_date"}
        if not required.issubset(expiry.columns):
            raise ValueError("Expiry CSV needs product, ts_code, delist_date columns")
        expiry = expiry[list(required)].copy()
        if expiry.duplicated(["product", "ts_code"]).any():
            raise ValueError("Expiry CSV has duplicate product/symbol pairs")
        expiry["delist_date"] = pd.to_datetime(expiry.delist_date, errors="raise").dt.normalize()
        frame = frame.merge(expiry, on=["product", "ts_code"], how="left", validate="many_to_one")
    else:
        frame["delist_date"] = pd.NaT
    return frame.sort_values(keys), rejected


def rollover(frame, confirm_days=2, force_days=5):
    if confirm_days < 1 or force_days < 0:
        raise ValueError("Invalid confirmation/expiry settings")
    result = []
    for product, group in frame.groupby("product", sort=True):
        current, month, candidate, count = None, None, None, 0
        for date, daily in group.groupby("trade_date", sort=True):
            daily = daily.sort_values(["vol", "ts_code"], ascending=[False, True])
            # Zero-volume bars can describe a held contract, but cannot lead a roll.
            eligible = daily[daily.vol > 0]
            eligible = eligible[eligible.delist_date.isna() | (eligible.delist_date > date)]
            leader = eligible.iloc[0] if not eligible.empty else None
            reason = "hold"
            if current is None:
                if leader is None:
                    continue
                current, month = leader.ts_code, leader.contract_sort
                reason = "initialize"
            elif current not in set(daily.ts_code):
                forward = eligible[eligible.contract_sort >= month]
                if forward.empty:
                    candidate, count = None, 0
                    reason = "unavailable_no_forward"
                else:
                    selected = forward.iloc[0]
                    current, month = selected.ts_code, selected.contract_sort
                    candidate, count = None, 0
                    reason = "current_unavailable"
            else:
                held = daily[daily.ts_code == current].iloc[0]
                near_expiry = pd.notna(held.delist_date) and (held.delist_date - date).days <= force_days
                if near_expiry:
                    forward = eligible[eligible.contract_sort > month]
                    if not forward.empty:
                        selected = forward.iloc[0]
                        current, month = selected.ts_code, selected.contract_sort
                        candidate, count = None, 0
                        reason = "near_expiry"
                elif leader is None or leader.ts_code == current:
                    candidate, count = None, 0
                elif leader.contract_sort < month:
                    candidate, count = None, 0
                    reason = "prevent backward rollover"
                else:
                    count = count + 1 if candidate == leader.ts_code else 1
                    candidate = leader.ts_code
                    reason = f"candidate_{count}_of_{confirm_days}"
                    if count >= confirm_days:
                        current, month = candidate, leader.contract_sort
                        candidate, count = None, 0
                        reason = "confirmed_rollover"
            result.append(dict(trade_date=date, product=product, ts_code=current,
                               daily_leader=None if leader is None else leader.ts_code,
                               switch_reason=reason))
    return pd.DataFrame(result, columns=["trade_date", "product", "ts_code", "daily_leader", "switch_reason"])


def build_continuous(bars, selections):
    # Selection observed at the end of a UTC bar takes effect on the next product bar.
    dates = bars[["product", "trade_date"]].drop_duplicates().sort_values(["product", "trade_date"])
    effective = dates.merge(selections, on=["product", "trade_date"], how="left", validate="one_to_one")
    effective["selection_date"] = effective.trade_date
    for name in ("ts_code", "daily_leader", "switch_reason", "selection_date"):
        effective[name] = effective.groupby("product")[name].shift(1)
    effective = effective.dropna(subset=["ts_code"])
    continuous = effective.merge(bars, on=["product", "trade_date", "ts_code"], how="left", validate="one_to_one")
    continuous["missing_selected_bar"] = continuous.close.isna()
    previous = bars[["product", "trade_date", "ts_code", "close"]].rename(columns={"close": "previous_contract_close"})
    previous = previous.merge(dates.assign(next_date=dates.groupby("product").trade_date.shift(-1)), on=["product", "trade_date"])
    previous = previous.drop(columns="trade_date").rename(columns={"next_date": "trade_date"})
    continuous = continuous.merge(previous, on=["product", "trade_date", "ts_code"], how="left", validate="one_to_one")
    continuous["contract_return"] = continuous.close / continuous.previous_contract_close - 1
    continuous.loc[continuous.previous_contract_close == 0, "contract_return"] = np.nan
    continuous = continuous.sort_values(["product", "trade_date"])
    previous_symbol = continuous.groupby("product").ts_code.shift(1)
    continuous["is_roll"] = previous_symbol.notna() & (continuous.ts_code != previous_symbol)
    # OHLC levels are unadjusted; contract_return avoids cross-contract price jumps.
    return continuous


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=RAW_DB)
    parser.add_argument("--output", type=Path, default=OUT_DB)
    parser.add_argument("--roots", nargs="+")
    parser.add_argument("--confirm-days", type=int, default=2)
    parser.add_argument("--force-days", type=int, default=5)
    parser.add_argument("--expiry-file", type=Path)
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error("Use a separate processed database")
    with duckdb.connect(str(args.input), read_only=True) as source:
        if args.roots:
            placeholders = ",".join("?" for _ in args.roots)
            frame = source.execute(f"SELECT * FROM raw_ohlcv_1d WHERE product_root IN ({placeholders})", args.roots).df()
        else:
            frame = source.sql("SELECT * FROM raw_ohlcv_1d").df()
    if frame.empty:
        raise ValueError("No downloaded OHLCV bars found")
    expiry = pd.read_csv(args.expiry_file) if args.expiry_file else None
    bars, rejected = prepare_bars(frame, expiry)
    selections = rollover(bars, args.confirm_days, args.force_days)
    continuous = build_continuous(bars, selections)
    same_day = selections.merge(bars, on=["product", "trade_date", "ts_code"], validate="one_to_one")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(args.output)) as target:
        target.execute("BEGIN TRANSACTION")
        try:
            for name, data in [("rolling_futures_same_day", same_day), ("rolling_futures", continuous)]:
                target.register("incoming", data)
                target.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM incoming")
                target.unregister("incoming")
            target.execute("COMMIT")
        except BaseException:
            target.execute("ROLLBACK")
            raise
    print(f"Excluded {rejected:,} non-outright symbol rows")
    print("Ranking: volume only; confirmation:", args.confirm_days, "days; forward rolls only")
    print("Bars lacking expiry metadata:", int(bars.delist_date.isna().sum()), "(no expiry-forced roll for those bars)")
    print(continuous.groupby("product").agg(rows=("trade_date", "size"), first=("trade_date", "min"), last=("trade_date", "max"), rolls=("is_roll", "sum"), missing=("missing_selected_bar", "sum"), returns=("contract_return", "count")).to_string())
    print("Saved:", args.output)
    print("rolling_futures_same_day = reference-style descriptive selection")
    print("rolling_futures = prior-bar selection + within-contract returns; UTC dates")


if __name__ == "__main__":
    main()
