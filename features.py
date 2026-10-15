"""Pure feature calculations for Chinese and US commodity research.

No database access, model fitting, file output, or execution on import.
Windows count observed rows, not calendar days. Inputs must be sorted through the
helpers below. Features include the current completed observation; they are NOT
implicitly shifted to a forecast date. Use explicit publication/availability times
with align_features_asof. A UTC daily bar start is not its availability time.
"""
import numpy as np
import pandas as pd


def _divide(numerator, denominator):
    """Undefined/nonpositive denominators remain missing, never infinite."""
    return numerator.div(denominator.where(denominator.gt(0))).replace(
        [np.inf, -np.inf], np.nan)


def compounded_return(returns, window):
    """Trailing compounded return; requires every observation in the window."""
    return (1 + returns).rolling(window, min_periods=window).apply(
        np.prod, raw=True) - 1


def return_features(returns):
    """Eight features shared by domestic and US portfolios/products.

    lag0 is the latest COMPLETED observation; lag1 is the previous observation.
    Shock scaling uses 20 observations strictly before the shock.
    """
    r = returns.astype(float)
    out = pd.DataFrame(index=r.index)
    out["return_lag0"] = r
    out["return_lag1"] = r.shift(1)
    out["absolute_return"] = r.abs()
    for window in (5, 20):
        out[f"momentum_{window}"] = compounded_return(r, window)
        out[f"volatility_{window}"] = r.rolling(window, min_periods=window).std()
    out["shock_20"] = _divide(r, r.shift(1).rolling(20, min_periods=20).std())
    return out


def activity_features(values, prefix="volume", window=20):
    """Relative activity, prior-window breakout ratio, and breakout flag.

    All reference statistics exclude the current observation. Zero activity is
    allowed; a zero historical denominator produces NaN. No forward filling.
    """
    values = values.astype(float).where(values.ge(0))
    prior = values.shift(1).rolling(window, min_periods=window)
    ratio = _divide(values, prior.max())
    return pd.DataFrame({
        f"{prefix}_relative_{window}": _divide(values, prior.mean()),
        f"{prefix}_breakout_ratio_{window}": ratio,
        f"{prefix}_breakout_{window}": ratio.gt(1).astype(float).where(ratio.notna()),
    }, index=values.index)


def aggregate_product_activity(contract_panel):
    """Sum activity across outright expiries within product/date, not products.

    Caller supplies outright contracts only, with no duplicate contract/date rows.
    Supports Chinese amount/oi and optional vol/volume; US vol/volume. Amount is
    monetary turnover, not contract volume. OI sums require consistent units and
    comparable snapshot times. An incomplete expiry panel gives incomplete totals.
    Returns a product panel suitable for product_features(activity=...).
    """
    frame = contract_panel.copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    if "volume" not in frame and "vol" in frame:
        frame = frame.rename(columns={"vol": "volume"})
    if "ts_code" in frame and frame.duplicated(["trade_date", "product", "ts_code"]).any():
        raise ValueError("Duplicate contract/date activity records")
    columns = [c for c in ("volume", "amount", "oi") if c in frame]
    if not columns:
        raise ValueError("No volume, amount, or oi columns supplied")
    return frame.groupby(["trade_date", "product"], as_index=False)[columns].sum(min_count=1)


def us_metals_weekday_panel(frame, metals=("GC", "SI", "HG", "ALI")):
    """Fold weekend US metal returns/activity into Monday, product by product.

    Requires ret. Sums volume/turnover; takes destination-day OI. Does not invent
    portfolio OHLC. Destination-day observations and returns are required. Weekend
    rows with explicitly missing returns invalidate that product's combined return.
    interval_bars records the number of source observations, not elapsed hours.
    """
    output = []
    frame = frame.copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    for product, group in frame.groupby("product", sort=False):
        group = group.sort_values("trade_date").copy()
        group["interval_bars"] = 1
        if product not in metals:
            output.append(group)
            continue
        dates = pd.DatetimeIndex(group["trade_date"])
        weekday = dates.dayofweek
        labels = dates + pd.to_timedelta(np.where(weekday >= 5, 7 - weekday, 0), unit="D")
        group["_destination"] = labels
        for date, chunk in group.groupby("_destination", sort=True):
            endpoint = chunk.loc[chunk.trade_date.eq(date)]
            if endpoint.empty:
                continue
            row = endpoint.iloc[-1].copy()
            row["ret"] = ((1 + chunk.ret).prod() - 1) if chunk.ret.notna().all() else np.nan
            row["interval_bars"] = len(chunk)
            for column in ("volume", "amount"):
                if column in chunk:
                    row[column] = chunk[column].sum(min_count=len(chunk))
            output.append(pd.DataFrame([row.drop(labels="_destination")]))
    if not output:
        return frame.iloc[:0].assign(interval_bars=pd.Series(dtype=int))
    return pd.concat(output, ignore_index=True).sort_values(["product", "trade_date"]).reset_index(drop=True)


def product_features(frame, market, activity=None, weekday_metals=True):
    """Features from one selected-contract row per product/date.

    China: trade_date, product, close/pre_close (or ret), amount, oi; volume optional.
    US: trade_date, product, contract_return (or ret), vol; ts_code/is_roll optional.
    Never shift close across contract changes to construct a return. If neither ret
    nor source pre-close/contract_return exists, an explicit error is raised.

    Optional activity is an already aggregated product/date panel from
    aggregate_product_activity. Without it, selected-contract activity is used and
    flagged by activity_scope; OI changes spanning rolls are masked. Volume ratios
    can still reflect roll migration, so full-product activity is preferable.
    """
    if market not in ("China", "US"):
        raise ValueError("market must be China or US")
    data = frame.copy()
    data["trade_date"] = pd.to_datetime(data["trade_date"])
    if data.duplicated(["product", "trade_date"]).any():
        raise ValueError("Expected one selected-contract row per product/date")
    if "volume" not in data and "vol" in data:
        data = data.rename(columns={"vol": "volume"})
    if "ret" not in data:
        if "contract_return" in data:
            data["ret"] = data["contract_return"]
        elif {"close", "pre_close"}.issubset(data.columns):
            data["ret"] = _divide(data["close"] - data["pre_close"], data["pre_close"])
        else:
            raise ValueError("Supply ret, contract_return, or close and pre_close")
    if activity is not None:
        activity = activity.copy()
        activity["trade_date"] = pd.to_datetime(activity["trade_date"])
        if "volume" not in activity and "vol" in activity:
            activity = activity.rename(columns={"vol": "volume"})
        columns = [c for c in ("volume", "amount", "oi") if c in activity]
        data = data.drop(columns=columns, errors="ignore").merge(
            activity[["trade_date", "product"] + columns],
            on=["trade_date", "product"], how="left", validate="one_to_one")
    if market == "US" and weekday_metals:
        data = us_metals_weekday_panel(data)
    else:
        data["interval_bars"] = 1
    outputs = []
    for product, group in data.groupby("product", sort=False):
        g = group.sort_values("trade_date").set_index("trade_date")
        out = return_features(g.ret)
        for column, prefix in (("volume", "volume"), ("amount", "turnover")):
            if column in g:
                out = out.join(activity_features(g[column], prefix))
        rolls = pd.Series(False, index=g.index)
        if "ts_code" in g:
            rolls = g.ts_code.ne(g.ts_code.shift()) & g.ts_code.shift().notna()
        if "is_roll" in g:
            rolls |= g.is_roll.fillna(False).astype(bool)
        if market == "China" and "oi" in g:
            for window in (1, 5):
                change = _divide(g.oi, g.oi.shift(window)) - 1
                if activity is None or "oi" not in activity:
                    change = change.mask(rolls.rolling(window, min_periods=1).max().astype(bool))
                out[f"oi_change_{window}"] = change
            out["return_oi_interaction"] = g.ret * out.oi_change_1
        out["product"], out["market"] = product, market
        out["interval_bars"] = g.interval_bars
        out["roll_flag"] = rolls.astype(int)
        out["activity_scope"] = "product_total" if activity is not None else "selected_contract"
        outputs.append(out.reset_index())
    return pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()


def portfolio_features(anchor_indexes, product_panel=None, contributions=None):
    """Calculate portfolio return features and weighted normalized product features.

    Inputs match anchor_indexes/anchor_contributions from cluster_index.py.
    product_panel is product_features output for either/both markets. Actual daily
    effective weights are used where supplied. Missing active-member features stay
    NaN: this function does not silently renormalize feature coverage. Portfolio
    volatility comes from portfolio returns, not averaged product volatility.
    """
    data = anchor_indexes.copy()
    data["trade_date"] = pd.to_datetime(data["trade_date"])
    keys = ["market", "anchor", "trade_date"]
    if data.duplicated(keys).any():
        raise ValueError("Duplicate portfolio/date rows")
    outputs = []
    for (market, anchor), group in data.groupby(["market", "anchor"], sort=False):
        group = group.sort_values("trade_date").set_index("trade_date")
        out = return_features(group.cluster_return)
        out["market"], out["anchor"] = market, anchor
        for column in ("fallback", "daily_exclusion", "effective_constituents_changed", "available_weight"):
            if column in group:
                out[column] = group[column]
        outputs.append(out.reset_index())
    result = pd.concat(outputs, ignore_index=True)
    if product_panel is None or contributions is None:
        return result
    products, exposure = product_panel.copy(), contributions.copy()
    for table in (products, exposure):
        table["trade_date"] = pd.to_datetime(table["trade_date"])
    normalized = [c for c in products if c.startswith(("volume_", "turnover_", "oi_"))
                  or c == "return_oi_interaction"]
    extra = [c for c in ("interval_bars", "roll_flag") if c in products]
    joined = exposure.merge(products[["market", "product", "trade_date"] + normalized + extra],
        on=["market", "product", "trade_date"], how="left", validate="many_to_one")
    weight_column = "effective_weight" if "effective_weight" in joined else "weight"
    joined = joined.loc[joined[weight_column].gt(0)].copy()
    aggregated = []
    for identity, group in joined.groupby(keys, sort=False):
        row = dict(zip(keys, identity))
        weights = group[weight_column]
        for column in normalized:
            row[column] = (group[column] * weights).sum() if group[column].notna().all() else np.nan
        row["positive_return_breadth"] = (group.product_return.gt(0).mean()
            if group.product_return.notna().all() else np.nan)
        row["constituent_count"] = len(group)
        row["largest_weight"] = weights.max()
        for column in extra:
            row[column] = group[column].max() if group[column].notna().all() else np.nan
        aggregated.append(row)
    if aggregated:
        result = result.merge(pd.DataFrame(aggregated), on=keys, how="left", validate="one_to_one")
    if "volume_relative_20" in result:
        result["shock_volume_interaction"] = result.shock_20 * result.volume_relative_20
    return result


def align_features_asof(features, forecast_grid, prefix, max_age):
    """Backward availability-time join for one market, with explicit staleness cap.

    features: anchor, available_at (timezone-aware), and feature columns.
    forecast_grid: anchor, prediction_cutoff (timezone-aware); optional target_start
    must be at or AFTER prediction_cutoff. Caller supplies genuine bar-completion
    and publication times, including Chinese OI publication delays. max_age is a
    pandas Timedelta-compatible value. Exact-cutoff observations are allowed.
    This does not infer exchange sessions, shift calendar rows, or create targets.
    """
    left, right = forecast_grid.copy(), features.copy()
    for table, column in ((left, "prediction_cutoff"), (right, "available_at")):
        times = pd.to_datetime(table[column])
        if times.dt.tz is None:
            raise ValueError(f"{column} must be timezone-aware")
        table[column] = times.dt.tz_convert("UTC").astype("datetime64[ns, UTC]")
    if "target_start" in left:
        target = pd.to_datetime(left.target_start)
        if target.dt.tz is None or not (target >= left.prediction_cutoff).all():
            raise ValueError("Timezone-aware target_start must be at or after prediction_cutoff")
    if right.duplicated(["anchor", "available_at"]).any():
        raise ValueError("Duplicate anchor/availability timestamps")
    if pd.Timedelta(max_age) < pd.Timedelta(0):
        raise ValueError("max_age must be nonnegative")
    right = right.rename(columns={c: f"{prefix}_{c}" for c in right if c != "anchor"})
    timestamp = f"{prefix}_available_at"
    result = pd.merge_asof(left.sort_values("prediction_cutoff"), right.sort_values(timestamp),
        by="anchor", left_on="prediction_cutoff", right_on=timestamp,
        direction="backward", tolerance=pd.Timedelta(max_age))
    result[f"{prefix}_age_hours"] = (result.prediction_cutoff - result[timestamp]).dt.total_seconds() / 3600
    return result


def cross_market_features(china_returns, us_returns, window=60, min_observations=None):
    """Features on explicitly aligned, available return observations.

    Supply equal-index Series on one observation grid, with comparable intervals.
    No automatic calendar intersection, forward filling, or timezone assumptions.
    Optional min_observations allows incomplete paired windows without filling.
    Sensitivity is rolling OLS slope of China returns on US returns (not causality).
    """
    if not china_returns.index.equals(us_returns.index):
        raise ValueError("Align market return intervals explicitly before this calculation")
    minimum = window if min_observations is None else min_observations
    if not 2 <= minimum <= window:
        raise ValueError("Require 2 <= min_observations <= window")
    cn, us = china_returns.astype(float), us_returns.astype(float)
    paired = cn.notna() & us.notna()
    cn, us = cn.where(paired), us.where(paired)
    out = pd.DataFrame(index=cn.index)
    out[f"correlation_{window}"] = cn.rolling(window, min_periods=minimum).corr(us)
    covariance = cn.rolling(window, min_periods=minimum).cov(us)
    out[f"sensitivity_{window}"] = _divide(covariance, us.rolling(window, min_periods=minimum).var())
    cn_scale = cn.shift(1).rolling(20, min_periods=20).std() * np.sqrt(20)
    us_scale = us.shift(1).rolling(20, min_periods=20).std() * np.sqrt(20)
    out["momentum_divergence_20"] = (_divide(compounded_return(cn, 20), cn_scale)
        - _divide(compounded_return(us, 20), us_scale))
    return out
