from pathlib import Path
import textwrap
from configs.mapping import us_roots, china_roots
import pandas as  pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


PLOT_SAVE_DIR = Path(__file__).resolve().parent / "plot_save"
PLOT_SAVE_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIRS = {
    "context": PLOT_SAVE_DIR / "context",
    "histogram": PLOT_SAVE_DIR / "histogram",
    "boxplot": PLOT_SAVE_DIR / "boxplot",
    "heatmap": PLOT_SAVE_DIR / "heatmap",
    "feature_heatmap": PLOT_SAVE_DIR / "feature_heatmap",
    "ml": PLOT_SAVE_DIR / "ml",
    "feature_importance": PLOT_SAVE_DIR / "feature_importance",
    "dendrogram": PLOT_SAVE_DIR / "dendrogram",
    "cluster_indexes": PLOT_SAVE_DIR / "cluster_indexes",
    "rolling_volatility": PLOT_SAVE_DIR / "rolling_volatility",
}
for directory in PLOT_DIRS.values():
    directory.mkdir(parents=True, exist_ok=True)



PRODUCT_NAMES = {
    "US": {
        "GC": "Gold", "SI": "Silver", "HG": "Copper", "ALI": "Aluminum",
        "HRC": "Hot-rolled coil", "PL": "Platinum", "PA": "Palladium",
        "CL": "WTI crude", "HO": "ULSD", "RB": "Gasoline", "NG": "Natural gas",
        "ZS": "Soybeans", "ZM": "Soybean meal", "ZL": "Soybean oil",
        "ZC": "Corn", "ZW": "SRW wheat", "KE": "HRW wheat",
        "ZR": "Rough rice", "HE": "Lean hogs",
    },
    "China": {
        "AU": "Gold", "AG": "Silver", "CU": "Copper", "AL": "Aluminum",
        "ZN": "Zinc", "PB": "Lead", "NI": "Nickel", "SN": "Tin",
        "HC": "Hot-rolled coil", "PT": "Platinum", "PD": "Palladium",
        "SC": "Crude oil", "FU": "Fuel oil", "LU": "Low-sulfur fuel oil",
        "BU": "Bitumen", "B": "Soybeans No. 2", "A": "Soybeans No. 1",
        "M": "Soybean meal", "Y": "Soybean oil", "C": "Corn",
        "PM": "Common wheat", "WH": "Strong-gluten wheat",
        "JR": "Japonica rice", "RI": "Early indica rice", "LR": "Late indica rice",
        "LH": "Live hogs",
    },
}


def constituent_caption(cn, us, sector, start, end, width=135):
    key = "Metal" if sector == "Metals" else sector
    lines = []
    for market, frame, mapping in [("US", us, us_roots), ("China", cn, china_roots)]:
        configured = (list(dict.fromkeys(root for roots in mapping.values() for root in roots))
                      if key == "All Sectors" else mapping[key])
        panel = frame.attrs.get("constituent_returns")
        if panel is not None:
            panel = pd.DataFrame(panel)
            valid = panel.loc[panel["trade_date"].between(start, end) & panel["ret"].notna()]
            present = set(valid["product"])
            roots = [root for root in configured if root in present]
            prefix = f"{market} line — products contributing during this window: "
        else:
            roots = configured
            prefix = f"{market} line — configured products: "
        names = "; ".join(f"{root} ({PRODUCT_NAMES[market].get(root, root)})" for root in roots)
        lines.append(textwrap.fill(prefix + names, width=width))
    lines.append("Equal weight among valid daily constituents; membership can vary by date.")
    return "\n".join(lines)


def add_details_panel(fig, cn, us, sector, start, end, event_notes=None):
    key = "Metal" if sector == "Metals" else sector
    lists = {}

    for market, frame, mapping in [("US", us, us_roots), ("China", cn, china_roots)]:
        roots = (list(dict.fromkeys(r for values in mapping.values() for r in values))
                 if key == "All Sectors" else mapping[key])
        records = frame.attrs.get("constituent_returns")
        if records is not None:
            panel = pd.DataFrame(records)
            present = set(panel.loc[panel.trade_date.between(start, end) & panel.ret.notna(), "product"])
            roots = [r for r in roots if r in present]
        lists[market] = [f"{r:<4}  {PRODUCT_NAMES[market].get(r, r)}" for r in roots]

    rows = max(len(lists["US"]), len(lists["China"]), len(event_notes or []), 8)
    footer_inches = 1.0 + rows * 0.19
    width, height = fig.get_size_inches()
    chart_inches = 7.5 if event_notes is not None else 4.5

    fig.set_size_inches(width, chart_inches + footer_inches + 1.0)
    bottom = (footer_inches + 0.65) / fig.get_size_inches()[1]
    fig.subplots_adjust(bottom=bottom, top=0.94, left=0.07, right=0.97)
    
    panel_top = bottom - 0.07
    panel_bottom = 0.065
    height = max(panel_top - panel_bottom, 0.1)
    
    for market, x, color in [("US", 0.07, "#4C72B0"), ("China", 0.32, "#C44E52")]:
        ax = fig.add_axes([x, panel_bottom, 0.23, height])
        ax.axis("off")
        ax.text(0, 1, f"{market.upper()} PRODUCTS", color=color, fontsize=12,
                fontweight="bold", va="top")
        ax.text(0, 0.88, "Product  Commodity", fontsize=9, color="#666666", va="top")
        ax.text(0, 0.79, "\n".join(lists[market]), fontfamily="monospace",
                fontsize=10, linespacing=1.5, va="top")
    
    notes = fig.add_axes([0.59, panel_bottom, 0.38, height])
    notes.axis("off")
    notes.text(0, 1, "EVENT WINDOWS" if event_notes is not None else "READING THE CHART",
               fontsize=12, fontweight="bold", va="top", color="#6F5495")
    
    content = ("\n".join(event_notes) + "\n\nShading: 30 calendar days after each event.\nOverlaps appear darker; events do not establish causality."
               if event_notes is not None else
               "Returns standardized separately for each market.\nCalendar-date comparison; timing is not yet aligned.")
    
    notes.text(0, 0.88, content, fontsize=10, linespacing=1.5, va="top")
    
    fig.text(0.07, 0.025,
             "Products listed contributed within the plotted window; equal-weight membership can vary by date.",
             fontsize=9, color="#666666")


def compare_plot(metal_sector_cn, metal_sector_us, sector="Metals"):
    returns = {
        "China": metal_sector_cn["sector_return"].sort_index(),
        "US": metal_sector_us["sector_return"].sort_index(),
    }

    # Start on the first date with valid returns in both markets.
    common = pd.concat(returns, axis=1).dropna()

    if common.empty:
        raise ValueError("No common valid dates between the markets.")

    start = common.index.min()
    end = common.index.max()

    indices = {}

    # Compound each complete market series before aligning the plot.
    for market, series in returns.items():
        series = series.loc[start:end]

        if series.isna().any():
            raise ValueError(
                f"{market} has missing sector returns within the plot window."
            )

        growth = (1 + series).cumprod()
        indices[market] = growth.div(growth.loc[start]).mul(100)

    events = [
        ("2022-02-24", "Russia invades Ukraine"),
        ("2022-03-16", "Fed announces rate hike"),
        ("2024-09-18", "Fed announces 50 bp cut"),
        ("2024-09-24", "China announces stimulus"),
        ("2025-04-02", "US announces reciprocal tariffs"),
        ("2026-02-28", "Middle East conflict begins"),
        ("2026-04-02", "US announces metals tariff changes"),
        ("2026-09-16", "Fed announces 25 bp hike"),
    ]

    fig, ax = plt.subplots(figsize=(20, 12))
    fig.subplots_adjust(bottom=0.34)

    colors = {"China": "#C44E52", "US": "#4C72B0"}

    for market, series in indices.items():
        ax.plot(
            series.index,
            series,
            label=f"{market} {sector.lower()}",
            color=colors[market],
            linewidth=1.8,
        )

    window_days = 30
    event_notes = []

    for number, (date, label) in enumerate(events, start=1):
        event_start = pd.Timestamp(date)
        event_end = event_start + pd.Timedelta(days=window_days)

        # Clip the shading to the displayed sample.
        shade_start = max(event_start, start)
        shade_end = min(event_end, end)

        if shade_start >= shade_end:
            continue

        ax.axvspan(
            shade_start,
            shade_end,
            color="#9467BD",
            alpha=0.12,
            linewidth=0,
        )

        midpoint = shade_start + (shade_end - shade_start) / 2

        ax.text(
            midpoint,
            0.96 - ((number - 1) % 3) * 0.09,
            str(number),
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=9,
            bbox=dict(
                boxstyle="circle,pad=0.25",
                facecolor="white",
                edgecolor="#777777",
            ),
        )

        event_notes.append(
            f"{number}. {event_start:%Y-%m-%d} — {label}"
        )


    ax.set_title(f"Equal-weight {sector} Sector Indices", fontsize=15)
    ax.set_xlabel("Date")
    ax.set_ylabel("Index (common starting date = 100)")
    ax.axhline(100, color="gray", linewidth=0.8, linestyle="--")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, loc="upper left")

    fig.autofmt_xdate()
    fig.subplots_adjust(bottom=0.30)

    add_details_panel(fig, metal_sector_cn, metal_sector_us, sector,
                      start, end, event_notes=event_notes)

    fig.savefig(
        PLOT_DIRS["context"] / f"{sector.lower()}_sector_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)
    return indices


def zoomin_sector(metal_sector_cn, metal_sector_us, sector="Metals",
                  start="2024-03-01", end="2024-06-30"):
    returns = pd.concat(
        [
            metal_sector_us["sector_return"].rename("US"),
            metal_sector_cn["sector_return"].rename("China"),
        ],
        axis=1,
    ).sort_index()

    # Pick a window to inspect; change these dates as needed.
    window = returns.loc[start:end]
    if window.empty:
        raise ValueError(f"No {sector} returns in the zoom window.")

    # Scale each series by its own volatility within this EDA window.
    standardized = (window - window.mean()) / window.std()

    fig, ax = plt.subplots(figsize=(14, 9))

    for market, color in [("US", "#4C72B0"), ("China", "#C44E52")]:
        series = standardized[market].dropna()

        ax.plot(
            series.index,
            series,
            color=color,
            label=market,
            marker="o",
            markersize=3,
        )

    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_title(f"{sector} Sector Return Shocks — Calendar-date Comparison")
    ax.set_ylabel("Standardized daily return")
    ax.set_xlabel("Date")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)

    fig.autofmt_xdate()
    add_details_panel(fig, metal_sector_cn, metal_sector_us, sector,
                      pd.Timestamp(start), pd.Timestamp(end))

    fig.savefig(
        PLOT_DIRS["context"] / f"{sector.lower()}_sector_zoomin.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)
    return standardized


def zoomin_metal(metal_sector_us, metal_sector_cn):
    """Compatibility wrapper retaining the original US-first argument order."""
    return zoomin_sector(metal_sector_cn, metal_sector_us, sector="Metals")


# histogram of rets for sectors
def histogram(sector_cn, sector_us, sector):
    returns = {
        "China": sector_cn["sector_return"].dropna() * 100,
        "US": sector_us["sector_return"].dropna() * 100,
    }

    pooled = pd.concat(list(returns.values()))
    bins = np.histogram_bin_edges(pooled, bins=40)

    fig, ax = plt.subplots(figsize=(10, 6))

    for market, color in [("China", "#C44E52"), ("US", "#4C72B0")]:
        values = returns[market]

        sns.histplot(
            x=values,
            bins=bins,
            stat="density",
            element="step",
            fill=True,
            alpha=0.25,
            color=color,
            label=f"{market} (n={len(values):,})",
            ax=ax,
        )

    ax.axvline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_title(f"{sector}: Daily Sector Return Distributions")
    ax.set_xlabel("Daily sector return (%)")
    ax.set_ylabel("Density")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)

    fig.tight_layout()
    fig.savefig(
        PLOT_DIRS["histogram"] / f"{sector.lower()}_return_histogram.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

# boxplot
def sector_return_boxplot(sectors):
    """Compare available daily returns, with full tails and sample sizes."""
    from matplotlib.patches import Patch
    import numpy as np

    fig, ax = plt.subplots(figsize=(12, 7))
    colors = {"China": "#C44E52", "US": "#4C72B0"}

    labels = []
    
    for index, (sector, (cn, us)) in enumerate(sectors.items()):
        
        counts = {}
        
        for market, frame, offset in [("China", cn, -0.18), ("US", us, 0.18)]:
            values = frame["sector_return"].dropna().to_numpy(dtype=float) * 100
            
            if not len(values) or not np.isfinite(values).all():
                raise ValueError(f"{sector}/{market}: returns must be nonempty and finite")
            
            counts[market] = len(values)
            
            ax.boxplot(
                [values], positions=[index + offset], widths=0.28,
                patch_artist=True, manage_ticks=False, whis=1.5,
                showfliers=True,
                
                boxprops=dict(facecolor=colors[market], edgecolor=colors[market], alpha=0.4),
                medianprops=dict(color="#222222", linewidth=1.5),
                whiskerprops=dict(color=colors[market]),
                capprops=dict(color=colors[market]),
                flierprops=dict(marker="o", markersize=3, alpha=0.3,
                                markerfacecolor=colors[market], markeredgecolor="none"),
            )
        labels.append(f"{sector}\nChina n={counts['China']:,} | US n={counts['US']:,}")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Daily sector return (%)")
    ax.set_title("Daily Sector Return Distributions: China vs US", fontsize=15, pad=14)
    
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.9, zorder=0)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    ax.legend(handles=[Patch(facecolor=color, alpha=0.4, label=market)
                       for market, color in colors.items()], frameon=False)
    
    fig.text(0.08, 0.025,
             "Box: 25th–75th percentiles; line: median; whiskers: 1.5 × IQR; dots: tail observations.\n"
             "Available observations in each market; changing constituent membership; calendar timing is not aligned.",
             fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    output = PLOT_DIRS["boxplot"] / "sector_return_boxplot.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    return output


def rolling_volatility_plot(sectors, window=20):
    """One panel per sector; rolling daily-return SD, not annualized."""
    import numpy as np
    import matplotlib.dates as mdates
    from matplotlib.lines import Line2D

    if window < 2 or not sectors:
        raise ValueError("Provide sectors and a window of at least two observations")
    rows = (len(sectors) + 1) // 2
   
    fig, axes = plt.subplots(rows, 2, figsize=(15, 4 * rows + 1),
                             sharex=True, sharey=True, squeeze=False)
    colors = {"China": "#C44E52", "US": "#4C72B0"}
    maximum_volatility = 0.0
   
    for ax, (sector, (cn, us)) in zip(axes.flat, sectors.items()):
        
        for market, frame in [("China", cn), ("US", us)]:
            returns = frame["sector_return"].sort_index().astype(float)
        
            if returns.index.has_duplicates:
                raise ValueError(f"{sector}/{market}: duplicate dates")
        
            if np.isinf(returns.to_numpy()).any():
                raise ValueError(f"{sector}/{market}: infinite returns")
        
            volatility = returns.rolling(window, min_periods=window).std(ddof=1) * 100
        
            valid = volatility.dropna()
        
            if not valid.empty:
                maximum_volatility = max(maximum_volatility, float(valid.max()))
            ax.plot(volatility.index, volatility, color=colors[market], linewidth=1.3)
        
        ax.set_title(sector, fontsize=12, fontweight="bold")
        ax.set_ylabel("Rolling daily volatility (%)")
        ax.set_xlabel("Date")
        
        ax.grid(alpha=0.2)
        
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    
    # Set shared limits once, after all sector curves have been plotted.
    axes.flat[0].set_ylim(0, maximum_volatility * 1.08 if maximum_volatility else 1)
    
    for ax in list(axes.flat)[len(sectors):]:
        ax.set_visible(False)
    
    fig.suptitle(f"{window}-Observation Rolling Volatility: China vs US", fontsize=16, y=0.98)
    
    fig.legend(handles=[Line2D([0], [0], color=color, linewidth=2, label=market)
                        for market, color in colors.items()],
               loc="upper center", bbox_to_anchor=(0.5, 0.945), ncol=2, frameon=False)
    
    fig.text(0.07, 0.025,
             f"Sample standard deviation of the latest {window} observations in each market; not annualized.\n"
             "Windows follow each market's own observation calendar. Missing returns are not filled; membership can vary.",
             fontsize=9, color="#555555")
    
    fig.tight_layout(rect=(0, 0.09, 1, 0.91))
    
    output = PLOT_DIRS["rolling_volatility"] / f"sector_rolling_volatility_{window}.png"
    
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    return output


def market_correlation_heatmap(frame, market, sector_roots, min_observations=60):
    """Retrospective within-market Pearson correlations, ordered by sector."""
    
    if market not in PRODUCT_NAMES:
        raise ValueError("market must be China or US")
    
    expected = list(dict.fromkeys(root for roots in sector_roots.values() for root in roots))
    selected = frame.loc[frame["product"].isin(expected), ["trade_date", "product", "ret"]].copy()
    
    if selected.duplicated(["trade_date", "product"]).any():
        raise ValueError("Expected one selected contract return per product/date")
    
    panel = selected.pivot(index="trade_date", columns="product", values="ret").sort_index()
    
    if np.isinf(panel.to_numpy(dtype=float)).any():
        raise ValueError("Infinite returns found; check previous closes")
    
    roots = [root for root in expected if root in panel.columns]
    
    if len(roots) < 2:
        raise ValueError("Need at least two products")
    
    panel = panel.reindex(columns=roots)
    valid = panel.notna().astype("int64")
    counts = valid.T @ valid
    
    correlations = panel.corr(method="pearson", min_periods=min_observations)
    
    output_dir = PLOT_DIRS["heatmap"]
    stem = market.lower() + "_product_return_correlation"
    correlations.to_csv(output_dir / f"{stem}.csv")
    counts.to_csv(output_dir / f"{stem}_counts.csv")

    size = max(10, len(roots) * 0.48 + 4)
    fig, ax = plt.subplots(figsize=(size, size - 1))
    ax.set_facecolor("#EEEEEE")
    
    labels = [f"{r} | {PRODUCT_NAMES[market].get(r, r)}" for r in roots]
    
    sns.heatmap(correlations, ax=ax, vmin=-1, vmax=1, center=0,
                cmap="RdBu_r", square=True, annot=True, fmt=".2f",
                annot_kws={"fontsize": 7 if len(roots) > 20 else 8},
                linewidths=0.3, linecolor="white",
                xticklabels=labels, yticklabels=labels,
                cbar_kws={"label": "Pearson return correlation", "shrink": 0.75})
    position = 0
    
    group_colors = {"Metal": "#8064A2", "Energy": "#B17631", "Agriculture": "#3C8D5A"}
    boundaries = []
    
    for sector, members in sector_roots.items():
        group = [r for r in roots if r in members]
    
        for i in range(position, position + len(group)):
            ax.get_xticklabels()[i].set_color(group_colors.get(sector, "#333333"))
            ax.get_yticklabels()[i].set_color(group_colors.get(sector, "#333333"))
        position += len(group)
    
        if 0 < position < len(roots):
            boundaries.append(position)
    
    for boundary in boundaries:
        ax.axhline(boundary, color="#333333", linewidth=1.5)
        ax.axvline(boundary, color="#333333", linewidth=1.5)
    
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=9)
    ax.set_yticklabels(labels, rotation=0, fontsize=9)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(f"{market}: Commodity Return Correlations", fontsize=16, pad=24)
    
    fig.text(0.02, 0.015,
             f"Full-sample EDA: {panel.index.min():%Y-%m-%d} to {panel.index.max():%Y-%m-%d}. "
             f"Pairwise available returns; minimum {min_observations} paired observations.\n"
             "Sector order: Metals (purple), Energy (brown), Agriculture (green). "
             "Gray cells: insufficient observations or undefined correlation.\n"
             "Pairwise sample sizes differ; counts saved separately. This is retrospective, not a forecasting input.",
             fontsize=9, color="#555555")
    
    fig.tight_layout(rect=(0, 0.075, 1, 1))
    output = output_dir / f"{stem}.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output

def plot_dendrograms(cn_d_corr, cn_link, us_d_corr, us_link,
                     distance_threshold=0.43, cutoff=None, show=False):
    """Plot both market hierarchies, with labels in linkage input order."""
   
    from scipy.cluster.hierarchy import dendrogram, is_valid_linkage

    if distance_threshold < 0:
        raise ValueError("distance_threshold must be nonnegative")
    fig, axes = plt.subplots(1, 2, figsize=(20, 9), sharey=True)
    
    try:
        for ax, distances, link, market in [
            (axes[0], cn_d_corr, cn_link, "China"),
            (axes[1], us_d_corr, us_link, "US"),
        ]:
    
            if len(distances.index) < 2 or len(link) != len(distances.index) - 1:
                raise ValueError(f"{market}: linkage must match at least two product labels")
    
            is_valid_linkage(np.asarray(link), throw=True)
    
            labels = [f"{r} | {PRODUCT_NAMES[market].get(r, r)}" for r in distances.index]
    
            dendrogram(link, labels=labels, leaf_rotation=90, leaf_font_size=9,
                       color_threshold=distance_threshold, ax=ax)
    
            ax.axhline(distance_threshold, color="gray", linestyle="--", linewidth=1,
                       label=f"Distance cut = {distance_threshold:g}")
            ax.set_title(market, fontsize=14, fontweight="bold")
            ax.set_xlabel("Commodity product")
            ax.set_ylabel("Correlation distance")
            ax.grid(axis="y", alpha=0.2)
            ax.legend(frameon=False, loc="upper right")
    
        title = "Hierarchical Commodity Clustering"
        suffix = ""
    
        if cutoff is not None:
            date = pd.Timestamp(cutoff)
            title += f" — cutoff {date:%Y-%m-%d}"
            suffix = f"_{date:%Y-%m-%d}"
    
        fig.suptitle(title, fontsize=17)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        output = PLOT_DIRS["dendrogram"] / f"market_dendrograms{suffix}.png"
        fig.savefig(output, dpi=300, bbox_inches="tight")
    
        if show:
            plt.show()
        return output
    
    finally:
        plt.close(fig)


def plot_cluster_indexes(database=None, show=False):
    """Two market panels of strictly chained, anchor-identified HRP indices."""
    import duckdb
    import matplotlib.dates as mdates
    from matplotlib.lines import Line2D
    from configs.paths import DATA_DIR

    database = Path(database) if database is not None else DATA_DIR / "cluster_indexes.duckdb"
    
    with duckdb.connect(str(database), read_only=True) as conn:
        frame = conn.sql("SELECT * FROM anchor_indexes ORDER BY market, anchor, trade_date").df()
    
    if frame.empty:
        raise ValueError("No anchor indices available; run cluster_index.py first")
    colors = {"gold": "#C49A00", "copper": "#C46A38", "crude": "#385B8E",
              "soybeans": "#459663", "wheat": "#9368A8"}
    names = {"gold": "Gold", "copper": "Copper", "crude": "Crude oil",
             "soybeans": "Soybeans", "wheat": "Wheat"}
    fig, axes = plt.subplots(1, 2, figsize=(17, 8), sharex=True, sharey=True)
    notes = []
    
    try:
        for ax, market in zip(axes, ["China", "US"]):
            market_frame = frame.loc[frame.market.eq(market)]
            
            for anchor, color in colors.items():
                series = market_frame.loc[market_frame.anchor.eq(anchor)].sort_values("trade_date")
                count = series.index_value.notna().sum()
                
                if not count:
                    notes.append(f"{market}: {names[anchor]} unavailable")
                    continue
                product = series.anchor_product.iloc[0]
                # Keep NaNs in the plotted series: never bridge unknown portfolio returns.
                ax.plot(series.trade_date, series.index_value, color=color, linewidth=1.7,
                        marker="o" if count < 5 else None, markersize=4,
                        label=f"{names[anchor]} ({product})")
                
                for column, marker, flag_color in [
                    ("fallback_start", "v", "#D62728"),
                    ("effective_constituents_changed", "v", "#1672D4"),
                ]:
                    flags = series.loc[series[column] & series.index_value.notna()]
                    ax.scatter(flags.trade_date, flags.index_value, marker=marker,
                               s=24, color=flag_color, edgecolor="white", linewidth=0.5, zorder=5)
                valid_returns = series.cluster_return.notna().sum()
                
                if count < valid_returns:
                    notes.append(f"{market} {names[anchor]}: {count:,} chained observations; later levels undefined after gaps")
           
            ax.axhline(100, color="gray", linestyle="--", linewidth=0.8)
            ax.set_title(market, fontsize=15, fontweight="bold")
            ax.set_xlabel("Date")
            ax.set_ylabel("HRP cluster index (initial capital = 100)")
            ax.grid(alpha=0.2)
            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            handles, labels = ax.get_legend_handles_labels()
            handles += [Line2D([], [], marker="v", markersize=5, color="#D62728", linestyle="None", label="Singleton fallback starts"),
                        Line2D([], [], marker="v", markersize=5, color="#1672D4", linestyle="None", label="Effective constituents change")]
            ax.legend(handles=handles, frameon=False, fontsize=9, loc="upper left")
        
        fig.suptitle("Adaptive Anchor-Cluster Indices: China and US", fontsize=18, y=0.98)
        fig.text(0.07, 0.025,
            "Monthly pure HRP; singleton anchors retain previous eligible members with newly estimated weights.\n"
            "US copper excludes missing members and renormalizes weights on affected days. Other unresolved return gaps remain undefined.\n"
            + "\n".join(notes), fontsize=8.5, color="#555555", va="bottom")
        fig.tight_layout(rect=(0, 0.23, 1, 0.95))
        output = PLOT_DIRS["cluster_indexes"] / "anchor_cluster_indexes.png"
        fig.savefig(output, dpi=300, bbox_inches="tight")
        
        if show:
            plt.show()
        
        return output
    
    finally:
        plt.close(fig)


def feature_correlation_heatmap(frame, title, filename, columns=None,
                                min_observations=60, method="spearman", show=False):
    """Exploratory feature redundancy; return figure path, correlations and counts.

    Caller supplies a single portfolio's observations, never pooled sector rows.
    Constant/all-missing features are omitted from the figure, not from the dataset.
    Pairwise sample sizes are returned for audit. Full-history EDA must not be used
    as automatic feature selection for a later held-out test.
    """
    if method not in ("pearson", "spearman"):
        raise ValueError("method must be pearson or spearman")
    panel = frame[columns] if columns is not None else frame.select_dtypes(include="number")
    panel = panel.replace([np.inf, -np.inf], np.nan)
    keep = panel.count().ge(min_observations) & panel.nunique().gt(1)
    panel = panel.loc[:, keep]
    if panel.shape[1] < 2:
        return None, pd.DataFrame(), pd.DataFrame()
    observed = panel.notna().astype("int64")
    counts = observed.T @ observed
    correlations = panel.corr(method=method, min_periods=min_observations)
    n = len(panel.columns)
    fig, ax = plt.subplots(figsize=(max(11, n * 0.35 + 4), max(10, n * 0.35 + 3)))
    try:
        ax.set_facecolor("#EEEEEE")
        sns.heatmap(correlations, ax=ax, cmap="RdBu_r", vmin=-1, vmax=1,
                    center=0, square=True, annot=n <= 16, fmt=".2f",
                    linewidths=0.2, cbar_kws={"label": f"{method.title()} correlation", "shrink": 0.75})
        ax.set_title(title, fontsize=15, pad=18)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=60, ha="right", fontsize=8)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
        fig.text(0.02, 0.015,
            f"Full-history exploratory analysis; pairwise available observations, minimum n={min_observations}. "
            "Gray cells have insufficient overlap.\n"
            "Constant/all-missing columns omitted from figure. Correlation is not predictive importance; "
            "perform learned feature selection on training data only.", fontsize=9, color="#555555")
        fig.tight_layout(rect=(0, 0.06, 1, 1))
        output = PLOT_DIRS["feature_heatmap"] / Path(filename).name
        fig.savefig(output, dpi=200, bbox_inches="tight")
        if show:
            plt.show()
        return output, correlations, counts
    finally:
        plt.close(fig)


def plot_ml_comparison(metrics, incremental, predictions, anchor="copper", show=False):
    """Presentation summary of a fixed held-out forecast experiment; no trading P&L."""
    models = ["Ridge", "Random forest", "Gradient boosting", "Ensemble"]
    scopes = [("domestic", "China only", "#909090"),
              ("domestic_us", "China + US", "#4C72B0"),
              ("domestic_us_cross", "China + US + cross", "#55A868")]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    try:
        ax = axes[0, 0]
        x = np.arange(len(models))
        for i, (scope, label, color) in enumerate(scopes):
            values = metrics.loc[metrics.feature_set.eq(scope)].set_index("model").reindex(models)
            ax.bar(x + (i - 1) * 0.24, values.rmse * 10000, width=0.23, label=label, color=color)
        zero = metrics.loc[metrics.model.eq("Zero return"), "rmse"].iloc[0] * 10000
        ax.axhline(zero, color="black", linestyle="--", linewidth=1, label="Zero forecast")
        ax.set_xticks(x, models, fontsize=9)
        ax.set_ylabel("Test RMSE (basis points; lower is better)")
        ax.set_title("Held-out forecast error")
        ax.legend(frameon=False, fontsize=8)
        ax.grid(axis="y", alpha=0.2)
        ax = axes[0, 1]
        values = incremental.loc[incremental.comparison.eq("full_vs_domestic")].set_index("model").reindex(models)
        ax.barh(models, values.mse_improvement_pct,
                color=["#55A868" if v >= 0 else "#C44E52" for v in values.mse_improvement_pct])
        ax.axvline(0, color="gray", linewidth=0.8)
        ax.set_xlabel("MSE reduction from adding US + cross features (%)")
        ax.set_title("Incremental predictive information")
        ax.grid(axis="x", alpha=0.2)
        ax = axes[1, 0]
        ensemble = predictions.loc[predictions.model.eq("Ensemble")]
        wide = ensemble.pivot(index="target_trade_date", columns="feature_set", values="prediction").sort_index()
        actual = ensemble.drop_duplicates("target_trade_date").set_index("target_trade_date").target_return.reindex(wide.index)
        domestic_loss = (actual - wide.domestic).pow(2)
        full_loss = (actual - wide.domestic_us_cross).pow(2)
        improvement = 100 * (domestic_loss.cumsum() - full_loss.cumsum()) / domestic_loss.cumsum()
        ax.plot(wide.index, improvement, color="#4C72B0", linewidth=1.6)
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.set_ylabel("Cumulative MSE reduction (%)")
        ax.set_title("Ensemble: full features versus China only")
        ax.tick_params(axis="x", rotation=25)
        ax.grid(alpha=0.2)
        ax = axes[1, 1]
        ax.axis("off")
        full = metrics.loc[metrics.feature_set.eq("domestic_us_cross")].set_index("model").reindex(models)
        cells = [[name, f"{row.rmse*10000:.1f}", f"{row.directional_accuracy*100:.1f}%", f"{row.oos_r2_zero:.3f}"]
                 for name, row in full.iterrows()]
        table = ax.table(cellText=cells, colLabels=["Full-feature model", "RMSE (bp)", "Direction", "R² vs zero"],
                         cellLoc="center", loc="center", colWidths=[0.37, 0.22, 0.20, 0.21])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        for (r, c), cell in table.get_celld().items():
            cell.set_edgecolor("white")
            cell.set_facecolor("#DDE7F0" if r == 0 else "#F2F4F6" if r % 2 else "white")
        ax.set_title("Full-feature test results", pad=12)
        count = int(full.n.iloc[0])
        fig.suptitle(f"{anchor.title()} Cluster Portfolio: Out-of-Sample Prediction", fontsize=18)
        fig.text(0.04, 0.02,
            f"Same {count} test observations for every model and feature set. Three chronological tuning folds; "
            "final models fixed throughout the test.\n"
            "Equal-average ensemble; negative improvement means US features hurt. "
            "Publication times are assumed (China +24h; US UTC bar end +5min). No causal or trading-profit claim.",
            fontsize=9, color="#555555")
        fig.tight_layout(rect=(0, 0.08, 1, 0.95))
        output = PLOT_DIRS["ml"] / f"{anchor}_ml_comparison.png"
        fig.savefig(output, dpi=220, bbox_inches="tight")
        if show:
            plt.show()
        return output
    finally:
        plt.close(fig)


def plot_sector_ml_summary(metrics, incremental, show=False):
    """Compare scale-free error reductions; identical dates within each portfolio."""
    order = [name for name in ["copper", "gold", "crude", "soybeans"] if name in metrics.anchor.unique()]
    models = ["Ridge", "Random forest", "Gradient boosting", "Ensemble"]
    improvements = incremental.loc[incremental.comparison.eq("full_vs_domestic")].pivot(
        index="anchor", columns="model", values="mse_improvement_pct").reindex(index=order, columns=models)
    skill = metrics.loc[metrics.feature_set.eq("domestic_us_cross")].pivot(
        index="anchor", columns="model", values="oos_r2_zero").reindex(index=order, columns=models) * 100
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    try:
        for ax, values, title in [(axes[0], improvements, "MSE reduction: full features vs China only (%)"),
                                  (axes[1], skill, "MSE reduction: full model vs zero forecast (%)")]:
            bound = max(1., float(values.abs().max().max()))
            sns.heatmap(values, ax=ax, annot=True, fmt=".2f", center=0,
                        vmin=-bound, vmax=bound, cmap="RdBu", linewidths=1, linecolor="white",
                        cbar_kws={"label": "Positive = lower forecast error", "shrink": 0.7})
            ax.set_title(title, fontsize=12, pad=14)
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right")
            ax.set_yticklabels([x.title() for x in order], rotation=0)
        fig.suptitle("US Predictive Information Across Anchored Portfolios", fontsize=17)
        fig.text(0.03, 0.02,
            "Same protocol and tuning grids across portfolios; same test dates within each portfolio. Test begins October 2025.\n"
            "Left: incremental US + cross-market value. Right: whether the full model beats a simple zero forecast. "
            "Sector comparisons are exploratory; no statistical significance established.", fontsize=9, color="#555555")
        fig.tight_layout(rect=(0, 0.12, 1, 0.92))
        output = PLOT_DIRS["ml"] / "sector_ml_summary.png"
        fig.savefig(output, dpi=220, bbox_inches="tight")
        if show:
            plt.show()
        return output
    finally:
        plt.close(fig)


def plot_feature_importance(importance, anchor="copper", top_n=10, show=False):
    """CV block-permutation effects; error bars are repeat spread, not confidence intervals."""
    subset = importance.loc[importance.anchor.eq(anchor) & importance.kind.eq("individual")]
    models = ["Ridge", "Random forest", "Gradient boosting", "Ensemble"]
    fig, axes = plt.subplots(2, 2, figsize=(17, 11))
    colors = {"cn": "#4C72B0", "us": "#DD8452", "cross": "#55A868"}
    try:
        for ax, model in zip(axes.flat, models):
            group = subset.loc[subset.model.eq(model)].copy()
            group = group.loc[group.importance_mean_pct.abs().nlargest(top_n).index]
            group = group.sort_values("importance_mean_pct")
            bars = [colors[name.split("_")[0]] for name in group.feature]
            ax.barh(group.feature, group.importance_mean_pct, xerr=group.permutation_std_pct,
                    color=bars, error_kw={"ecolor": "#555555", "elinewidth": 0.8, "capsize": 2})
            ax.axvline(0, color="gray", linewidth=0.8)
            ax.set_title(model, fontsize=13)
            ax.set_xlabel("Validation MSE increase when shuffled (%)")
            ax.tick_params(axis="y", labelsize=8)
            ax.grid(axis="x", alpha=0.2)
        from matplotlib.patches import Patch
        fig.legend(handles=[Patch(color=color, label=label) for label, color in
            [("Chinese features", colors["cn"]), ("US features", colors["us"]),
             ("Cross-market features", colors["cross"])]], loc="upper center", ncol=3,
             bbox_to_anchor=(0.5, 0.94), frameon=False)
        fig.suptitle(f"{anchor.title()}: Feature Importance on Chronological Validation Folds", fontsize=17)
        fig.text(0.03, 0.02,
            "Top features by absolute effect; positive = shuffling hurts predictions, negative = shuffling improves them.\n"
            "Five-observation blocks shuffled within validation folds; correlated features can share importance. "
            "Error bars show permutation-repeat spread, not confidence intervals.\n"
            "Selected CV hyperparameters reused; diagnostic rankings are not independent performance estimates. "
            "Final test outcomes are not used; no causal interpretation.", fontsize=9, color="#555555")
        fig.tight_layout(rect=(0, 0.10, 1, 0.90))
        output = PLOT_DIRS["feature_importance"] / f"{anchor}_feature_importance.png"
        fig.savefig(output, dpi=220, bbox_inches="tight")
        if show:
            plt.show()
        return output
    finally:
        plt.close(fig)


def plot_group_importance(importance, show=False):
    """Joint permutation importance avoids summing correlated individual effects."""
    order = [a for a in ["copper", "gold", "crude", "soybeans"] if a in importance.anchor.unique()]
    names = ["Chinese features", "US features", "Cross-market features", "US + cross-market"]
    data = importance.loc[importance.model.eq("Ensemble") & importance.kind.eq("group")].pivot(
        index="anchor", columns="feature", values="importance_mean_pct").reindex(index=order, columns=names)
    bound = max(1., float(data.abs().max().max()))
    fig, ax = plt.subplots(figsize=(11, 6))
    try:
        sns.heatmap(data, ax=ax, annot=True, fmt=".2f", center=0, vmin=-bound, vmax=bound,
                    cmap="RdBu", linewidths=1, linecolor="white",
                    cbar_kws={"label": "Validation MSE increase after joint shuffling (%)"})
        ax.set_title("Ensemble: Feature-Group Importance", fontsize=16, pad=18)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_yticklabels([a.title() for a in order], rotation=0)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")
        fig.text(0.03, 0.02,
            "Positive = the fitted model benefits from that group; negative = shuffling improves validation error.\n"
            "Columns within each group share one block permutation. Groups overlap and are not additive. "
            "Model diagnostic, not causal evidence or a replacement for feature-group ablation.", fontsize=9, color="#555555")
        fig.tight_layout(rect=(0, 0.13, 1, 1))
        output = PLOT_DIRS["feature_importance"] / "feature_group_importance.png"
        fig.savefig(output, dpi=220, bbox_inches="tight")
        if show:
            plt.show()
        return output
    finally:
        plt.close(fig)
