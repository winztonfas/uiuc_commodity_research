"""Run the unchanged copper ML protocol on additional anchored portfolios.

Run: python sector_ml.py
Defaults: gold, crude, soybeans. Existing copper results are included in the summary
without retuning. Every model uses training-only chronological tuning. Comparing
sectors after seeing test results is exploratory, not independent confirmation of a
selected winning sector. No test-driven changes to models or grids are performed.
"""
import argparse
from pathlib import Path
import duckdb
import pandas as pd
from configs.paths import DATA_DIR, PROJECT_DIR
from copper_ml import main as run_experiment
from plot import plot_sector_ml_summary


def summarize():
    tables = {"test_metrics": [], "incremental_value": [], "selected_parameters": [], "cv_folds": []}
    for anchor in ("copper", "gold", "crude", "soybeans"):
        path = DATA_DIR / f"{anchor}_ml.duckdb"
        if not path.exists():
            continue
        with duckdb.connect(str(path), read_only=True) as conn:
            for name in tables:
                frame = conn.sql(f"SELECT * FROM {name}").df()
                frame["anchor"] = anchor
                tables[name].append(frame)
    combined = {name: pd.concat(parts, ignore_index=True) for name, parts in tables.items()}
    figure = plot_sector_ml_summary(combined["test_metrics"], combined["incremental_value"])
    directory = PROJECT_DIR / "results" / "sector_ml"
    directory.mkdir(parents=True, exist_ok=True)
    for name, frame in combined.items():
        frame.to_csv(directory / f"{name}.csv", index=False)
    metrics = combined["test_metrics"]
    increment = combined["incremental_value"].query("comparison == 'full_vs_domestic'")
    full = metrics.query("feature_set == 'domestic_us_cross'").merge(
        increment[["anchor", "model", "mse_improvement_pct"]], on=["anchor", "model"], validate="one_to_one")
    lines = ["# Anchored-portfolio ML comparison", "",
        "Identical model families, tuning grids, publication-delay assumptions, chronological folds, "
        "and October 2025 holdout boundary. Dates are identical across feature sets within each portfolio; "
        "sample counts can differ across portfolios. Copper results are reused without retuning.", "",
        "Positive MSE reduction means US + cross-market features improve on the domestic-only version. "
        "Positive R² versus zero means the full model also beats forecasting zero. Both comparisons matter.", "",
        "| Portfolio | Model | Test n | MSE reduction vs domestic | R² vs zero | R² vs training mean | Direction |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for row in full.itertuples():
        lines.append(f"| {row.anchor.title()} | {row.model} | {row.n} | {row.mse_improvement_pct:.2f}% | "
                     f"{row.oos_r2_zero:.4f} | {row.oos_r2_train_mean:.4f} | {row.directional_accuracy:.1%} |")
    lines += ["", "All sectors/models are reported. Choosing a presentation case after viewing these test results "
              "is exploratory selection; it does not establish a reliable winning sector. No significance "
              "or causal spillover is claimed. Avoid using the same test outcomes to retune the protocol.",
              "", f"Figure: `{figure}`", ""]
    (directory / "results.md").write_text("\n".join(lines), encoding="utf-8")
    print(full[["anchor", "model", "n", "mse_improvement_pct", "oos_r2_zero", "oos_r2_train_mean"]].to_string(index=False))
    print("Saved", directory, "and", figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchors", nargs="+", choices=["copper", "gold", "crude", "soybeans"],
                        default=["gold", "crude", "soybeans"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    for anchor in dict.fromkeys(args.anchors):
        run_experiment(["--anchor", anchor, "--seed", str(args.seed)])
    summarize()


if __name__ == "__main__":
    main()
