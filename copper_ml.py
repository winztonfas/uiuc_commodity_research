"""Anchored-portfolio ML baseline; defaults to copper for backwards compatibility.

Run: python copper_ml.py --anchor copper
Three models, three feature groups, and a fixed equal-average ensemble. Hyperparameter
selection, imputation, missing indicators, variance filtering, and scaling use training
folds only. Test outcomes never choose models, parameters, or ensemble weights.
Results are forecasts of the cutoff-frozen Chinese anchor-cluster return, not a single
copper contract or a trading backtest. Timing assumptions are inherited from the
prediction dataset. No significance or causality is inferred from feature importance.
"""
import argparse
import hashlib
import json
import platform
from pathlib import Path
import duckdb
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from configs.paths import DATA_DIR, PROJECT_DIR
from plot import plot_ml_comparison


def load_experiment(path, anchor="copper"):
    with duckdb.connect(str(path), read_only=True) as conn:
        data = conn.execute("SELECT * FROM prediction_dataset WHERE anchor=? ORDER BY target_trade_date", [anchor]).df()
        membership = conn.execute("SELECT * FROM cv_membership WHERE anchor=?", [anchor]).df()
        columns = conn.sql("SELECT * FROM feature_columns").df()
        source_config = conn.sql("SELECT * FROM run_config").df()
    data = data.loc[data.model_eligible].copy()
    train = data.loc[data.split.eq("train")].reset_index(drop=True)
    test = data.loc[data.split.eq("test")].reset_index(drop=True)
    if train.empty or test.empty:
        raise ValueError(f"Missing {anchor} training/test observations")
    if train.label_available_at.max() > test.prediction_cutoff.min():
        raise ValueError("Training label not known at first test cutoff")
    if not train.target_trade_date.max() < test.target_trade_date.min():
        raise ValueError("Train/test chronology violated")
    splits, fold_summary = [], []
    for fold, group in membership.groupby("fold", sort=True):
        indexes = {}
        for role in ("train", "validation"):
            dates = group.loc[group.role.eq(role), "target_trade_date"]
            indexes[role] = np.flatnonzero(train.target_trade_date.isin(dates).to_numpy())
            if not len(indexes[role]) or len(indexes[role]) != len(dates):
                raise ValueError(f"Fold {fold}: dates missing or duplicated")
        tr, va = train.iloc[indexes["train"]], train.iloc[indexes["validation"]]
        if tr.label_available_at.max() > va.prediction_cutoff.min():
            raise ValueError(f"Fold {fold}: unknown training labels")
        splits.append((indexes["train"], indexes["validation"]))
        fold_summary.append(dict(fold=int(fold), train_rows=len(tr), validation_rows=len(va),
            train_start=tr.target_trade_date.min(), train_end=tr.target_trade_date.max(),
            validation_start=va.target_trade_date.min(), validation_end=va.target_trade_date.max()))
    if len(splits) < 2:
        raise ValueError("Need at least two chronological folds")
    groups = {name: columns.loc[columns.feature_group.eq(name), "feature"].tolist()
              for name in ("domestic", "us", "cross_market")}
    scopes = {"domestic": groups["domestic"],
              "domestic_us": groups["domestic"] + groups["us"],
              "domestic_us_cross": groups["domestic"] + groups["us"] + groups["cross_market"]}
    for names in scopes.values():
        if len(names) != len(set(names)) or any(name.startswith("target_") for name in names):
            raise ValueError("Invalid predictor allowlist")
    return train, test, splits, scopes, pd.DataFrame(fold_summary), source_config


def model_specs(seed):
    return {
        "Ridge": (Ridge(), {"model__alpha": [0.1, 1., 10., 100., 1000.]}),
        
        "Random forest": (RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=2),
            {"model__max_depth": [3, 6], "model__min_samples_leaf": [10, 30], "model__max_features": [0.7]}),
        
        "Gradient boosting": (GradientBoostingRegressor(random_state=seed, min_samples_leaf=20,
                                                       n_iter_no_change=None),
            {"model__n_estimators": [100, 200], "model__learning_rate": [0.03, 0.1], "model__max_depth": [1, 2]}),
    }


def make_pipeline(model):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
        ("variance", VarianceThreshold()),
        ("scaler", StandardScaler()),
        ("model", model),
    ])


def score_predictions(actual, predicted, mean_baseline):
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    
    mse = mean_squared_error(actual, predicted)
    
    zero_mse = float(np.mean(actual**2))
    mean_mse = float(np.mean((actual - mean_baseline)**2))
    
    correlation = (float(np.corrcoef(actual, predicted)[0, 1])
                   if np.std(predicted) > 0 and np.std(actual) > 0 else np.nan)
    
    return dict(n=len(actual), mse=mse, rmse=float(np.sqrt(mse)),
        mae=mean_absolute_error(actual, predicted), r2=r2_score(actual, predicted),
        oos_r2_zero = 1 - mse/zero_mse if zero_mse > 0 else np.nan,
        oos_r2_train_mean= 1 - mse/mean_mse if mean_mse > 0 else np.nan,
        directional_accuracy=float(np.mean(np.sign(actual)==np.sign(predicted))),
        prediction_correlation=correlation)


def incremental_results(metrics):
    rows = []
    indexed = metrics.set_index(["model", "feature_set"])
    for model in ("Ridge", "Random forest", "Gradient boosting", "Ensemble"):
        for label, small, large in [
            ("us_vs_domestic", "domestic", "domestic_us"),
            ("full_vs_domestic", "domestic", "domestic_us_cross"),
            ("cross_vs_domestic_us", "domestic_us", "domestic_us_cross")]:
            before, after = indexed.loc[(model, small)], indexed.loc[(model, large)]
            rows.append(dict(model=model, comparison=label,
                mse_improvement_pct=100*(before.mse-after.mse)/before.mse,
                rmse_change=after.rmse-before.rmse,
                directional_accuracy_change=after.directional_accuracy-before.directional_accuracy,
                n=int(after.n)))
    return pd.DataFrame(rows)


def markdown_results(metrics, incremental, selected, train, test, figure, anchor="copper"):
    lines = [f"# {anchor.title()} portfolio ML baseline", "",
        f"Training: {train.target_trade_date.min():%Y-%m-%d} to {train.target_trade_date.max():%Y-%m-%d} ({len(train)} observations).",
        f"Held-out test: {test.target_trade_date.min():%Y-%m-%d} to {test.target_trade_date.max():%Y-%m-%d} ({len(test)} observations).", "",
        "Three expanding-window tuning folds. Models are refitted once on all eligible training rows and then frozen for the test.",
        "Preprocessing is fitted within each fold. Features: China only (21), +US (36), +cross-market (39).",
        "Ensemble weights are fixed at one third each. Zero and training-mean forecasts are benchmarks.", "",
        "| Model | Features | Test RMSE (bp) | MAE (bp) | R² vs zero | Direction accuracy |",
        "|---|---|---:|---:|---:|---:|"]
    
    for row in metrics.itertuples():
        lines.append(f"| {row.model} | {row.feature_set} | {row.rmse*10000:.2f} | {row.mae*10000:.2f} | {row.oos_r2_zero:.4f} | {row.directional_accuracy:.1%} |")
    
    lines += ["", "## Incremental out-of-sample value", "",
        "Positive values mean lower MSE after adding US and cross-market features. These paired comparisons use identical test dates.", "",
        "| Model | Full versus domestic MSE reduction |", "|---|---:|"]
    
    for row in incremental.loc[incremental.comparison.eq("full_vs_domestic")].itertuples():
        lines.append(f"| {row.model} | {row.mse_improvement_pct:.2f}% |")
    
    lines += ["", "## Interpretation and limits", "",
        "These are initial exploratory forecasts, not evidence of causal spillovers or a profitable trading strategy. "
        "An improvement over a domestic model can coexist with underperformance versus a zero-return forecast. "
        "The test period must not be reused to tune the model after viewing these results.", "",
        "Availability uses assumed China +24h and US UTC completion +5min delays. Targets freeze weights and contracts at cutoff; "
        "they can differ from the descriptive index on roll/rebalance dates. Rows with missing frozen-contract targets are excluded. "
        "Same evaluation dates are used across all feature sets. Regime analysis and statistical uncertainty remain extensions.", "",
        "Negative R² versus zero means worse MSE than forecasting zero every day. "
        "Direction accuracy for a zero forecast is the share of exactly zero actual returns, not a useful directional classifier.", "",
        "## Selected parameters", ""]
    
    for row in selected.itertuples():
        lines.append(f"- {row.model}, {row.feature_set}: `{row.best_params}`")
    lines += ["", f"Figure: `{figure}`", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DATA_DIR / "prediction_data.duckdb")
    parser.add_argument("--anchor", choices=["copper", "gold", "crude", "soybeans"], default="copper")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.output is None:
        args.output = DATA_DIR / f"{args.anchor}_ml.duckdb"
    if args.output.resolve() == args.input.resolve():
        parser.error("Results cannot overwrite input")
    train, test, splits, scopes, folds, source_config = load_experiment(args.input, args.anchor)
    results_dir = PROJECT_DIR / "results" / f"{args.anchor}_ml"
    model_dir = DATA_DIR / "ml_models" / args.anchor
    results_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    y, y_test = train.target_return, test.target_return
    mean = float(y.mean())
    records, selected, cv_rows, retained, fitted = [], [], [], [], {}
    for scope, names in scopes.items():
        for model_name, (model, grid) in model_specs(args.seed).items():
            print(f"Tuning {args.anchor}: {model_name} / {scope}: {len(names)} candidate features", flush=True)
            search = GridSearchCV(make_pipeline(model), grid, cv=splits,
                scoring="neg_mean_squared_error", refit=True, n_jobs=1, error_score="raise",
                return_train_score=False)
            with threadpool_limits(limits=2):
                search.fit(train[names], y)
                prediction = search.predict(test[names])
            fitted[(scope, model_name)] = prediction
            chosen = search.best_estimator_
            selected.append(dict(model=model_name, feature_set=scope, input_features=len(names),
                best_cv_mse=float(-search.best_score_), best_params=json.dumps(search.best_params_, sort_keys=True)))
            output_names = chosen[:-1].get_feature_names_out()
            retained.extend(dict(model=model_name, feature_set=scope, feature=str(name)) for name in output_names)
            for candidate, params in enumerate(search.cv_results_["params"]):
                for fold in range(len(splits)):
                    cv_rows.append(dict(model=model_name, feature_set=scope, candidate=candidate,
                        fold=fold+1, mse=float(-search.cv_results_[f"split{fold}_test_score"][candidate]),
                        parameters=json.dumps(params, sort_keys=True), selected=candidate==search.best_index_))
            joblib.dump(chosen, model_dir / f"{scope}_{model_name.lower().replace(' ', '_')}.joblib")
            records.append(dict(model=model_name, feature_set=scope, prediction=prediction))
        ensemble = np.mean([fitted[(scope, name)] for name in model_specs(args.seed)], axis=0)
        records.append(dict(model="Ensemble", feature_set=scope, prediction=ensemble))
    records += [dict(model="Zero return", feature_set="baseline", prediction=np.zeros(len(test))),
                dict(model="Training mean", feature_set="baseline", prediction=np.full(len(test), mean))]
    metrics, predictions = [], []
    for record in records:
        metrics.append(dict(model=record["model"], feature_set=record["feature_set"],
            **score_predictions(y_test, record["prediction"], mean)))
        rows = test[["target_trade_date", "prediction_cutoff", "target_return"]].copy()
        rows["model"], rows["feature_set"] = record["model"], record["feature_set"]
        rows["prediction"] = record["prediction"]
        rows["squared_error"] = (rows.target_return-rows.prediction)**2
        predictions.append(rows)
    metrics, predictions, selected = pd.DataFrame(metrics), pd.concat(predictions, ignore_index=True), pd.DataFrame(selected)
    incremental = incremental_results(metrics)
    figure = plot_ml_comparison(metrics, incremental, predictions, anchor=args.anchor)
    dataset_hash = hashlib.sha256(pd.concat([train, test])[list(scopes["domestic_us_cross"]) +
        ["target_trade_date", "target_return", "split"]].to_csv(index=False).encode()).hexdigest()
    config = pd.DataFrame([dict(created_at_utc=pd.Timestamp.now(tz="UTC"), seed=args.seed,
        sklearn_version=sklearn.__version__, python_version=platform.python_version(),
        input=str(args.input), anchor=args.anchor, dataset_sha256=dataset_hash, train_rows=len(train), test_rows=len(test),
        selection="mean chronological CV MSE; no test-based selection", ensemble="equal average of three models",
        refit_rule="one fit on complete training sample; fixed during test", figure=str(figure))])
    tables = {"test_metrics": metrics, "test_predictions": predictions, "incremental_value": incremental,
        "selected_parameters": selected, "cv_scores": pd.DataFrame(cv_rows), "cv_folds": folds,
        "retained_features": pd.DataFrame(retained), "run_config": config, "source_timing_config": source_config}
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
    for name in ("test_metrics", "incremental_value", "selected_parameters", "cv_scores", "cv_folds"):
        tables[name].to_csv(results_dir / f"{name}.csv", index=False)
    (results_dir / "results.md").write_text(markdown_results(metrics, incremental, selected, train, test, figure, args.anchor), encoding="utf-8")
    print(metrics[["model", "feature_set", "n", "rmse", "oos_r2_zero", "directional_accuracy"]].to_string(index=False))
    print("Full versus domestic MSE improvement (%):")
    print(incremental.loc[incremental.comparison.eq("full_vs_domestic"), ["model", "mse_improvement_pct"]].to_string(index=False))
    print("Saved", args.output, "and", results_dir, "and", figure)


if __name__ == "__main__":
    main()
