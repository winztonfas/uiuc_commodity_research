"""Chronological-validation feature importance for the four anchored portfolios.

Run: python feature_importance.py
Refits saved full-feature hyperparameters on each fold's training portion. Permutes
individual input features and joint feature groups in contiguous observation blocks
within validation, never in the final test. Missingness is shuffled with feature values
through the complete fitted preprocessing pipeline. Model-dependent diagnostic only:
selected hyperparameters used these CV folds, so this is not nested validation.
"""
import argparse
import json
from pathlib import Path
import duckdb
import numpy as np
import pandas as pd
from sklearn.base import clone
from threadpoolctl import threadpool_limits
from configs.paths import DATA_DIR, PROJECT_DIR
from copper_ml import load_experiment, model_specs, make_pipeline
from plot import plot_feature_importance, plot_group_importance


def block_permutation(length, block_size, rng):
    blocks = [np.arange(start, min(start+block_size, length)) for start in range(0, length, block_size)]
    if len(blocks) < 2:
        raise ValueError("Need at least two observation blocks")
    order = rng.permutation(len(blocks))
    if np.array_equal(order, np.arange(len(blocks))):
        order = np.roll(order, 1)
    return np.concatenate([blocks[i] for i in order])


def perturbation_plan(columns):
    plan = [("individual", column, [column]) for column in columns]
    groups = {"Chinese features": [c for c in columns if c.startswith("cn_")],
              "US features": [c for c in columns if c.startswith("us_")],
              "Cross-market features": [c for c in columns if c.startswith("cross_")],
              "US + cross-market": [c for c in columns if c.startswith(("us_", "cross_"))]}
    return plan + [("group", name, members) for name, members in groups.items()]


def fold_importance(train, splits, columns, parameters, seed=42, repeats=10, block_size=5):
    rows, baselines = [], []
    plan = perturbation_plan(columns)
    for fold, (training, validation) in enumerate(splits, 1):
        print(f"  Fold {fold}: refit on {len(training)}, explain {len(validation)} validation observations", flush=True)
        x_train, y_train = train.iloc[training][columns], train.iloc[training].target_return
        x_val, actual = train.iloc[validation][columns], train.iloc[validation].target_return.to_numpy()
        if train.iloc[training].label_available_at.max() > train.iloc[validation].prediction_cutoff.min():
            raise ValueError("Fold label availability violation")
        rng = np.random.default_rng(seed + fold)
        permutations = [block_permutation(len(x_val), block_size, rng) for _ in range(repeats)]
        chunks, metadata = [], []
        for kind, name, members in plan:
            for repeat, order in enumerate(permutations):
                shuffled = x_val.copy()
                shuffled.loc[:, members] = x_val.iloc[order][members].to_numpy()
                chunks.append(shuffled)
                metadata.append((kind, name, repeat))
        batch = pd.concat(chunks, ignore_index=True)
        base_predictions, perturbed_predictions = {}, {}
        for name, (model, _) in model_specs(seed).items():
            pipeline = make_pipeline(clone(model)).set_params(**parameters[name])
            # Prediction batching makes single-worker forest traversal efficient;
            # tree fits/seed/parameters are identical to the baseline specification.
            if name == "Random forest":
                pipeline.set_params(model__n_jobs=1)
            with threadpool_limits(limits=2):
                pipeline.fit(x_train, y_train)
                base_predictions[name] = pipeline.predict(x_val)
                perturbed_predictions[name] = pipeline.predict(batch).reshape(len(metadata), len(x_val))
        base_predictions["Ensemble"] = np.mean(list(base_predictions.values()), axis=0)
        perturbed_predictions["Ensemble"] = np.mean(list(perturbed_predictions.values()), axis=0)
        for name in base_predictions:
            baseline = float(np.mean((actual-base_predictions[name])**2))
            losses = np.mean((perturbed_predictions[name]-actual[None, :])**2, axis=1)
            baselines.append(dict(fold=fold, model=name, validation_n=len(actual), baseline_mse=baseline,
                zero_mse=float(np.mean(actual**2)), validation_start=train.iloc[validation].target_trade_date.min(),
                validation_end=train.iloc[validation].target_trade_date.max(),
                latest_training_label=train.iloc[training].label_available_at.max(),
                first_validation_cutoff=train.iloc[validation].prediction_cutoff.min()))
            for (kind, feature, repeat), loss in zip(metadata, losses):
                rows.append(dict(fold=fold, model=name, kind=kind, feature=feature, repeat=repeat,
                    n=len(actual), baseline_mse=baseline, permuted_mse=float(loss), delta_mse=float(loss-baseline)))
    return pd.DataFrame(rows), pd.DataFrame(baselines)


def summarize_importance(raw):
    keys = ["anchor", "model", "kind", "feature"]
    values = raw.copy()
    values["delta_sse"] = values.delta_mse * values.n
    values["base_sse"] = values.baseline_mse * values.n
    repeat = values.groupby(keys+["repeat"], as_index=False).agg(delta_sse=("delta_sse", "sum"), base_sse=("base_sse", "sum"))
    repeat["importance_pct"] = 100 * repeat.delta_sse / repeat.base_sse
    summary = repeat.groupby(keys, as_index=False).agg(importance_mean_pct=("importance_pct", "mean"),
        permutation_std_pct=("importance_pct", lambda x: float(x.std(ddof=0))))
    fold = values.groupby(keys+["fold"], as_index=False).agg(mean_delta_mse=("delta_mse", "mean"))
    stability = fold.groupby(keys, as_index=False).agg(positive_folds=("mean_delta_mse", lambda x: int(x.gt(0).sum())), folds=("fold", "nunique"))
    return summary.merge(stability, on=keys, validate="one_to_one")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchors", nargs="+", choices=["copper", "gold", "crude", "soybeans"], default=["copper", "gold", "crude", "soybeans"])
    parser.add_argument("--input", type=Path, default=DATA_DIR / "prediction_data.duckdb")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--block-size", type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 2 or args.block_size < 1:
        parser.error("Use at least two repeats and positive block size")
    rows, baselines = [], []
    for anchor in dict.fromkeys(args.anchors):
        print(f"Feature importance: {anchor}", flush=True)
        train, _test, splits, scopes, _folds, _source = load_experiment(args.input, anchor)
        with duckdb.connect(str(DATA_DIR/f"{anchor}_ml.duckdb"), read_only=True) as conn:
            chosen = conn.sql("SELECT * FROM selected_parameters WHERE feature_set='domestic_us_cross'").df()
            run = conn.sql("SELECT * FROM run_config").df().iloc[0]
        seed = int(run.seed)
        parameters = {row.model: json.loads(row.best_params) for row in chosen.itertuples()}
        # Assert saved full-feature model choices are available for every family.
        if set(parameters) != set(model_specs(seed)):
            raise ValueError("Missing saved model hyperparameters")
        raw, baseline = fold_importance(train, splits, scopes["domestic_us_cross"], parameters,
            seed, args.repeats, args.block_size)
        raw["anchor"], baseline["anchor"] = anchor, anchor
        rows.append(raw)
        baselines.append(baseline)
    raw, baseline = pd.concat(rows, ignore_index=True), pd.concat(baselines, ignore_index=True)
    summary = summarize_importance(raw)
    directory = PROJECT_DIR / "results" / "feature_importance"
    directory.mkdir(parents=True, exist_ok=True)
    outputs = [plot_feature_importance(summary, anchor) for anchor in dict.fromkeys(args.anchors)]
    outputs.append(plot_group_importance(summary))
    tables = {"importance_summary": summary, "permutation_results": raw, "validation_baselines": baseline,
        "run_config": pd.DataFrame([dict(created_at_utc=pd.Timestamp.now(tz="UTC"), repeats=args.repeats,
            block_size=args.block_size, scope="chronological validation only; full feature set",
            selection="saved CV-selected hyperparameters; non-nested diagnostic",
            uncertainty="permutation-repeat spread is not a confidence interval",
            group_rule="same block permutation for all columns in a group; groups overlap, not additive")])}
    with duckdb.connect(str(DATA_DIR/"feature_importance.duckdb")) as conn:
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
    summary.to_csv(directory/"importance_summary.csv", index=False)
    baseline.to_csv(directory/"validation_baselines.csv", index=False)
    lines = ["# Feature importance", "",
        "Block permutation importance on three chronological validation folds, using the full 39-feature models. "
        "Each fold is refitted on its own historical training observations; test outcomes never enter this calculation.", "",
        "Importance = percentage increase in pooled validation MSE after shuffling a feature or group. "
        "Positive suggests the fitted model uses it beneficially; negative means shuffling improved its predictions. "
        "This is model-dependent and does not establish causal spillovers or market predictability.", "",
        f"Permutation setup: {args.repeats} repetitions, {args.block_size}-observation blocks. "
        "Error bars show repeat variation, not confidence intervals. Hyperparameters were selected using these CV folds; "
        "the rankings are diagnostic, not independently validated. Correlated features can substitute for one another. "
        "Joint group importance is calculated separately, not summed from individual scores.", "",
        "## Ensemble: strongest individual effects", "",
        "| Portfolio | Feature | MSE increase after shuffling | Positive folds |", "|---|---|---:|---:|"]
    for anchor in dict.fromkeys(args.anchors):
        group = summary.loc[summary.anchor.eq(anchor) & summary.model.eq("Ensemble") & summary.kind.eq("individual")]
        top = group.loc[group.importance_mean_pct.abs().nlargest(5).index]
        for row in top.itertuples():
            lines.append(f"| {anchor.title()} | {row.feature} | {row.importance_mean_pct:.3f}% | {row.positive_folds}/{row.folds} |")
    lines += ["", "## Figures", ""] + [f"- `{path}`" for path in outputs]
    lines += ["", "## Method reference", "", "[Scikit-learn permutation importance](https://scikit-learn.org/stable/modules/permutation_importance.html)", ""]
    (directory/"results.md").write_text("\n".join(lines), encoding="utf-8")
    print(summary.loc[summary.model.eq("Ensemble") & summary.kind.eq("group"),
        ["anchor", "feature", "importance_mean_pct", "positive_folds"]].to_string(index=False))
    print("Saved", directory, "and", DATA_DIR/"feature_importance.duckdb")


if __name__ == "__main__":
    main()
