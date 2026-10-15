# Gold portfolio ML baseline

Training: 2022-11-03 to 2025-09-29 (708 observations).
Held-out test: 2025-10-09 to 2026-09-30 (241 observations).

Three expanding-window tuning folds. Models are refitted once on all eligible training rows and then frozen for the test.
Preprocessing is fitted within each fold. Features: China only (21), +US (36), +cross-market (39).
Ensemble weights are fixed at one third each. Zero and training-mean forecasts are benchmarks.

| Model | Features | Test RMSE (bp) | MAE (bp) | R² vs zero | Direction accuracy |
|---|---|---:|---:|---:|---:|
| Ridge | domestic | 223.82 | 156.93 | 0.0040 | 54.4% |
| Random forest | domestic | 226.36 | 158.35 | -0.0188 | 51.5% |
| Gradient boosting | domestic | 224.81 | 157.83 | -0.0048 | 52.7% |
| Ensemble | domestic | 224.87 | 157.53 | -0.0054 | 51.0% |
| Ridge | domestic_us | 223.70 | 157.10 | 0.0050 | 53.5% |
| Random forest | domestic_us | 226.14 | 158.56 | -0.0168 | 48.1% |
| Gradient boosting | domestic_us | 225.18 | 157.93 | -0.0082 | 51.5% |
| Ensemble | domestic_us | 224.86 | 157.75 | -0.0053 | 50.2% |
| Ridge | domestic_us_cross | 224.16 | 157.82 | 0.0009 | 50.2% |
| Random forest | domestic_us_cross | 226.04 | 159.21 | -0.0159 | 49.4% |
| Gradient boosting | domestic_us_cross | 225.21 | 158.00 | -0.0084 | 51.0% |
| Ensemble | domestic_us_cross | 224.98 | 158.22 | -0.0064 | 51.5% |
| Zero return | baseline | 224.26 | 157.26 | 0.0000 | 0.0% |
| Training mean | baseline | 224.03 | 157.31 | 0.0021 | 52.3% |

## Incremental out-of-sample value

Positive values mean lower MSE after adding US and cross-market features. These paired comparisons use identical test dates.

| Model | Full versus domestic MSE reduction |
|---|---:|
| Ridge | -0.31% |
| Random forest | 0.28% |
| Gradient boosting | -0.36% |
| Ensemble | -0.10% |

## Interpretation and limits

These are initial exploratory forecasts, not evidence of causal spillovers or a profitable trading strategy. An improvement over a domestic model can coexist with underperformance versus a zero-return forecast. The test period must not be reused to tune the model after viewing these results.

Availability uses assumed China +24h and US UTC completion +5min delays. Targets freeze weights and contracts at cutoff; they can differ from the descriptive index on roll/rebalance dates. Rows with missing frozen-contract targets are excluded. Same evaluation dates are used across all feature sets. Regime analysis and statistical uncertainty remain extensions.

Negative R² versus zero means worse MSE than forecasting zero every day. Direction accuracy for a zero forecast is the share of exactly zero actual returns, not a useful directional classifier.

## Selected parameters

- Ridge, domestic: `{"model__alpha": 1000.0}`
- Random forest, domestic: `{"model__max_depth": 3, "model__max_features": 0.7, "model__min_samples_leaf": 10}`
- Gradient boosting, domestic: `{"model__learning_rate": 0.03, "model__max_depth": 1, "model__n_estimators": 100}`
- Ridge, domestic_us: `{"model__alpha": 1000.0}`
- Random forest, domestic_us: `{"model__max_depth": 3, "model__max_features": 0.7, "model__min_samples_leaf": 10}`
- Gradient boosting, domestic_us: `{"model__learning_rate": 0.03, "model__max_depth": 1, "model__n_estimators": 100}`
- Ridge, domestic_us_cross: `{"model__alpha": 1000.0}`
- Random forest, domestic_us_cross: `{"model__max_depth": 3, "model__max_features": 0.7, "model__min_samples_leaf": 30}`
- Gradient boosting, domestic_us_cross: `{"model__learning_rate": 0.03, "model__max_depth": 1, "model__n_estimators": 100}`

Figure: `D:\PycharmProjects\uiuc_commodity_research-ml\plot_save\ml\gold_ml_comparison.png`
