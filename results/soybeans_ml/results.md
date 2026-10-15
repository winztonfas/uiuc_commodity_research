# Soybeans portfolio ML baseline

Training: 2022-11-03 to 2025-09-29 (708 observations).
Held-out test: 2025-10-09 to 2026-09-30 (241 observations).

Three expanding-window tuning folds. Models are refitted once on all eligible training rows and then frozen for the test.
Preprocessing is fitted within each fold. Features: China only (21), +US (36), +cross-market (39).
Ensemble weights are fixed at one third each. Zero and training-mean forecasts are benchmarks.

| Model | Features | Test RMSE (bp) | MAE (bp) | R² vs zero | Direction accuracy |
|---|---|---:|---:|---:|---:|
| Ridge | domestic | 73.49 | 53.87 | 0.0004 | 49.8% |
| Random forest | domestic | 73.97 | 54.64 | -0.0125 | 46.9% |
| Gradient boosting | domestic | 73.91 | 54.48 | -0.0109 | 49.8% |
| Ensemble | domestic | 73.68 | 54.24 | -0.0045 | 49.8% |
| Ridge | domestic_us | 74.35 | 54.82 | -0.0229 | 45.2% |
| Random forest | domestic_us | 74.17 | 54.75 | -0.0180 | 46.9% |
| Gradient boosting | domestic_us | 74.00 | 54.47 | -0.0133 | 49.4% |
| Ensemble | domestic_us | 74.01 | 54.61 | -0.0136 | 48.1% |
| Ridge | domestic_us_cross | 74.81 | 55.12 | -0.0358 | 48.5% |
| Random forest | domestic_us_cross | 74.09 | 54.43 | -0.0159 | 48.1% |
| Gradient boosting | domestic_us_cross | 74.41 | 54.64 | -0.0247 | 45.6% |
| Ensemble | domestic_us_cross | 74.24 | 54.64 | -0.0199 | 47.7% |
| Zero return | baseline | 73.51 | 53.89 | 0.0000 | 0.0% |
| Training mean | baseline | 73.52 | 53.89 | -0.0001 | 49.4% |

## Incremental out-of-sample value

Positive values mean lower MSE after adding US and cross-market features. These paired comparisons use identical test dates.

| Model | Full versus domestic MSE reduction |
|---|---:|
| Ridge | -3.62% |
| Random forest | -0.33% |
| Gradient boosting | -1.37% |
| Ensemble | -1.53% |

## Interpretation and limits

These are initial exploratory forecasts, not evidence of causal spillovers or a profitable trading strategy. An improvement over a domestic model can coexist with underperformance versus a zero-return forecast. The test period must not be reused to tune the model after viewing these results.

Availability uses assumed China +24h and US UTC completion +5min delays. Targets freeze weights and contracts at cutoff; they can differ from the descriptive index on roll/rebalance dates. Rows with missing frozen-contract targets are excluded. Same evaluation dates are used across all feature sets. Regime analysis and statistical uncertainty remain extensions.

Negative R² versus zero means worse MSE than forecasting zero every day. Direction accuracy for a zero forecast is the share of exactly zero actual returns, not a useful directional classifier.

## Selected parameters

- Ridge, domestic: `{"model__alpha": 1000.0}`
- Random forest, domestic: `{"model__max_depth": 3, "model__max_features": 0.7, "model__min_samples_leaf": 30}`
- Gradient boosting, domestic: `{"model__learning_rate": 0.03, "model__max_depth": 1, "model__n_estimators": 100}`
- Ridge, domestic_us: `{"model__alpha": 1000.0}`
- Random forest, domestic_us: `{"model__max_depth": 3, "model__max_features": 0.7, "model__min_samples_leaf": 30}`
- Gradient boosting, domestic_us: `{"model__learning_rate": 0.03, "model__max_depth": 1, "model__n_estimators": 100}`
- Ridge, domestic_us_cross: `{"model__alpha": 1000.0}`
- Random forest, domestic_us_cross: `{"model__max_depth": 3, "model__max_features": 0.7, "model__min_samples_leaf": 30}`
- Gradient boosting, domestic_us_cross: `{"model__learning_rate": 0.03, "model__max_depth": 1, "model__n_estimators": 100}`

Figure: `D:\PycharmProjects\uiuc_commodity_research-ml\plot_save\ml\soybeans_ml_comparison.png`
