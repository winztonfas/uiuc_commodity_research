# Copper portfolio ML baseline

Training: 2022-12-05 to 2025-09-29 (686 observations).
Held-out test: 2025-10-09 to 2026-09-30 (239 observations).

Three expanding-window tuning folds. Models are refitted once on all eligible training rows and then frozen for the test.
Preprocessing is fitted within each fold. Features: China only (21), +US (36), +cross-market (39).
Ensemble weights are fixed at one third each. Zero and training-mean forecasts are benchmarks.

| Model | Features | Test RMSE (bp) | MAE (bp) | R² vs zero | Direction accuracy |
|---|---|---:|---:|---:|---:|
| Ridge | domestic | 98.05 | 71.96 | -0.0102 | 57.3% |
| Random forest | domestic | 97.80 | 72.17 | -0.0048 | 55.6% |
| Gradient boosting | domestic | 97.88 | 72.16 | -0.0065 | 54.8% |
| Ensemble | domestic | 97.82 | 72.03 | -0.0053 | 54.8% |
| Ridge | domestic_us | 98.24 | 72.10 | -0.0139 | 51.9% |
| Random forest | domestic_us | 97.73 | 71.82 | -0.0035 | 57.7% |
| Gradient boosting | domestic_us | 97.29 | 71.48 | 0.0055 | 56.9% |
| Ensemble | domestic_us | 97.67 | 71.77 | -0.0022 | 57.3% |
| Ridge | domestic_us_cross | 98.26 | 72.14 | -0.0143 | 51.9% |
| Random forest | domestic_us_cross | 97.64 | 72.01 | -0.0016 | 57.3% |
| Gradient boosting | domestic_us_cross | 97.31 | 71.49 | 0.0051 | 56.9% |
| Ensemble | domestic_us_cross | 97.65 | 71.84 | -0.0018 | 56.5% |
| Zero return | baseline | 97.56 | 72.04 | 0.0000 | 0.0% |
| Training mean | baseline | 97.39 | 71.73 | 0.0036 | 57.7% |

## Incremental out-of-sample value

Positive values mean lower MSE after adding US and cross-market features. These paired comparisons use identical test dates.

| Model | Full versus domestic MSE reduction |
|---|---:|
| Ridge | -0.41% |
| Random forest | 0.32% |
| Gradient boosting | 1.16% |
| Ensemble | 0.34% |

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
- Random forest, domestic_us_cross: `{"model__max_depth": 3, "model__max_features": 0.7, "model__min_samples_leaf": 10}`
- Gradient boosting, domestic_us_cross: `{"model__learning_rate": 0.03, "model__max_depth": 1, "model__n_estimators": 100}`

Figure: `D:\PycharmProjects\uiuc_commodity_research-ml\plot_save\ml\copper_ml_comparison.png`
