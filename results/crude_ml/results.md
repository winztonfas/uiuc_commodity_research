# Crude portfolio ML baseline

Training: 2022-11-03 to 2025-09-29 (708 observations).
Held-out test: 2025-10-09 to 2026-09-30 (241 observations).

Three expanding-window tuning folds. Models are refitted once on all eligible training rows and then frozen for the test.
Preprocessing is fitted within each fold. Features: China only (21), +US (36), +cross-market (39).
Ensemble weights are fixed at one third each. Zero and training-mean forecasts are benchmarks.

| Model | Features | Test RMSE (bp) | MAE (bp) | R² vs zero | Direction accuracy |
|---|---|---:|---:|---:|---:|
| Ridge | domestic | 265.24 | 183.20 | -0.0349 | 49.8% |
| Random forest | domestic | 266.90 | 185.73 | -0.0479 | 48.1% |
| Gradient boosting | domestic | 267.45 | 186.22 | -0.0523 | 48.5% |
| Ensemble | domestic | 266.19 | 184.75 | -0.0424 | 47.7% |
| Ridge | domestic_us | 266.66 | 183.96 | -0.0460 | 47.7% |
| Random forest | domestic_us | 267.31 | 186.35 | -0.0511 | 48.5% |
| Gradient boosting | domestic_us | 267.31 | 186.00 | -0.0512 | 47.3% |
| Ensemble | domestic_us | 266.77 | 185.10 | -0.0469 | 49.0% |
| Ridge | domestic_us_cross | 267.17 | 184.49 | -0.0501 | 48.1% |
| Random forest | domestic_us_cross | 267.46 | 186.33 | -0.0523 | 49.4% |
| Gradient boosting | domestic_us_cross | 267.05 | 185.77 | -0.0491 | 47.3% |
| Ensemble | domestic_us_cross | 266.95 | 185.25 | -0.0483 | 46.9% |
| Zero return | baseline | 260.73 | 179.37 | 0.0000 | 0.0% |
| Training mean | baseline | 260.48 | 179.17 | 0.0019 | 55.2% |

## Incremental out-of-sample value

Positive values mean lower MSE after adding US and cross-market features. These paired comparisons use identical test dates.

| Model | Full versus domestic MSE reduction |
|---|---:|
| Ridge | -1.46% |
| Random forest | -0.42% |
| Gradient boosting | 0.30% |
| Ensemble | -0.57% |

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

Figure: `D:\PycharmProjects\uiuc_commodity_research-ml\plot_save\ml\crude_ml_comparison.png`
