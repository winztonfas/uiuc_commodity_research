# Anchored-portfolio ML comparison

Identical model families, tuning grids, publication-delay assumptions, chronological folds, and October 2025 holdout boundary. Dates are identical across feature sets within each portfolio; sample counts can differ across portfolios. Copper results are reused without retuning.

Positive MSE reduction means US + cross-market features improve on the domestic-only version. Positive R² versus zero means the full model also beats forecasting zero. Both comparisons matter.

| Portfolio | Model | Test n | MSE reduction vs domestic | R² vs zero | R² vs training mean | Direction |
|---|---|---:|---:|---:|---:|---:|
| Copper | Ridge | 239 | -0.41% | -0.0143 | -0.0179 | 51.9% |
| Copper | Random forest | 239 | 0.32% | -0.0016 | -0.0052 | 57.3% |
| Copper | Gradient boosting | 239 | 1.16% | 0.0051 | 0.0015 | 56.9% |
| Copper | Ensemble | 239 | 0.34% | -0.0018 | -0.0054 | 56.5% |
| Gold | Ridge | 241 | -0.31% | 0.0009 | -0.0012 | 50.2% |
| Gold | Random forest | 241 | 0.28% | -0.0159 | -0.0180 | 49.4% |
| Gold | Gradient boosting | 241 | -0.36% | -0.0084 | -0.0105 | 51.0% |
| Gold | Ensemble | 241 | -0.10% | -0.0064 | -0.0085 | 51.5% |
| Crude | Ridge | 241 | -1.46% | -0.0501 | -0.0520 | 48.1% |
| Crude | Random forest | 241 | -0.42% | -0.0523 | -0.0543 | 49.4% |
| Crude | Gradient boosting | 241 | 0.30% | -0.0491 | -0.0511 | 47.3% |
| Crude | Ensemble | 241 | -0.57% | -0.0483 | -0.0503 | 46.9% |
| Soybeans | Ridge | 241 | -3.62% | -0.0358 | -0.0356 | 48.5% |
| Soybeans | Random forest | 241 | -0.33% | -0.0159 | -0.0157 | 48.1% |
| Soybeans | Gradient boosting | 241 | -1.37% | -0.0247 | -0.0246 | 45.6% |
| Soybeans | Ensemble | 241 | -1.53% | -0.0199 | -0.0198 | 47.7% |

All sectors/models are reported. Choosing a presentation case after viewing these test results is exploratory selection; it does not establish a reliable winning sector. No significance or causal spillover is claimed. Avoid using the same test outcomes to retune the protocol.

Figure: `D:\PycharmProjects\uiuc_commodity_research-ml\plot_save\ml\sector_ml_summary.png`
