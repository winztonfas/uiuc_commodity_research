# Feature importance

Block permutation importance on three chronological validation folds, using the full 39-feature models. Each fold is refitted on its own historical training observations; test outcomes never enter this calculation.

Importance = percentage increase in pooled validation MSE after shuffling a feature or group. Positive suggests the fitted model uses it beneficially; negative means shuffling improved its predictions. This is model-dependent and does not establish causal spillovers or market predictability.

Permutation setup: 10 repetitions, 5-observation blocks. Error bars show repeat variation, not confidence intervals. Hyperparameters were selected using these CV folds; the rankings are diagnostic, not independently validated. Correlated features can substitute for one another. Joint group importance is calculated separately, not summed from individual scores.

## Ensemble: strongest individual effects

| Portfolio | Feature | MSE increase after shuffling | Positive folds |
|---|---|---:|---:|
| Copper | cn_momentum_20 | -0.459% | 2/3 |
| Copper | us_momentum_20 | -0.403% | 0/3 |
| Copper | cn_momentum_5 | 0.187% | 2/3 |
| Copper | cn_return_lag1 | 0.150% | 3/3 |
| Copper | cn_shock_20 | 0.143% | 3/3 |
| Gold | cn_oi_change_1 | 0.856% | 2/3 |
| Gold | cross_correlation_60 | 0.257% | 3/3 |
| Gold | cross_sensitivity_60 | 0.146% | 3/3 |
| Gold | cn_momentum_20 | 0.114% | 3/3 |
| Gold | us_momentum_5 | -0.092% | 0/3 |
| Crude | cn_momentum_20 | 1.049% | 3/3 |
| Crude | cn_volume_relative_20 | -0.582% | 0/3 |
| Crude | cn_turnover_relative_20 | -0.415% | 0/3 |
| Crude | cn_turnover_breakout_ratio_20 | -0.402% | 0/3 |
| Crude | cn_absolute_return | -0.382% | 1/3 |
| Soybeans | us_return_lag1 | -0.399% | 1/3 |
| Soybeans | us_volatility_20 | -0.333% | 2/3 |
| Soybeans | cross_momentum_divergence_20 | 0.321% | 2/3 |
| Soybeans | us_volume_relative_20 | -0.269% | 1/3 |
| Soybeans | cn_absolute_return | -0.206% | 2/3 |

## Figures

- `D:\PycharmProjects\uiuc_commodity_research-ml\plot_save\feature_importance\copper_feature_importance.png`
- `D:\PycharmProjects\uiuc_commodity_research-ml\plot_save\feature_importance\gold_feature_importance.png`
- `D:\PycharmProjects\uiuc_commodity_research-ml\plot_save\feature_importance\crude_feature_importance.png`
- `D:\PycharmProjects\uiuc_commodity_research-ml\plot_save\feature_importance\soybeans_feature_importance.png`
- `D:\PycharmProjects\uiuc_commodity_research-ml\plot_save\feature_importance\feature_group_importance.png`

## Method reference

[Scikit-learn permutation importance](https://scikit-learn.org/stable/modules/permutation_importance.html)
