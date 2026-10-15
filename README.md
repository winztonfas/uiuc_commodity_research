# Predictive Information Across US and Chinese Commodity Futures

Research on whether available US futures information improves forecasts of Chinese commodity **portfolio returns**, rather than a single futures contract.

**Course:** IE498MLF, Fall 2026 · **Updated:** October 15, 2026  
**Repository:** https://github.com/winztonfas/uiuc_commodity_research

Current findings are modest: US information gives small, model-dependent gains for copper, while most tested configurations fail to beat simple return forecasts. Improvement is interpreted as **predictive information**, not proof of causal spillovers or trading profitability.

## 1. Introduction and exploratory data analysis

### Objectives

- Describe cross-market relationships between economically related commodity groups.
- Construct adaptive portfolios whose members change with historical return relationships.
- Compare domestic-only forecasts with forecasts using additional US information.
- Examine which features and groups the fitted models use across chronological validation periods.

Regime-conditioned performance and China-to-US prediction remain future extensions, not completed results.

### Data

The research window is **October 2021–September 2026**. Local Chinese source tables can contain earlier history used for product-feature warmup. This does not extend adaptive target history back to 2021.

| Item | China | US |
|---|---|---|
| Provider | Tushare futures daily data | Databento `GLBX.MDP3`, daily OHLCV |
| Configured products | 23 | 16 |
| Selected-contract rows in the research window | 25,843 | 22,251 |
| First selected-contract observation in that window | 2021-10-08 | 2021-10-03 |
| Last observation | 2026-09-30 | 2026-09-30 |
| Fields used | OHLC, `vol`, `amount`, `oi` | OHLC and `vol` |
| Dates | Chinese exchange trading-day labels | UTC daily interval labels |

Counts describe selected product-date rows, not raw expiry contracts, and are a snapshot of the configured universe. Chinese `amount` is monetary turnover, not contract volume. Chinese settlement fields exist in the source but are not used for these close-based targets. Returns remain in local price currencies, without FX hedging or common-currency conversion.

| Sector | Chinese roots | US roots |
|---|---|---|
| Metals | AU, AG, CU, AL, ZN, PB, NI, SN, PT, PD | GC, SI, HG, ALI |
| Energy | SC, FU, LU, BU | CL, HO, RB, NG |
| Agriculture | B, M, Y, C, PM, WH, JR, RI, LH | ZS, ZM, ZL, ZC, ZW, KE, ZR, HE |

[configs/mapping.py](configs/mapping.py) is authoritative. PT/PD have short histories. Downloaded products outside this mapping, such as HRC, are excluded from the current analysis.

### Descriptive benchmarks

Equal-weight sector baskets average the available product returns by date and compound them into reference indices. Coverage can change. These are transparent benchmarks, not perfectly matched country exposures.

EDA includes cumulative indices, histograms, boxplots, rolling volatility, product correlation heatmaps, and dendrograms. Visual co-movement does not establish a predictive lead. Full-history heatmaps are descriptive and do not select predictors for the held-out test.

- [Descriptive calculations](descriptive.py)
- [Plot functions](plot.py)
- [Return distributions](plot_save/boxplot/sector_return_boxplot.png)
- [Rolling volatility](plot_save/rolling_volatility/sector_rolling_volatility_20.png)

## 2. Preprocessing, feature extraction, and feature selection

### Continuous-contract returns

[rollover.py](rollover.py) constructs US product series from raw outright futures bars. It uses volume leadership, two-observation confirmation, and forward-only rolls. An end-of-bar selection takes effect on the next product observation. Expiry-based handling requires an optional expiry file.

US `contract_return` compares the selected expiry's current close with that same expiry's previous product-date close. It avoids artificial returns from switching contract price levels. OHLC levels remain unadjusted and are not a fully back-adjusted price series.

Chinese descriptive returns use `(close - pre_close) / pre_close`. Imported Chinese contract selection can depend on same-day activity, which is historical information only after publication. ML targets therefore freeze contract choices at the cutoff and look up both endpoint closes in the raw same-expiry table.

### Calendars and missing values

For US metals, compound weekend returns into Monday rather than deleting them:

```text
combined return = product of (1 + source-bar returns) - 1
```

Volume/turnover are summed over folded observations. A destination weekday observation is required; genuine weekday gaps remain missing. A completed Monday aggregate is not available on Monday morning. Other sectors retain their existing date grids.

The descriptive US copper index excludes missing members on an affected date and renormalizes remaining HRP weights. Ten weekday exclusions occurred in the recorded run after weekend folding. Effective weights and exclusions are stored. This is a descriptive available-member index, not a claim of executable advance knowledge about missing observations.

ML targets do not drop constituents using future target-day availability. Missing or invalid endpoint quotes for frozen Chinese contracts make a target unavailable. Two copper targets are excluded for this reason. Missing targets are never imputed.

### Rolling clustering and HRP

[cluster_index.py](cluster_index.py) builds portfolios separately in each market:

1. Compound daily returns into `W-SAT` weekly observations; do not average prices.
2. At monthly cutoffs, use the previous **52 completed weeks**.
3. Require at most 10 missing weeks and nonzero variation. A common complete weekly sample must retain at least 42 observations and two products.
4. Transform correlation to distance: `d(i,j) = sqrt((1 - correlation(i,j)) / 2)`.
5. Use **average linkage** and a distance cut of **0.46**.
6. Retain clusters with at least two products and estimate **HRP weights** from historical covariance.
7. Hold target weights fixed within the monthly vintage, apart from the documented descriptive copper availability convention.

A hierarchical distance cut does not guarantee an identical minimum correlation between every pair in a retained cluster.

| Anchor | China | US |
|---|---|---|
| Gold | AU | GC |
| Copper | CU | HG |
| Crude oil | SC | CL |
| Soybeans | B | ZS |
| Wheat | WH | ZW |

An anchor labels a changing portfolio, not a single-contract target. If it becomes a singleton, retain previous eligible members and recalculate their HRP weights. If no previous eligible cluster exists, or estimation history is insufficient, leave it unavailable. Chinese wheat remains empty in this run; WH/PM liquidity is inadequate for the intended comparison.

Red chart triangles mark singleton fallback starts. Blue triangles mark effective constituent changes, including daily exclusions and reinstatements. Weights-only changes do not trigger member-change markers.

**Portfolio construction is separate from ML.** The models never optimize HRP weights or clusters to reduce forecast MSE. Portfolios can update under their historical rules during testing while the final prediction models remain fixed. Separate country clusters do not imply identical exposures.

[Adaptive indices](plot_save/cluster_indexes/anchor_cluster_indexes.png)

### Forecast definition and availability

[build_prediction_data.py](build_prediction_data.py) uses:

| Item | Definition |
|---|---|
| Forecast cutoff / target start | Previous observed Chinese trading date, 15:00 `Asia/Shanghai` |
| Target end | Target Chinese trading date, 15:00 |
| Chinese availability assumption | Source close plus **24 hours** |
| US availability assumption | Source UTC daily interval end plus **5 minutes** |
| Join | Latest available source row by backward as-of join, default maximum age 30 days |

Delays are explicit conservative **assumptions**, not observed historical arrival timestamps. The Chinese delay can make features two target observations old. Do not change delay assumptions just to improve scores.

For an ordinary Monday target, the cutoff is Friday afternoon and the latest completed US daily bar is normally Thursday UTC. Completed US Friday/Sunday information is too late for the entire Chinese Monday interval, which can include Friday night.

The builder enforces:

```text
feature_available_at <= prediction_cutoff <= target_start < target_end
```

Target weights and selected contracts must be known at the cutoff. Each constituent return uses its frozen contract's previous-date and target-date closes, then frozen HRP weights aggregate those returns. Targets can differ from descriptive index returns on roll/rebalance dates.

Audit columns retain source dates, assumed publication timestamps, ages, endpoints, frozen selections/weights, and missing-target status. Calendar-date cross-market EDA features are not reused as predictors: cross-market features are recalculated after availability alignment.

### Feature definitions: 39 candidates

[features.py](features.py) contains calculation functions only. [calculate_features.py](calculate_features.py) saves the exploratory feature tables.

The allowlist is **21 Chinese + 15 US + 3 cross-market**. Training-created missing indicators can increase the transformed dimension; constant-feature removal can reduce it. `lag0` means the latest completed observation available at the cutoff, not the target date. Windows count observations, not calendar days.

| Shared feature, prefixed `cn_` or `us_` | Definition | Count |
|---|---|---:|
| `return_lag0`, `return_lag1` | Latest and preceding available portfolio return | 2 |
| `absolute_return` | Absolute latest return | 1 |
| `momentum_5`, `momentum_20` | Compounded trailing 5-/20-observation returns | 2 |
| `volatility_5`, `volatility_20` | Sample standard deviation of trailing 5/20 portfolio returns, not annualized | 2 |
| `shock_20` | Latest return / preceding 20-return standard deviation, excluding current return | 1 |
| `volume_relative_20` | Product volume / previous 20-observation mean, then weighted aggregation | 1 |
| `volume_breakout_ratio_20` | Product volume / previous 20-observation maximum, then weighted aggregation | 1 |
| `volume_breakout_20` | Product breakout indicator, then weighted aggregation | 1 |
| `positive_return_breadth` | Unweighted fraction of active products with positive returns | 1 |
| `constituent_count`, `largest_weight` | Active product count and maximum effective weight | 2 |
| `shock_volume_interaction` | Portfolio shock times weighted relative volume | 1 |
| **Total per market** | | **15** |

| Chinese additions, prefixed `cn_` | Definition | Count |
|---|---|---:|
| `turnover_relative_20` | Product amount / previous 20-observation mean, then weighted aggregation | 1 |
| `turnover_breakout_ratio_20` | Product amount / previous 20-observation maximum, then weighted aggregation | 1 |
| `turnover_breakout_20` | Product turnover breakout indicator, then weighted aggregation | 1 |
| `oi_change_1`, `oi_change_5` | Product OI percentage changes over 1/5 observations, then weighted aggregation | 2 |
| `return_oi_interaction` | Product return times 1-observation OI change, then weighted aggregation | 1 |
| **China-only additions** | | **6** |

| Cross-market feature | Definition |
|---|---|
| `cross_correlation_60` | Correlation over 60 forecast observations, minimum 30 paired cutoff-known returns |
| `cross_sensitivity_60` | Corresponding Chinese/US covariance divided by US return variance |
| `cross_momentum_divergence_20` | Difference between local 20-observation momenta, each divided by available 20-observation volatility times `sqrt(20)` |

Cross-market returns can have different endpoints. These are cutoff-known associations, not causal or necessarily concurrent correlations. US observations can repeat across forecast cutoffs; timestamps and age columns expose this.

Normalize activity within products before aggregation using effective weights. Do not add raw volumes across commodities. Compute portfolio volatility from portfolio returns, not average constituent volatility. The current run uses selected-contract activity rather than all-expiry totals, so rollover migration can affect activity ratios. OI changes spanning selected-contract rolls are masked.

Fallback, member-change, exclusion, available-weight, timestamp/age, roll, interval-bar-count, and paired-count fields are diagnostics outside the 39-feature allowlist. Warmup, unsupported fields, and zero denominators remain missing in calculated tables.

### Learned preprocessing and selection

Every pipeline fits, within its own training fold:

1. Median imputation and missing indicators.
2. `VarianceThreshold` removal of constant features.
3. `StandardScaler` standardization.
4. The regression model.

Validation/test data only receive fitted transformations. This baseline has no PCA or supervised feature-selection stage. Full-history heatmaps do not automatically select or remove inputs.

## 3. Model fitting and evaluation

Models are **Ridge regression**, **random forest regression**, and **gradient boosting regression**. Each is fitted separately for `domestic` (21 inputs), `domestic_us` (36), and `domestic_us_cross` (39).

This provides group ablations: add US to domestic features, then add/remove cross-market features. Comparisons within a portfolio use identical eligible evaluation dates.

### Held-out test

The split boundary is **2025-10-01**. The actual first Chinese test date is **2025-10-09**, after the holiday, and the last is **2026-09-30**.

| Portfolio | Training n | Test n |
|---|---:|---:|
| Copper | 686 | 239 |
| Gold | 708 | 241 |
| Crude oil | 708 | 241 |
| Soybeans | 708 | 241 |

Copper training runs **2022-12-05–2025-09-29**. The original research data start earlier, but initial estimation, portfolio eligibility, and alignment delay the usable model sample.

After tuning, refit each pipeline once on all eligible training observations. Model parameters remain **fixed throughout the final test**; this is not monthly model retraining.

### Boundary purging

Every training target must be available at the first evaluation forecast cutoff. For the October 9 test target, that cutoff is September 30 at 15:00 China time. Under the assumed 24-hour delay, September 30's label is unavailable then. That row is marked `purged_boundary` and excluded from training. The same rule applies within CV.

### Metrics and simple benchmarks

| Metric | Meaning |
|---|---|
| MSE | Average squared return error |
| RMSE | Square root of MSE, in return units |
| MAE | Average absolute return error |
| Standard R² | Error reduction relative to deviations from the test mean, an evaluation reference |
| OOS R² vs zero | `1 - model_MSE / zero_forecast_MSE` |
| OOS R² vs training mean | `1 - model_MSE / training_mean_forecast_MSE` |
| Direction accuracy | Fraction with `sign(actual) == sign(prediction)`; zeros are a separate sign |
| Prediction correlation | Pearson correlation between forecasts and actual returns, when defined |

The zero benchmark predicts 0 on every date without fitting a model. The training-mean benchmark always predicts the training average, never the test average. Positive OOS R² means lower MSE than its benchmark; negative means higher error, not proof that no predictive signal exists. Basis points equal return multiplied by 10,000.

Direction accuracy is not profitability and does not guarantee lower MSE. Always-positive predictions can exceed 50% in an upward-biased sample, so 50% is not automatically a meaningful directional benchmark.

## 4. Hyperparameter tuning

`GridSearchCV` receives explicitly saved chronological indices, not ordinary K-fold. The training window expands and validation always follows it. Refit preprocessing and the estimator from scratch for every candidate/fold.

| Copper fold | Train start | Train end | Validation start | Validation end | Train / validation n |
|---|---|---|---|---|---:|
| 1 | 2022-12-05 | 2024-03-28 | 2024-04-01 | 2024-09-30 | 319 / 123 |
| 2 | 2022-12-05 | 2024-09-27 | 2024-10-08 | 2025-03-31 | 442 / 118 |
| 3 | 2022-12-05 | 2025-03-28 | 2025-04-01 | 2025-09-29 | 560 / 125 |

Other portfolios use the same validation periods with different eligible training counts.

| Model | Grid | Combinations |
|---|---|---:|
| Ridge | `alpha`: 0.1, 1, 10, 100, 1000 | 5 |
| Random forest | `max_depth`: 3, 6; `min_samples_leaf`: 10, 30 | 4 |
| Gradient boosting | `n_estimators`: 100, 200; `learning_rate`: 0.03, 0.1; `max_depth`: 1, 2 | 8 |

Random forest fixes 200 trees and `max_features=0.7`. Boosting fixes `min_samples_leaf=20` and disables its internal early-stopping validation split. Saved experiments use random seed 42.

Select the lowest **unweighted average of three validation MSEs**, separately for each model/input set. Across three input sets this is `3 × 3 × (5+4+8) = 153` CV fits, then 9 final refits per portfolio. Candidate validation scores are saved; training-loss curves are not.

The full copper model selects Ridge `alpha=1000`; forest depth 3 and leaf size 10; boosting 100 trees, learning rate 0.03, and depth 1. Other feature sets choose parameters separately.

Grid search uses one worker; the random forest uses two CPU workers. Thread pools are limited during fitting. Test outcomes never choose hyperparameters or ensemble weights.

## 5. Ensembling

For each input set:

```text
ensemble forecast = (Ridge forecast + random forest forecast + boosting forecast) / 3
```

Weights are fixed at one third, not tuned on the test. These model-combination weights are distinct from HRP commodity weights. Averaging can reduce model-specific variation but does not guarantee beating the best individual model.

## 6. Results and conclusions

### Incremental information

Percentage MSE reduction from adding US plus cross-market features to the same domestic-only model. Positive means improvement, negative means deterioration.

| Portfolio | Ridge | Random forest | Boosting | Ensemble |
|---|---:|---:|---:|---:|
| Copper | -0.41% | +0.32% | +1.16% | +0.34% |
| Gold | -0.31% | +0.28% | -0.36% | -0.10% |
| Crude | -1.46% | -0.42% | +0.30% | -0.57% |
| Soybeans | -3.62% | -0.33% | -1.37% | -1.53% |

### Full-feature model performance

OOS R² values below are fractions, not percentages.

| Portfolio | Model | Test n | RMSE (bp) | OOS R² vs zero | OOS R² vs train mean | Direction |
|---|---|---:|---:|---:|---:|---:|
| Copper | Ridge | 239 | 98.26 | -0.0143 | -0.0179 | 51.9% |
| Copper | Random forest | 239 | 97.64 | -0.0016 | -0.0052 | 57.3% |
| Copper | Gradient boosting | 239 | 97.31 | 0.0051 | 0.0015 | 56.9% |
| Copper | Ensemble | 239 | 97.65 | -0.0018 | -0.0054 | 56.5% |
| Gold | Ridge | 241 | 224.16 | 0.0009 | -0.0012 | 50.2% |
| Gold | Random forest | 241 | 226.04 | -0.0159 | -0.0180 | 49.4% |
| Gold | Gradient boosting | 241 | 225.21 | -0.0084 | -0.0105 | 51.0% |
| Gold | Ensemble | 241 | 224.98 | -0.0064 | -0.0085 | 51.5% |
| Crude | Ridge | 241 | 267.17 | -0.0501 | -0.0520 | 48.1% |
| Crude | Random forest | 241 | 267.46 | -0.0523 | -0.0543 | 49.4% |
| Crude | Gradient boosting | 241 | 267.05 | -0.0491 | -0.0511 | 47.3% |
| Crude | Ensemble | 241 | 266.95 | -0.0483 | -0.0503 | 46.9% |
| Soybeans | Ridge | 241 | 74.81 | -0.0358 | -0.0356 | 48.5% |
| Soybeans | Random forest | 241 | 74.09 | -0.0159 | -0.0157 | 48.1% |
| Soybeans | Gradient boosting | 241 | 74.41 | -0.0247 | -0.0246 | 45.6% |
| Soybeans | Ensemble | 241 | 74.24 | -0.0199 | -0.0198 | 47.7% |

Copper boosting lowers domestic-model MSE by about **1.16%**, but lowers zero-forecast MSE by only about **0.51%**. This is error improvement, not investment return. Most configurations do not beat simple benchmarks. Gold/crude gains are inconsistent and US additions worsen soybean forecasts in this run.

All sectors are reported. Selecting copper for presentation after inspecting the tests is exploratory, not independent confirmation of a winning sector. Statistical significance is not established. The test has already been inspected; do not repeatedly tune the protocol against it.

- [Copper detailed results](results/copper_ml/results.md)
- [All-portfolio results](results/sector_ml/results.md)
- [Copper results figure](plot_save/ml/copper_ml_comparison.png)
- [Four-portfolio comparison](plot_save/ml/sector_ml_summary.png)

### Feature importance

[feature_importance.py](feature_importance.py) refits saved full-model parameters on each chronological training fold. It shuffles individual features or groups within **validation** in five-observation blocks, with ten repetitions, then predicts without retraining. Missingness moves with feature values through the full pipeline.

Importance is the percentage increase in pooled validation MSE. Positive means disruption worsens predictions; negative means shuffling helps. Repeat effects pool observation-weighted squared errors across folds. Error bars are permutation-repeat spread, not confidence intervals.

Joint groups are Chinese features, US features, cross-market features, and US plus cross-market. Group columns share a permutation. Groups overlap and their effects are not additive. Correlated features can substitute for one another.

**Positive folds** counts folds where mean MSE change across repetitions is above zero. A 3/3 result is sign consistency, not statistical significance or direction accuracy.

Hyperparameters were selected using these CV periods, so importance is a **non-nested diagnostic**, not independent validation. Final-test outcomes are not used. Rankings do not establish causal spillovers or make a weak model economically useful.

- [Importance explanation and tables](results/feature_importance/results.md)
- [Copper importance plot](plot_save/feature_importance/copper_feature_importance.png)
- [Joint group importance](plot_save/feature_importance/feature_group_importance.png)

### Limits and next steps

- Publication delays are assumed; historical arrival timestamps are unavailable.
- Daily Chinese bars cannot isolate night/day sessions. Fresh US Friday data overlaps a full Chinese Monday target.
- Selected-contract activity is sensitive to expiry migration; all-expiry activity is a robustness extension.
- Local clusters have different exposures, and singleton fallbacks can retain members outside the latest eligible cluster.
- Calendar folding and observed-member normalization are descriptive conventions, not executable trading rules.
- Roughly 700 training observations and one test year provide limited evidence of stability.
- No position rule, transaction costs, slippage, collateral yield, trading backtest, or profit claim is included.
- Future work: measured timing, intraday targets, chronological refitting, uncertainty analysis, regime-conditioned evaluation, and bidirectional prediction.

## Reproduction and repository structure

### Setup and data prerequisites

Use Python 3.12 to match the saved environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On macOS, activate with `source .venv/bin/activate`. Analysis paths are relative to the repository through [configs/paths.py](configs/paths.py).

| Local source | Required contents |
|---|---|
| `data/processed/futures_data.ddb` | Chinese `rolling_futures` and raw `future_daily` |
| `data/processed/us_continuous.duckdb` | US `rolling_futures`, including `contract_return` and volume |
| `data/raw/databento/commodities.duckdb` | `raw_ohlcv_1d`, only needed to rebuild US continuous data |

Data are not included in Git. The Chinese database comes from an external preparation workflow; this repository does not provide a complete fresh-download Chinese pipeline. Raw same-expiry Chinese bars are required to construct frozen targets.

Legacy download helpers are in `bentofetch/` and require separate Databento access, `databento`, and `python-dotenv`. Review paths before using them: the estimator currently imports from `fetch.bento_fetch`, and the fetcher derives `.env` and data locations from its own folder. These helpers are not the validated reproduction entry point below. No paid download is needed when source databases already exist.

Keep `DATABENTO_API_KEY` outside code. `.env`, raw data, processed databases, and model artifacts are excluded by `.gitignore`. Provider terms govern data access and redistribution.

### Run order, from repository root

```powershell
# Optional, only if rebuilding US continuous data from existing raw bars:
python rollover.py

# Descriptive outputs:
python descriptive.py
python cluster.py

# Portfolios and candidate features:
python cluster_index.py
python calculate_features.py

# Availability-aligned targets and chronological splits:
python build_prediction_data.py

# Copper baseline:
python copper_ml.py --anchor copper

# Other three portfolios, then combined results:
python sector_ml.py

# Validation-based importance:
python feature_importance.py
```

A single extra portfolio can be run with `python copper_ml.py --anchor gold` (or `crude`, `soybeans`). The historical filename is retained, but its training implementation is shared.

The default alignment is equivalent to:

```powershell
python build_prediction_data.py --test-start 2025-10-01 --cn-delay-hours 24 --us-delay-minutes 5
```

Rebuild downstream results if data or definitions change. Scripts replace their own derived outputs. Importing `features.py` alone performs no data loading, fitting, or file writing.

| Output | Contents |
|---|---|
| `data/processed/cluster_indexes.duckdb` | Cluster membership, HRP weights, daily contributions, fallbacks and effective constituents |
| `data/processed/features.duckdb` | Product/portfolio features, coverage and descriptive correlations |
| `data/processed/prediction_data.duckdb` | Prediction rows, frozen targets, allowlist, CV membership, splits and timing assumptions |
| `data/processed/{anchor}_ml.duckdb` | Metrics, forecasts, CV scores, selected parameters, retained features and configuration |
| `data/processed/ml_models/{anchor}/` | Fitted joblib pipelines |
| `data/processed/feature_importance.duckdb` | Rankings, per-repeat effects and validation baselines |
| `results/` | Readable results and CSV tables |
| `plot_save/` | Figures organized by plot type |

Saved ML configurations record seed, software versions, dataset hash, sample counts, and fitting rules. Prediction timestamps/audits stay outside the explicit predictor allowlist.

Checks cover timing, cutoff-known weights/contracts, target weight normalization, label purging, matched model dates, ensemble arithmetic, training-only scaler statistics, and contribution reconciliation. These checks do not establish statistical or economic value.

## Presentation and course deliverables

The editable slides are in [presentation/overleaf](presentation/overleaf). Compile [main.tex](presentation/overleaf/main.tex) with pdfLaTeX. The source contains 14 main slides and backup slides, with the author's name, NetID, and October 15, 2026 date.

The course also requires a **separate written report** with at least five substantive pages, plus a cover and repository appendix, page numbers, headings, and formatted figures/tables. It must cover the six numbered sections above and avoid unformatted Python output. The slide deck is not a substitute.

Use `IE498MLF_Fall2026_My Name_project.pdf` for the report, replacing `My Name` with the student's name.

## References

- [Tushare futures daily fields](https://tushare.pro/document/2?doc_id=138)
- [Databento OHLCV conventions](https://databento.com/docs/schemas-and-data-formats/ohlcv)
- [SHFE trading hours](https://www.shfe.cn/eng/reports/CalendarHolidays/TradingHours/)
- [López de Prado (2016), HRP methodology](https://ssrn.com/abstract=2708678). This implementation uses average linkage and the documented anchor/fallback conventions.
- [Scikit-learn chronological validation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html). The project supplies its own saved chronological index pairs to grid search.
- [Scikit-learn permutation importance](https://scikit-learn.org/stable/modules/permutation_importance.html)
