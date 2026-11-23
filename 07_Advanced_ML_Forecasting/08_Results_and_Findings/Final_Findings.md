# Final findings — new official run

**Verdict:** Not Yet Ready for an unconditional all-nine-requirements-complete submission. The runnable research package is complete; the remaining blockers are missing source/provenance evidence and missing Package B final runner, not unexecuted seven-model experiments.

All seven methods executed at hourly/daily levels. The expanded finite search includes safe pooled same-clock history (distinct from the strict weekday/hour baseline),recent28/56-day comparable history,last comparable observation/age,exact calendar lags and trend ratios, plus five context ablations. It includes Package B's documented RF/XGBoost/LightGBM parameters, without copying its scores or same-day actual weather/IDs. The model comparison below is the official newly executed result.

| ForecastLevel | Model | FeatureSet | Observations | MAE | RMSE | WAPE | ValidationRank |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Daily | Random Forest | HistoryEnrollment | 43 | 82.522 | 135.998 | 42.244 | 1 |
| Daily | Historical Average | HistoricalOnly | 43 | 102.195 | 180.150 | 52.314 | 2 |
| Daily | XGBoost | HistoryEnrollment | 43 | 83.639 | 143.842 | 42.815 | 3 |
| Daily | LightGBM | HistoryEnrollment | 43 | 94.241 | 163.965 | 48.242 | 4 |
| Daily | Ridge | Full | 43 | 79.353 | 128.271 | 40.621 | 5 |
| Daily | Seasonal Naive | HistoricalOnly | 43 | 71.465 | 117.972 | 36.583 | 6 |
| Daily | Prophet | HistoryEnrollment | 43 | 97.809 | 170.229 | 50.069 | 7 |
| Hourly | Historical Average | HistoricalOnly | 280 | 18.342 | 27.032 | 61.699 | 1 |
| Hourly | LightGBM | HistoryEnrollment | 280 | 16.076 | 22.858 | 54.075 | 2 |
| Hourly | XGBoost | Full | 280 | 16.462 | 23.486 | 55.375 | 3 |
| Hourly | Random Forest | HistoryEnrollment | 280 | 17.060 | 24.207 | 57.387 | 4 |
| Hourly | Ridge | HistoryCalendar | 280 | 17.033 | 24.408 | 57.294 | 5 |
| Hourly | Seasonal Naive | HistoricalOnly | 280 | 14.768 | 21.040 | 49.676 | 6 |
| Hourly | Prophet | HistoryCalendar | 280 | 16.905 | 23.805 | 56.865 | 7 |

## Validation selection

| ForecastLevel | Model | ValidationMAE | MAE | RMSE | WAPE |
| --- | --- | --- | --- | --- | --- |
| Daily | Random Forest | 39.108 | 82.522 | 135.998 | 42.244 |
| Hourly | Historical Average | 9.405 | 18.342 | 27.032 | 61.699 |

Historical Average wins hourly validation (MAE9.405), narrowly ahead of LightGBM9.469. Random Forest wins daily validation (39.108), ahead of Historical Average39.652. These are the locked final choices, not the model with the lowest May score. Seasonal Naive remains best on common May observations at both levels. This signals weak generalization under the chosen split. A new historical-average or ML tuning round chosen after seeing May would contaminate the evaluation.

## Improvement or deterioration relative to Seasonal Naive

| ForecastLevel | Model | MAE | SeasonalNaiveMAE | MAEImprovementPct |
| --- | --- | --- | --- | --- |
| Daily | Random Forest | 82.522 | 71.465 | -15.472 |
| Hourly | Historical Average | 18.342 | 14.768 | -24.203 |

Positive percentages mean reduced MAE; negative means deterioration. Do not report WAPE as classification accuracy.

## Package differences supported by evidence

Both packages have identical2032hourly keys/targets. B's preserved output scores match its saved predictions; its WAPE is a fraction while the new report uses percent. Its lower hourly scores on280matched keys are approximatelyLightGBM10.880 andRF10.932; dailyXGBoost44.036 on43days. Those values are external historical references, not newly certified forecast-safe results.

B's45-feature manifest contains realized forecast-day weather,context keys,regular-class-day/date-gated enrollment and same-day event fields without archived as-of evidence. B's location is not an explicit numeric feature in that manifest. A explicitly encodes location and masks unavailable source context. In addition,the supplied HistoricalSameHour_Avg exactly matches a pooled location/clock-hour average across weekdays,not the professor's same-weekday/hour baseline. The new pipeline retains the strict baseline and adds the safe pooled statistic as a separately named ML feature. B's final runner is missing;its daily aggregation is therefore not directly auditable. Different library versions and preprocessing details also prevent claiming bitwise reproduction of B. The borrowed parameter settings are documented and executed with the new safe feature pipeline. No one-factor experiment quantitatively attributes the full score gap to any single issue.

## Daily approach

Dedicated daily forecasts and sums over observed hourly forecasts were compared on the same64validation location-days for all14approach/model combinations. SeeValidation_Daily_Approach_Comparison.csv. The predeclared reliability gate retains an independent recorded-day model because future full operating-hour coverage is unknown. The aggregated alternative is reported without claiming a complete operating day. This decision is locked before May.

## Explanation and detailed findings

### Hourly: Historical Average

| Location | Observations | MAE | RMSE | WAPE | P90AbsoluteError |
| --- | --- | --- | --- | --- | --- |
| Einstein's | 170 | 25.301 | 32.979 | 72.131 | 56.294 |
| Panda Express | 46 | 14.028 | 18.285 | 31.524 | 30.500 |
| Residential Dining | 64 | 2.959 | 8.237 | 60.302 | 3.821 |

Highest weekday MAE: Tuesday, 23.624 units; N=57.

Highest hour MAE: 10.0, 33.190 units; N=24.

### Daily: Random Forest

| Location | Observations | MAE | RMSE | WAPE | P90AbsoluteError |
| --- | --- | --- | --- | --- | --- |
| Einstein's | 18 | 161.492 | 205.080 | 48.650 | 306.353 |
| Panda Express | 10 | 46.653 | 58.150 | 22.702 | 98.167 |
| Residential Dining | 15 | 11.672 | 17.214 | 47.320 | 29.081 |

Highest weekday MAE: Tuesday, 112.998 units; N=8.

Selected hourly baseline uses equal-weight earlier same-weekday/hour demand. Selected daily RF and best advanced hourly LightGBM have validation permutation importance and SHAP charts. History,location,and scheduled activity can be associated with predictions; importance does not establish causality. Raw weekly counts are ungated by Spring Break and do not certify actual class meetings or attendance on every date. Their use is conditional on source availability.

## Remaining limitations and what resolves them

1. Missing timed/dated section export: the current588-row schedule has blank start/end times, so its69-row weekly enrollment profile cannot be reconstructed independently from that workbook. Historical publication snapshots are also missing. Do not treat scheduled registrations as measured attendance. Enrollment-assisted results are conditional retrospective experiments.
2. Outlet operating/closure calendar absent: daily target and hourly aggregation cover recorded hours, not proven full opening days.
3. Package B final run_ml_models.py is absent. Exact original daily feature aggregation/preprocessing and original training replay cannot be certified; the two documented parameter sets were independently rerun on safe features.
4. May was previously examined in project development. This run locked its choices before current May scoring but cannot turn May into a fresh prospective test.
5. No live weather/event ingestion or website end-to-end forecast integration is claimed. Prepared historical website tables and a batch inference contract are included.

## Requirements

| No | RequirementGroup | Status | EvidenceAndLimit |
| --- | --- | --- | --- |
| 1 | Official dataset / enrollment provenance | Partially satisfied | 2032 unique hourly rows,287 days,63891 units reconcile to ETL;588 section seat calculations and weekly profile alignment checked. Original class times and historical publication/as-of snapshots absent. |
| 2 | Predictor safety / Spring Break / cutoff | Partially satisfied | Explicit allowlist, strict prior-date histories, previous-day weather/events; break labels/proxies and IDs excluded. Enrollment snapshot and prior-day ingestion-time availability remain assumptions. |
| 3 | All seven forecasting approaches | Fully satisfied | Seven hourly and seven daily methods actually trained/evaluated;14 saved model/config reload checks; both B documented settings per tree family tested on safe features. |
| 4 | Hourly and daily forecasting / aggregation | Partially satisfied | Independent hourly/daily models;14-way aggregation-vs-direct validation comparison on64common recorded days. Full operating-day hour coverage cannot be verified without outlet calendar. |
| 5 | Chronological tuning and selection | Fully satisfied for historical benchmark | JanFeb fit,MarApr tune;selection locked before current May scoring. No May-driven tuning in this run. Prior May exposure explicitly disclosed; fresh prospective holdout still needed. |
| 6 | Overall / location / weekday / hour evaluation | Fully satisfied | 14 overall and623 grouped rows;shared280hourly/43daily May keys;counts,MAE,RMSE,WAPE,zero-denominator policy,error distributions independently checked. |
| 7 | Explainability and interpretation | Fully satisfied | Selected daily RF importance/SHAP;exact selected hourly baseline explanation;best advanced hourly LightGBM explained;association not causality. |
| 8 | A/B differences and reproducible borrowed settings | Partially satisfied | Same targets,matched B metrics independently recalculated;45B features audited;documented B settings rerun. Exact B end-to-end code unavailable:run_ml_models.py omitted from ZIP;original daily aggregation cannot be recovered exactly. |
| 9 | Runnable code / notebook / models / workbook / ZIP | Fully satisfied when final checks pass | New isolated environment;two complete runs compared;relative project paths,model reloads,inference checks,comparison workbook,andZIP integrity verified by accompanying evidence. |
ADDITIONAL CROSS-SOURCE CHECK
Timed meeting evidence exists in the current Tableau package. An audit-only copy is included; it was not substituted into training data. Comparing all69supplied weekly profile rows with recovered meetings matches7enrollment rows at exact clock times and9using any overlap in the hour;class-count matches are11and16respectively. This does not prove the supplied profile is wrong:effective dates,section inclusion and aggregation rules may differ. The exact weekly-profile generator and as-of metadata are needed to reconcile these sources. Thus the issue is incomplete profile provenance/reconstruction,not the total absence of all timed campus records. See06_Model_Evaluation/Enrollment_Cross_Source_Status.json.
