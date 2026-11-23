# Requirement status

Fulfilled in this run:
- Separate baseline and advanced project folders with input, code, outputs, and reports.
- Hourly and daily actual/prediction files.
- MAE, RMSE, WAPE with zero-denominator handling and breakdowns by location, weekday, and hour.
- Enrollment-related fields checked and retained when present; Spring Break fields excluded from ML predictors.
- Chronological Test holdout and fair common eligible comparison.
- Seasonal Naive, Historical Same-Hour Average, Ridge, Random Forest, XGBoost, and daily SARIMA.

Not run in this environment:
- LightGBM: Python package unavailable; status is recorded as skipped.
- Prophet: Python package unavailable; status is recorded as skipped.
- Hourly SARIMA: skipped because the source has irregular, location-specific operating hours; daily SARIMA is included.

These are execution/environment gaps, not invented results. Install the missing packages and rerun the supplied scripts to add those models.
