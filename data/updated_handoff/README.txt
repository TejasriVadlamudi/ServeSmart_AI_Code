ServeSmart Final ML Handoff

FINAL ML FLAT FILES
data/ServeSmart_Final_Hourly_ML.csv
  One row per location-date-hour; 2,032 rows.
  Includes original source fields, daily peak student activity,
  and hourly fitting eligibility.
  Timestamp preserves both the date and hour.

data/ServeSmart_Final_Daily_ML.csv
  One row per location-date; 287 rows.
  Daily targets sum recorded hourly quantities, including adjustments.

MODEL INPUTS
models/Selected_Model_Inputs.csv lists the exact predictor order.
Other flat-file columns must not automatically become model inputs.
RecordedHours describes observed coverage and is not a predictor.

DASHBOARD PREDICTIONS
Separate files contain 316 hourly and 43 daily May predictions.
These are historical benchmark predictions, not forecasts for new dates.
Actual contains recorded quantities.
Predicted contains the validation-selected model's predictions.
Missing baseline predictions remain blank.
CommonBaselineEligible identifies the shared comparison sample.
SignedError = Predicted - Actual.
AbsolutePercentageError is blank when Actual is zero.

SELECTED MODELS
Hourly: Random Forest with student features.
Daily: XGBoost with student features.
Selection used March-April MAE before May benchmark evaluation.
Saved pipelines were fitted on January-April observations.
New-date forecasting requires the corresponding historical inputs
and applicable calendar and schedule information.

DATA HANDLING
Keep hourly and daily demand in separate tables.
Do not sum daily totals repeated across hourly records.
Missing dining hours or dates are not automatically zero demand.
Student counts represent scheduled enrollment, not measured attendance.
Daily peak enrollment is not a count of unique daily students.
Historical demand inputs exclude the forecast date and future dates.
Negative hourly targets are retained but excluded from hourly fitting.
Hourly and daily predictions are independent and need not reconcile.

EXPORT FORMAT
Date fields use YYYY-MM-DD.
Hourly Timestamp uses YYYY-MM-DD HH:MM:SS.
Floating-point values are exported with sufficient precision to
preserve the stored numerical values when read back.

SUPPORTING FILES
Evaluation tables document validation and May performance.
Grouped May metrics cover location, weekday, and hour where applicable.
Model metadata records selected settings and package versions.
Share the completed notebook separately for methods and visualizations.
