# ServeSmart AI — Phase 2 Forecasting (versioned 2026-11-22)

This folder is a versioned Phase 2 workspace. The original uploaded CSV and notebook are preserved in `inputs/` and are not edited.

## Confirmed input contract

- Grain: `Location + Date + Hour`
- Rows: 2,032
- Target: `Target_HourlyUnits`
- Target total: 63,891 units
- Historical dates are retained; `SpringBreak` and `Spring_Break` are excluded from predictor features.
- Missing/unconfirmed hours remain absent; they are not converted to zero.
- `HourStatus` distinguishes `observed_sales`, `observed_zero`, and `observed_negative_adjustment`.
- `CampusClosedFlag` is retained as campus context and is not used as outlet operating status.
- The one negative target (Residential Dining, 2026-02-17 14:00, -1) is preserved in the prepared data and excluded from hourly fitting through `HourlyFitEligible`.

## Run

```bash
python run_phase2.py
```

The script always runs validation, data-status preparation, predictor manifest creation, Seasonal Naive (7-day), and Historical Same-Hour Average baselines for hourly and daily forecasts. It writes metrics, predictions, checks, dependency status, and a JSON run summary under `outputs/` and logs under `logs/`.

The requested Ridge, Random Forest, XGBoost, LightGBM, and Prophet implementations are in `advanced_models.py` with deterministic time-aware grid tuning helpers. This environment currently lacks the required libraries, so those models are not scored here. The exact blockers are recorded in `outputs/dependency_status.csv`. The code is reusable in an environment with the dependencies installed.

## Predictor availability

The flat file includes scheduled/student-activity fields (`EstimatedStudentsOnCampus`, `ScheduledStudents_DailyPeak`) and historical demand fields. The predictor manifest is generated from numeric fields that are present before the forecast cutoff. Transaction outcomes and target-derived fields, including `UniqueItems`, are excluded. `predictor_availability.csv` labels each candidate as prior-history, student/schedule, calendar/context, weather-dependent, or excluded post-outcome. No literal enrollment column exists; the student fields are estimates/scheduled activity.

## Evaluation policy

The test period is the source `DataSplit == Test` period. Metrics are calculated on common eligible rows for each model and separately by location, weekday, and hour where applicable. Reported metrics are MAE, RMSE, and WAPE; model failures and coverage are recorded rather than imputed.

## Limitations

This package does not modify Streamlit or Tableau. Prophet and advanced tree-model execution require installation of their optional dependencies. The supplied source has recorded location-hours only, so no complete hourly calendar is invented.
