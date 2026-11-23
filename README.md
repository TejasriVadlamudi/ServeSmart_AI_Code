# ServeSmart AI Forecasting Application

This repository contains the ServeSmart AI capstone application, corrected Phase 1 ETL deliverables, and the finalized Phase 2 forecasting experiment. Existing application files and older comparison outputs are preserved for reproducibility.

## Final forecasting results

The authoritative finalized Phase 2 run is stored in `07_Advanced_ML_Forecasting/FINAL_Advanced_ML_Outputs_20261122/`.

| Forecast level | Selected model | WAPE | Test rows |
|---|---|---:|---:|
| Hourly | Random Forest | 37.37% | 316 |
| Daily | XGBoost | 22.54% | 43 |

These results use the common eligible test rows from the finalized run. The older comparison in `outputs/models/` is intentionally retained and labeled as an earlier app-layer experiment; it used a different comparison artifact and evaluation coverage, so its metrics should not be mixed with the finalized Phase 2 results.

## Project deliverables

- `04_ETL_Pipeline/phase1_corrected_file/`: corrected Phase 1 ETL script and verification notes.
- `07_Advanced_ML_Forecasting/preparation/`: reusable Phase 2 preparation scripts and requirements.
- `07_Advanced_ML_Forecasting/validation/`: input-contract, predictor, data-quality, and dependency checks.
- `07_Advanced_ML_Forecasting/FINAL_Advanced_ML_Outputs_20261122/`: finalized model metrics, predictions, manifests, and comparison workbook.
- `docs/project_documentation/`: concise documentation of results, evaluation scope, and reproducibility.

## What is included

- `src/forecasting_data.py`: approved-file loading, schema checks, chronological splits, and deployment-safe predictors.
- `src/metrics.py`: MAE, RMSE, and WAPE calculations plus model comparison tables.
- `src/train_models.py`: reusable Random Forest and optional XGBoost scaffold. It requires an approved ML flat file and never trains on the demo path.
- `src/decision_support.py`: configurable preparation, inventory, and staffing signals. These are planning prompts, not exact purchasing or staffing orders.
- `app.py`: Streamlit dashboard with Home, Customer Portal, Manager Portal, and Analytics views.
- `outputs/baselines/approved/`: Sanjana/Thanishka's approved baseline outputs and verification files.
- `outputs/models/model_comparison_common_test.csv`: common-row May comparison used to select the current baseline winner.
- `data/queue_service_times.csv`: optional operational input with `Date`, `Hour`, `Station`, and `WaitMinutes` columns.
- `data/campus_closures.csv`: optional future closure calendar with `Date` and `Reason` columns.
- `assets/servesmart-hero.png`: landing-page illustration used by the Streamlit Home view.
- `tests/test_core.py`: dependency-light smoke tests for the core logic.

## Run locally

```bash
python3 -m pip install -r requirements.txt
python3 -m pytest -q
streamlit run app.py
```

The app uses `data/ServeSmart_ML_Flat_File_Updated.csv` and the approved baseline
forecasts in `outputs/baselines/approved/all_hour_forecasts.csv` when those files
are present. The current common-row comparison selects Seasonal Naive as the
strongest model for the May test period.

The Customer Portal now accepts a forecast date. For dates beyond the loaded
file, it estimates each station/hour from the most recent four matching
weekday/hour observations, falling back to recent same-hour history. Sundays,
explicit campus-closed dates, and unsupported station hours show `Closed` or
`No forecast` rather than fabricated demand. If `data/queue_service_times.csv`
is populated, exact queue/service-time values are used; until then, wait values
marked with `*` are directional demand estimates rather than measured queue
times.

The demo Manager Portal is intentionally search-first: it shows today's predicted
sales by station and only displays historical item-level results after a manager
asks a question. The Customer Portal shows today's estimated wait by station.

## Train advanced models for comparison

```bash
python3 src/train_models.py \
  --approved-ml-file /path/to/approved_ML_flat_file.csv \
  --output-dir outputs/models
```

The training command expects at minimum `Location`, `Date`, `Hour`, and
`Target_HourlyUnits`. It sorts by time and evaluates on a later test period.
It intentionally excludes current-period sales/transaction fields from the
default feature list to avoid leakage. XGBoost is optional; Random Forest is
the required scaffold model. To compare the advanced outputs with the approved
baselines on identical May test rows:

```bash
python3 src/compare_models.py \
  --baseline-file outputs/baselines/approved/all_hour_forecasts.csv \
  --advanced-file outputs/models/advanced_test_predictions.csv
```

## Important limitations

- The current dashboard uses the approved baseline comparison; customer wait times remain directional estimates until queue and service-time data are available.
- Recommendations use configurable thresholds and recent demand patterns; they do not claim exact staffing, purchasing, or inventory quantities.
- Weather, event, and campus-context values must be known or forecastable at the time a real forecast is produced.
- Model quality, calibration, and operational adoption still require review with the project team.
