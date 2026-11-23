# 07 — Advanced ML Forecasting

Use the folders below by purpose. Do not mix their outputs.

## Presentation / final result

`FINAL_Advanced_ML_Outputs_20261122/`

This is the newest executed advanced-ML run. Its `outputs/` contains the Random Forest, XGBoost, and LightGBM results for hourly and daily forecasting. Use its `model_metrics.csv`, `predictions_test.csv`, `validation_tuning.csv`, and `ServeSmart_Phase2_Final_ML_Comparison.xlsx` for the presentation.

## Supporting preparation run

`SUPPORT_Phase2_Preparation_20261122/`

This contains the newer input contract, validation, predictor manifest, baseline preparation, reusable scripts, and dependency records. Its `run_summary.json` says the advanced models were not executed in that run because optional dependencies were unavailable. Do not use its baseline-only metrics as the final advanced-ML result.

## Supporting notebook

`SUPPORT_Notebooks/` contains the forecasting notebook used as supporting documentation.

The earlier packaged advanced-ML folder was moved to `99_Original_Extracted_Archives/legacy_advanced_ml_packaged_20261122/` for reproducibility. It is not the canonical result.
