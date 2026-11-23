# ServeSmart finalized forecasting results

## Authoritative experiment

The finalized Phase 2 experiment is the executed run in:

`07_Advanced_ML_Forecasting/FINAL_Advanced_ML_Outputs_20261122/outputs/model_metrics.csv`

| Forecast level | Selected model | WAPE | MAE | RMSE | Observations |
|---|---|---:|---:|---:|---:|
| Hourly | Random Forest | 37.37% | 9.93 | 13.92 | 316 |
| Daily | XGBoost | 22.54% | 44.04 | 59.77 | 43 |

The source handoff contains 2,032 hourly rows and 63,891 units. The finalized run preserves one negative target row, excludes it from supervised fitting, and evaluates the May common eligible test rows.

## Why older metrics differ

The repository already contained an earlier app-layer comparison under `outputs/models/`. That comparison reports different values because it used a separate baseline/advanced-model artifact and a different common-row coverage. It is retained rather than silently deleted so the project history remains auditable.

Use the finalized Phase 2 `model_metrics.csv` for the capstone result. Use the older `outputs/models/` files only as historical comparison material.

## Reproducibility files

- Preparation scripts: `07_Advanced_ML_Forecasting/preparation/`
- Validation outputs: `07_Advanced_ML_Forecasting/validation/`
- Final predictions and manifests: `07_Advanced_ML_Forecasting/FINAL_Advanced_ML_Outputs_20261122/outputs/`
- Final comparison workbook: `07_Advanced_ML_Forecasting/FINAL_Advanced_ML_Outputs_20261122/ServeSmart_Phase2_Final_ML_Comparison.xlsx`
