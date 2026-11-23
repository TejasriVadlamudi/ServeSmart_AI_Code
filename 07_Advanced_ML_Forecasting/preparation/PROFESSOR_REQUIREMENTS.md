# Professor requirement checklist

| Requirement | Status in this version |
|---|---|
| Preserve originals and create versioned Phase 2 folder | PASS |
| Confirm 2,032 rows and 63,891 units | PASS / checked by script |
| Observed-sales / observed-zero status | PASS |
| Keep missing hours absent | PASS |
| Keep CampusClosedFlag separate from outlet status | PASS |
| Document one negative target | PASS |
| Remove SpringBreak/Spring_Break from predictors; retain dates | PASS |
| Inspect student activity predictors and forecast-time availability | PASS; manifest, availability register, and README |
| Seasonal Naive baseline | PASS; executed |
| Historical Same-Hour Average baseline | PASS; executed |
| LightGBM | BLOCKED by missing `lightgbm` |
| Prophet | BLOCKED by missing `prophet` |
| Ridge / Random Forest / XGBoost tuning | BLOCKED by missing optional ML libraries; reusable paths documented |
| Hourly and daily predictions | PASS for executed baselines |
| MAE, RMSE, WAPE overall and by location/weekday/hour | PASS for executed baselines where group rows exist |
| Coverage and failures | PASS; output files record executed/blocked status |
| Final Excel model comparison | Generated from outputs after run |
| Streamlit/Tableau changes | NOT STARTED by request |
