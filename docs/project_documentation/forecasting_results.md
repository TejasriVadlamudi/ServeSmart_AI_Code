# ServeSmart forecasting results

The authoritative results are in `07_Advanced_ML_Forecasting/06_Model_Evaluation/Overall_Model_Comparison.csv` and the app-facing copy is in `data/forecasting/Overall_Model_Comparison.csv`.

| Forecast level | Selected model | WAPE | Evaluation rows |
|---|---|---:|---:|
| Hourly | Random Forest | 35.4479% | 280 |
| Daily | XGBoost | 29.0414% | 43 |

The integrated website backtest files are `data/forecasting/Hourly_Website_Backtest_Data.csv` and `data/forecasting/Daily_Website_Backtest_Data.csv`. They are historical May 2026 benchmark predictions with an explicit forecast cutoff; they must not be described as prospective live forecasts.

These recovered selected-model results reproduce 35.4479% hourly WAPE on 280 common rows and 29.0414% daily WAPE on 43 common rows. The older 37.37% and 22.54% figures are not reproducible aggregate WAPE values and are not used.

The app uses the hourly Random Forest rows when an exact backtest date is available. For dates beyond the scored history, it uses a clearly labeled historical fallback because the old model’s required future feature snapshot is not available. Daily XGBoost predictions appear in Analytics as the daily backtest view.

The ZIP's own findings and remaining-blocker files are included under `07_Advanced_ML_Forecasting/08_Results_and_Findings/`. In particular, the package does not support an unconditional claim that all nine professor requirements are complete.
