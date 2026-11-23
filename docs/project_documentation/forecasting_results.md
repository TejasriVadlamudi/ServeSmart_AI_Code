# ServeSmart forecasting results

The authoritative results are in `07_Advanced_ML_Forecasting/06_Model_Evaluation/Overall_Model_Comparison.csv` and the app-facing copy is in `data/forecasting/Overall_Model_Comparison.csv`.

| Forecast level | Selected model | WAPE | Evaluation rows |
|---|---|---:|---:|
| Hourly | Historical Average | 61.6987% | 280 |
| Daily | Random Forest | 42.2436% | 43 |

The integrated website backtest files are `data/forecasting/Hourly_Website_Backtest_Data.csv` and `data/forecasting/Daily_Website_Backtest_Data.csv`. They are historical May 2026 benchmark predictions with an explicit forecast cutoff; they must not be described as prospective live forecasts.

The app uses the hourly Historical Average rows when an exact backtest date is available. For dates beyond the scored history, it uses a rolling historical fallback based on prior same-weekday/hour observations and labels that source in the UI. Daily Random Forest predictions appear in Analytics as the daily backtest view.

The ZIP's own findings and remaining-blocker files are included under `07_Advanced_ML_Forecasting/08_Results_and_Findings/`. In particular, the package does not support an unconditional claim that all nine professor requirements are complete.
