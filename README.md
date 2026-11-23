# ServeSmart AI

ServeSmart is a Streamlit application for campus-dining demand visibility and directional operational planning. The repository keeps the corrected ETL pipeline, customer and manager portals, Tableau asset, and the consolidated forecasting research package together.

## Current forecasting package

The authoritative forecasting package is under `07_Advanced_ML_Forecasting/` and was integrated from `ServeSmart_Advanced_Forecasting_Ready_To_Submit.zip`.

| Forecast level | Validation-selected model | May backtest WAPE | Common May rows |
|---|---|---:|---:|
| Hourly | Random Forest | 35.45% | 280 |
| Daily | XGBoost | 29.04% | 43 |

These are WAPE values for the audited `64138acf` historical May handoff, not classification accuracy. Seasonal Naive remains the stronger benchmark on these common rows at both levels (49.68% hourly and 36.58% daily); the older 37.37% and 22.54% figures are not used as final claims.

The Streamlit app reads the website-ready files in `data/forecasting/`:

- `Approved_ML_Flat_File.csv` — approved historical feature and target rows.
- `Hourly_Website_Backtest_Data.csv` — hourly Random Forest backtest predictions.
- `Daily_Website_Backtest_Data.csv` — daily XGBoost backtest predictions.
- `Overall_Model_Comparison.csv` — selected models and evaluation metrics.
- `Model_Registry.json` — model and feature-set registry.

The May files are historical benchmark data. For a future date with no scored row, the customer and manager portals use a clearly labeled rolling historical fallback based only on prior observations. Missing schedules, closures, and unavailable history still produce `Closed`, `No forecast`, or `Unavailable` rather than fabricated demand.

## Repository areas

- `app.py` — Streamlit Home, Customer Portal, Manager Portal, and Analytics views.
- `src/forecasting_data.py` — approved-file loading, schema checks, chronological splits, and deployment-safe predictors.
- `src/decision_support.py` — directional food-preparation, inventory, and staffing prompts driven by `ForecastUnits` and the selected forecast source.
- `04_ETL_Pipeline/phase1_corrected_file/` — corrected Phase 1 ETL source.
- `07_Advanced_ML_Forecasting/` — consolidated forecasting code, selected models, predictions, evaluation, explainability, documentation, and limitations.
- `data/forecasting/` — small app-facing forecasting handoff.

## Run locally

```bash
python3 -m pip install -r requirements.txt
streamlit run app.py
```

Run the automated checks with:

```bash
pytest -q
```

The forecasting package's own reproducibility and verification instructions are in `07_Advanced_ML_Forecasting/README.md` and `07_Advanced_ML_Forecasting/OPEN_THIS_FIRST.txt`.

## Scope and limitations

The authoritative package records a strict “not yet ready for an unconditional all-nine-requirements-complete claim” verdict because timed enrollment sources, historical availability evidence, an outlet operating calendar, and Package B's omitted final runner remain documented blockers. Those limitations are preserved rather than hidden.

Queue wait times remain directional unless the optional `data/queue_service_times.csv` feed is populated. Forecast values represent recorded POS units, not customers or attendance, and decision-support messages are planning prompts rather than exact purchasing or staffing orders.
