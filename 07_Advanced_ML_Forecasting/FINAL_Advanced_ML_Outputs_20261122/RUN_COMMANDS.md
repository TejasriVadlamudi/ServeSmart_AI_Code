# ServeSmart Phase 2 ML execution record

Source archive was copied into this versioned folder and its original source files were not changed.

```bash
python3 -V
python3 -c "import xgboost; print(xgboost.__version__)"
python3 -c "import lightgbm; print(lightgbm.__version__)"
/opt/homebrew/bin/brew list --versions libomp
/usr/bin/python3 -m venv --system-site-packages .venv
.venv/bin/python -c "import sys,platform,xgboost,lightgbm; print(sys.executable); print(platform.machine()); print('xgboost',xgboost.__version__); print('lightgbm',lightgbm.__version__)"
.venv/bin/python run_ml_models.py
/Users/teju/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node build_final_workbook.mjs
```

The ML runner uses the copied flat file, fits Random Forest, XGBoost, and LightGBM on Train, selects parameters on the chronological Validation split, refits on Train plus Validation, and scores the shared eligible Test rows at hourly and daily levels. It writes the CSV metrics, predictions, tuning record, dependency status, and final workbook in `outputs/`.
