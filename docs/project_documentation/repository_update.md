# Repository forecasting update

The obsolete app-layer baseline and prior Phase 2 forecasting artifacts were removed from the current tree and replaced with the consolidated Ready-to-Submit forecasting package. The pre-update commit remains in Git history, and the local tag `pre-authoritative-forecasting-update` points to it for recovery.

Unrelated Streamlit presentation, ETL, Tableau, assets, queue data, campus closures, and operating-hours files were preserved. The app now has one forecasting source of truth: `data/forecasting/`.

The dashboard labels hourly Random Forest and daily XGBoost as the recovered selected models, displays their verified WAPE metrics, loads their website backtests, and carries the forecast source into its planning output. The recovery record preserves the exact common-key verification from commit `64138acf`.
