# Repository forecasting update

The obsolete app-layer baseline and prior Phase 2 forecasting artifacts were removed from the current tree and replaced with the consolidated Ready-to-Submit forecasting package. The pre-update commit remains in Git history, and the local tag `pre-authoritative-forecasting-update` points to it for recovery.

Unrelated Streamlit presentation, ETL, Tableau, assets, queue data, campus closures, and operating-hours files were preserved. The app now has one forecasting source of truth: `data/forecasting/`.

The dashboard labels hourly Historical Average and daily Random Forest as the selected models, displays their WAPE metrics, loads their website backtests, and no longer references the removed Seasonal Naive fallback artifacts. Decision support accepts the selected forecast values and carries the forecast source into its planning output.
