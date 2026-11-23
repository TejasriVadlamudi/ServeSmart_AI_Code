"""Optional advanced Phase 2 model implementations.

These functions fail loudly with the exact missing dependency when the optional
library is unavailable. They are intentionally separate from run_phase2.py so
the baseline/reporting workflow remains runnable in a minimal environment.
"""
from __future__ import annotations

import itertools
import numpy as np


def time_aware_grid(param_grid):
    """Return deterministic parameter combinations for chronological tuning."""
    keys = list(param_grid)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(param_grid[k] for k in keys))]


def tune_sklearn_estimator(estimator_factory, candidates, x_train, y_train, x_valid, y_valid, metric):
    """Fit candidates on earlier data and select the lowest validation metric."""
    rows=[]
    for params in candidates:
        model=estimator_factory(params)
        model.fit(x_train, y_train)
        pred=model.predict(x_valid)
        rows.append({"Parameters":params,"ValidationMetric":float(metric(y_valid,pred))})
    return min(rows, key=lambda r:r["ValidationMetric"]), rows


def build_ridge(params):
    from sklearn.linear_model import Ridge
    return Ridge(**params)


def build_random_forest(params):
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(random_state=42, n_jobs=-1, **params)


def build_xgboost(params):
    from xgboost import XGBRegressor
    return XGBRegressor(random_state=42, n_jobs=-1, objective="reg:squarederror", **params)


def build_lightgbm(params):
    from lightgbm import LGBMRegressor
    return LGBMRegressor(random_state=42, verbosity=-1, **params)


def fit_prophet(train_frame, forecast_frame, params=None):
    """Fit one Prophet model using ds/y columns and return forecast rows."""
    from prophet import Prophet
    params = params or {}
    model=Prophet(**params)
    model.fit(train_frame[["ds","y"]])
    return model, model.predict(forecast_frame[["ds"]])


def dependency_error(model_name):
    """Probe the model library and return a structured status instead of guessing."""
    packages={"Ridge":"sklearn","Random Forest":"sklearn","XGBoost":"xgboost","LightGBM":"lightgbm","Prophet":"prophet"}
    package=packages[model_name]
    try:
        __import__(package)
        return {"Model":model_name,"Status":"available","Detail":""}
    except Exception as exc:
        return {"Model":model_name,"Status":"blocked","Detail":f"{type(exc).__name__}: {exc}"}
