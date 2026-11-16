from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_metrics(actual, predicted) -> dict[str, float]:
    y_true = np.asarray(actual, dtype=float)
    y_pred = np.asarray(predicted, dtype=float)
    error = y_true - y_pred
    denominator = np.abs(y_true).sum()
    return {
        "Observations": int(len(y_true)),
        "MAE": float(np.mean(np.abs(error))) if len(y_true) else np.nan,
        "RMSE": float(np.sqrt(np.mean(error ** 2))) if len(y_true) else np.nan,
        "WAPE (%)": float(np.abs(error).sum() / denominator * 100) if denominator else np.nan,
    }


def compare_predictions(frame: pd.DataFrame, actual_column: str, prediction_columns: dict[str, str]) -> pd.DataFrame:
    rows = []
    for name, column in prediction_columns.items():
        scored = frame[[actual_column, column]].dropna()
        rows.append({"Model": name, **calculate_metrics(scored[actual_column], scored[column])})
    return pd.DataFrame(rows)
