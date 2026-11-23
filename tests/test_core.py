import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from decision_support import build_recommendations
from forecasting_data import chronological_split, load_ml_flat_file, safe_features
from metrics import calculate_metrics, compare_predictions


def test_load_and_safe_features():
    path = Path(__file__).parents[1] / "data/forecasting/Approved_ML_Flat_File.csv"
    frame = load_ml_flat_file(path)
    assert len(frame) == 2032
    assert "TransactionCount" not in safe_features(frame)
    assert len(safe_features(frame)) == 22


def test_chronological_split_has_all_periods():
    frame = pd.DataFrame({"Location": ["A"] * 10, "Date": pd.date_range("2026-01-01", periods=10), "Hour": range(10), "Target_HourlyUnits": range(10)})
    split = chronological_split(frame)
    assert set(split["DataSplit"]) == {"Train", "Validation", "Test"}
    assert split.loc[split["DataSplit"].eq("Train"), "Date"].max() < split.loc[split["DataSplit"].eq("Test"), "Date"].min()


def test_metrics_and_rules():
    assert calculate_metrics([1, 3], [1, 1])["MAE"] == 1.0
    frame = pd.DataFrame({"Location": ["A", "A"], "Date": pd.date_range("2026-01-01", periods=2), "Hour": [10, 11], "ForecastUnits": [2, 20]})
    comparison = compare_predictions(pd.DataFrame({"y": [1, 2], "p": [1, 3]}), "y", {"demo": "p"})
    assert comparison.loc[0, "Model"] == "demo"
    recs = build_recommendations(frame)
    assert len(recs) == 2 and "Planning note" in recs.columns
