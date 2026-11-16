from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from metrics import calculate_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare approved baselines and advanced models on identical test rows.")
    parser.add_argument("--baseline-file", required=True)
    parser.add_argument("--advanced-file", required=True)
    parser.add_argument("--output-file", default="outputs/models/model_comparison_common_test.csv")
    args = parser.parse_args()

    baseline = pd.read_csv(args.baseline_file)
    advanced = pd.read_csv(args.advanced_file)
    keys = ["Location", "Date", "Hour"]
    baseline = baseline.rename(columns={"location": "Location", "date": "Date", "hour": "Hour", "target": "Target_HourlyUnits"})
    baseline["Date"] = baseline["Date"].astype(str)
    advanced["Date"] = pd.to_datetime(advanced["Date"]).dt.strftime("%Y-%m-%d")
    test_baseline = baseline[baseline["split"].eq("test") & baseline["common_scored_row"].eq(True)].copy()
    merged = test_baseline.merge(advanced, on=keys, how="inner", suffixes=("_baseline", "_advanced"))
    merged = merged.dropna(subset=["Target_HourlyUnits_baseline", "seasonal_naive", "historical_hour_mean", "Random Forest", "XGBoost"])
    actual = merged["Target_HourlyUnits_baseline"]
    rows = []
    for model, column in [
        ("Seasonal Naive", "seasonal_naive"),
        ("Historical Hour Mean", "historical_hour_mean"),
        ("Random Forest", "Random Forest"),
        ("XGBoost", "XGBoost"),
    ]:
        rows.append({"Model": model, "Scope": "May test common rows", "Observations": len(merged), **calculate_metrics(actual, merged[column])})
    result = pd.DataFrame(rows).sort_values(["WAPE (%)", "RMSE", "MAE"]).reset_index(drop=True)
    output = Path(args.output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
