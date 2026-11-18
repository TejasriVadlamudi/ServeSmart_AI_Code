"""Export the fair four-method forecasting comparison for reporting."""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILE = ROOT / "outputs" / "models" / "model_comparison_common_test.csv"
OUTPUT_FILE = ROOT / "outputs" / "models" / "four_method_comparison.csv"

EXPECTED_METHODS = {
    "Seasonal Naive",
    "Historical Hour Mean",
    "XGBoost",
    "Random Forest",
}


def export_results() -> pd.DataFrame:
    """Save the common-test results for all four forecasting methods."""
    comparison = pd.read_csv(SOURCE_FILE)
    comparison = comparison[
        ["Model", "Scope", "Observations", "MAE", "RMSE", "WAPE (%)"]
    ].copy()

    found_methods = set(comparison["Model"])
    missing_methods = EXPECTED_METHODS - found_methods
    if missing_methods:
        raise ValueError(f"Missing forecasting methods: {sorted(missing_methods)}")

    comparison["MAE"] = comparison["MAE"].round(2)
    comparison["RMSE"] = comparison["RMSE"].round(2)
    comparison["WAPE (%)"] = comparison["WAPE (%)"].round(2)
    comparison = comparison.sort_values("MAE").reset_index(drop=True)
    comparison.insert(0, "Rank", range(1, len(comparison) + 1))

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(OUTPUT_FILE, index=False)
    return comparison


if __name__ == "__main__":
    result = export_results()
    print(result.to_string(index=False))
    print(f"\nSaved results to: {OUTPUT_FILE}")
