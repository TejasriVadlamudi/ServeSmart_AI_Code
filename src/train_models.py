from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from forecasting_data import APPROVED_ML_FEATURES, chronological_split, load_ml_flat_file, safe_features
from metrics import calculate_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ServeSmart forecasting models on an approved ML flat file.")
    parser.add_argument("--approved-ml-file", required=True, help="Path to the approved ML flat file")
    parser.add_argument("--output-dir", default="outputs/models")
    args = parser.parse_args()
    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder
    except ImportError as exc:
        raise SystemExit("Install requirements.txt before training: scikit-learn is required.") from exc

    frame = chronological_split(load_ml_flat_file(args.approved_ml_file))
    features = safe_features(frame)
    if features != APPROVED_ML_FEATURES:
        raise SystemExit("The approved ML file does not contain the complete 22-feature handoff schema.")
    categorical = [c for c in features if frame[c].dtype == "object"]
    numeric = [c for c in features if c not in categorical]
    preprocessor = ColumnTransformer([("num", SimpleImputer(strategy="median"), numeric), ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical)])
    model = Pipeline([("preprocess", preprocessor), ("model", RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1))])
    train = frame[frame["DataSplit"].eq("Train") & frame["Target_NegativeFlag"].ne(1)].copy()
    test = frame[frame["DataSplit"].eq("Test")]
    model.fit(train[features], train["Target_HourlyUnits"])
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    import joblib
    joblib.dump(model, out / "random_forest_pipeline.joblib")
    rf_predictions = model.predict(test[features])
    prediction_rows = test[["Location", "Date", "Hour", "DataSplit", "Target_HourlyUnits"]].copy()
    prediction_rows["Random Forest"] = rf_predictions
    comparison = [{"Model": "Random Forest", **calculate_metrics(test["Target_HourlyUnits"], rf_predictions)}]
    try:
        from xgboost import XGBRegressor
        xgb = Pipeline([("preprocess", preprocessor), ("model", XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, objective="reg:squarederror", random_state=42, n_jobs=2))])
        xgb.fit(train[features], train["Target_HourlyUnits"])
        joblib.dump(xgb, out / "xgboost_pipeline.joblib")
        xgb_predictions = xgb.predict(test[features])
        prediction_rows["XGBoost"] = xgb_predictions
        comparison.append({"Model": "XGBoost", **calculate_metrics(test["Target_HourlyUnits"], xgb_predictions)})
    except ImportError:
        (out / "xgboost_status.txt").write_text("XGBoost not installed; Random Forest output is still available.\n", encoding="utf-8")
    pd.DataFrame(comparison).to_csv(out / "model_comparison.csv", index=False)
    prediction_rows.to_csv(out / "advanced_test_predictions.csv", index=False)
    (out / "training_manifest.json").write_text(json.dumps({"approved_file": str(args.approved_ml_file), "features": features, "target": "Target_HourlyUnits", "split": "DataSplit labels from approved handoff", "negative_targets_excluded_from_fit": True, "train_rows_used": int(len(train)), "test_rows_scored": int(len(test))}, indent=2), encoding="utf-8")
    print(f"Saved Random Forest pipeline and manifest to {out}")


if __name__ == "__main__":
    main()
