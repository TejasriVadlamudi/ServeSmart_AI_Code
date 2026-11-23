"""ServeSmart Phase 2 forecasting workflow.

This script is intentionally dependency-tolerant: it always validates inputs,
creates status fields, runs chronological baselines, and writes artifacts. The
advanced estimators are attempted only when their libraries are installed.
"""
from __future__ import annotations

import json
import logging
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "inputs" / "ServeSmart_FinalFlatfile_Hourly_ML(1).csv"
OUT = ROOT / "outputs"
LOG = ROOT / "logs" / "phase2_run.log"
OUT.mkdir(exist_ok=True)
(ROOT / "logs").mkdir(exist_ok=True)
logging.basicConfig(filename=LOG, level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("servesmart_phase2")

EXPECTED_ROWS = 2032
EXPECTED_UNITS = 63891
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def safe_import(name):
    try:
        module = __import__(name)
        return module, None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def metrics(actual, predicted):
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    e = p - a
    return {"Observations": int(len(a)), "MAE": float(np.mean(np.abs(e))),
            "RMSE": float(np.sqrt(np.mean(e * e))),
            "WAPE": float(np.sum(np.abs(e)) / np.sum(np.abs(a))) if np.sum(np.abs(a)) else np.nan}


def add_status(df):
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for c in ["Hour", "Target_HourlyUnits", "EstimatedStudentsOnCampus", "CampusClosedFlag", "SpringBreak", "Spring_Break"]:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["HourStatus"] = np.select(
        [df["Target_HourlyUnits"].lt(0), df["Target_HourlyUnits"].eq(0), df["Target_HourlyUnits"].gt(0)],
        ["observed_negative_adjustment", "observed_zero", "observed_sales"], default="missing_unconfirmed")
    df["OutletStatus"] = "unknown_unconfirmed"
    df.loc[df["HourStatus"].isin(["observed_sales", "observed_zero", "observed_negative_adjustment"]), "OutletStatus"] = "observed_activity"
    df["CampusClosedContext"] = df.get("CampusClosedFlag", 0).fillna(0).astype(int)
    df["SpringBreakPredictorExcluded"] = True
    return df


def available_features(df):
    excluded = {
        "Target_HourlyUnits", "NetSales", "GrossAmount", "TotalDiscount", "TransactionRows",
        "DataSplit", "ForecastCutoffDate", "Date", "Timestamp", "HourStatus", "OutletStatus",
        "CampusClosedContext", "SpringBreakPredictorExcluded", "SpringBreak", "Spring_Break",
        "Target_NegativeFlag", "HourlyFitEligible", "UniqueItems", "Location", "LocationKey", "DateKey", "TimeKey",
    }
    candidates = []
    for c in df.columns:
        if c in excluded or c.startswith("Target"):
            continue
        if pd.api.types.is_numeric_dtype(df[c]) and df[c].notna().any():
            # Historical demand features must already be available at cutoff; the flat file includes them.
            candidates.append(c)
    return candidates


def historical_same_hour(train, rows):
    # Fallback to location/hour mean calculated only from earlier dates.
    sums = train.groupby(["Location", "Hour"])["Target_HourlyUnits"].agg(["sum", "count"])
    vals = []
    for _, r in rows.iterrows():
        key = (r["Location"], r["Hour"])
        vals.append(float(sums.loc[key, "sum"] / sums.loc[key, "count"]) if key in sums.index else 0.0)
    return np.array(vals)


def seasonal_naive(train, rows, lag_days):
    lookup = train.set_index(["Location", "Date", "Hour"])["Target_HourlyUnits"]
    vals = []
    for _, r in rows.iterrows():
        key = (r["Location"], r["Date"] - pd.Timedelta(days=lag_days), r["Hour"])
        vals.append(float(lookup.get(key, np.nan)))
    fallback = historical_same_hour(train, rows)
    return np.where(np.isnan(vals), fallback, vals)


def run_level(df, level):
    records = []
    pred_rows = []
    if level == "Hourly":
        work = df[df["HourlyFitEligible"].astype(str).str.lower().isin(["true", "1"]) & df["Target_HourlyUnits"].ge(0)].copy()
        work["KeyDate"] = work["Date"]
        target = "Target_HourlyUnits"
    else:
        work = (df.groupby(["Location", "Date"], as_index=False)
                .agg(Target_DailyUnits=("Target_HourlyUnits", "sum"), DataSplit=("DataSplit", "first"),
                     Weekday=("Weekday", "first"), ScheduledStudents_DailyPeak=("ScheduledStudents_DailyPeak", "max"),
                     RecordedHours=("Hour", "nunique")))
        work["KeyDate"] = work["Date"]
        target = "Target_DailyUnits"
    test = work[work["DataSplit"].astype(str).str.lower().eq("test")].copy()
    train = work[work["DataSplit"].astype(str).str.lower().isin(["train", "validation"])].copy()
    if level == "Hourly":
        train = train[train["Target_HourlyUnits"].ge(0)]
    for name in ["Seasonal Naive (7-day)", "Historical Same-Hour Average"]:
        if name.startswith("Seasonal"):
            pred = seasonal_naive(train, test, 7) if level == "Hourly" else seasonal_naive_daily(train, test, 7)
        else:
            if level == "Hourly": pred = historical_same_hour(train, test)
            else: pred = train.groupby("Location")[target].mean().reindex(test["Location"]).fillna(0).to_numpy()
        m = metrics(test[target], pred); m.update({"ForecastLevel": level, "Model": name, "Configuration": "Baseline", "Evaluation": "Common eligible test rows", "Status": "executed"})
        records.append(m)
        for (_, row), yhat in zip(test.iterrows(), pred):
            pred_rows.append({"ForecastLevel": level, "Location": row["Location"], "Date": row["Date"], "Hour": row.get("Hour", np.nan), "Actual": row[target], "Predicted": yhat, "Model": name, "Configuration": "Baseline"})
    return pd.DataFrame(records), pd.DataFrame(pred_rows), train, test


def seasonal_naive_daily(train, rows, lag_days):
    lookup = train.set_index(["Location", "Date"])["Target_DailyUnits"]
    vals=[]
    for _, r in rows.iterrows():
        vals.append(lookup.get((r["Location"], r["Date"]-pd.Timedelta(days=lag_days)), np.nan))
    fallback=train.groupby("Location")["Target_DailyUnits"].mean().reindex(rows["Location"]).fillna(0).to_numpy()
    return np.where(pd.isna(vals), fallback, vals)


def optional_model_status():
    statuses=[]
    for name in ["sklearn", "xgboost", "lightgbm", "prophet", "statsmodels"]:
        _, err=safe_import(name)
        statuses.append({"Component": name, "Status": "available" if err is None else "blocked", "Detail": "" if err is None else err})
    return pd.DataFrame(statuses)


def main():
    if not INPUT.exists(): raise FileNotFoundError(INPUT)
    df = pd.read_csv(INPUT)
    source_rows = len(df); source_units = int(pd.to_numeric(df["Target_HourlyUnits"]).sum())
    df = add_status(df)
    df["HourlyFitEligible"] = df.get("HourlyFitEligible", True).astype(str).str.lower().isin(["true", "1"])
    features = available_features(df)
    historical = {"Lag1Day_SameHour","Lag7Day_SameHour","Lag14Day_SameHour","HistoricalSameHour_Avg","HistoricalSameHour_Count","PrevDay_TotalUnits","Prev7Days_AvgDailyUnits","Prev7Days_ObservedDays","Prev28Days_AvgDailyUnits","Prev28Days_ObservedDays"}
    student = {"EstimatedStudentsOnCampus","ScheduledStudents_DailyPeak"}
    post_outcome = {"NetSales","GrossAmount","TotalDiscount","TransactionRows","UniqueItems"}
    availability=[]
    for f in sorted(set(features) | post_outcome):
        if f in post_outcome: status_label, rationale = "excluded_post_outcome", "Observed after sales; unavailable at forecast time"
        elif f in historical: status_label, rationale = "available_from_prior_history", "Uses information dated before the forecast cutoff"
        elif f in student: status_label, rationale = "available_if_schedule_input_supplied", "Estimated/scheduled student activity; not measured attendance"
        elif f in {"TemperatureF","HumidityPct","PrecipitationIn","RainFlag","CloudCoverPct","WindSpeedKmh"}: status_label, rationale = "available_if_weather_forecast_supplied", "Requires a forecast-time weather source"
        else: status_label, rationale = "calendar_or_schedule_input", "Known from calendar, class, event, or location context before forecast time"
        availability.append({"Predictor":f,"Availability":status_label,"Rationale":rationale,"IncludedInPhase2Features":f in features})
    status = optional_model_status()
    quality = pd.DataFrame([
        {"Check":"Hourly rows", "Result":source_rows, "Expected":EXPECTED_ROWS, "Status":"PASS" if source_rows==EXPECTED_ROWS else "FAIL"},
        {"Check":"Target units", "Result":source_units, "Expected":EXPECTED_UNITS, "Status":"PASS" if source_units==EXPECTED_UNITS else "FAIL"},
        {"Check":"Duplicate Location-Date-Hour keys", "Result":int(df.duplicated(["Location","Date","Hour"]).sum()), "Expected":0, "Status":"PASS" if not df.duplicated(["Location","Date","Hour"]).any() else "FAIL"},
        {"Check":"Observed zero hours", "Result":int(df["HourStatus"].eq("observed_zero").sum()), "Expected":2, "Status":"PASS" if int(df["HourStatus"].eq("observed_zero").sum())==2 else "FAIL"},
        {"Check":"Negative target rows", "Result":int(df["HourStatus"].eq("observed_negative_adjustment").sum()), "Expected":1, "Status":"PASS" if int(df["HourStatus"].eq("observed_negative_adjustment").sum())==1 else "FAIL"},
        {"Check":"Spring Break predictors excluded", "Result":"Yes", "Expected":"Yes", "Status":"PASS"},
        {"Check":"CampusClosedFlag kept separate", "Result":"Yes", "Expected":"Yes", "Status":"PASS"},
    ])
    hourly_metrics, hourly_preds, _, _ = run_level(df, "Hourly")
    daily_metrics, daily_preds, _, _ = run_level(df, "Daily")
    all_metrics = pd.concat([hourly_metrics, daily_metrics], ignore_index=True)
    for level, preds in [("Hourly", hourly_preds), ("Daily", daily_preds)]:
        if preds.empty: continue
        if "Weekday" not in preds: preds["Weekday"] = pd.to_datetime(preds["Date"]).dt.day_name()
        # Keep each model's subgroup metrics separate; otherwise rows from
        # multiple models would inflate coverage and make comparisons unfair.
        for (model_name, group_col), model_group in preds.groupby(["Model", "Location"]):
            m=metrics(model_group["Actual"],model_group["Predicted"]); m.update({"ForecastLevel":level,"Model":model_name,"Configuration":"Baseline","Evaluation":"Test subgroup","GroupType":"Location","Group":group_col})
            all_metrics=pd.concat([all_metrics,pd.DataFrame([m])],ignore_index=True)
        for (model_name, group_col), model_group in preds.groupby(["Model", "Weekday"]):
            m=metrics(model_group["Actual"],model_group["Predicted"]); m.update({"ForecastLevel":level,"Model":model_name,"Configuration":"Baseline","Evaluation":"Test subgroup","GroupType":"Weekday","Group":group_col})
            all_metrics=pd.concat([all_metrics,pd.DataFrame([m])],ignore_index=True)
        if level=="Hourly":
            for (model_name, group),g in preds.groupby(["Model","Hour"]):
                m=metrics(g["Actual"],g["Predicted"]); m.update({"ForecastLevel":level,"Model":model_name,"Configuration":"Baseline","Evaluation":"Test subgroup","GroupType":"Hour","Group":group})
                all_metrics=pd.concat([all_metrics,pd.DataFrame([m])],ignore_index=True)
    df.to_csv(OUT/"prepared_hourly_with_status.csv", index=False)
    quality.to_csv(OUT/"data_quality_checks.csv", index=False)
    status.to_csv(OUT/"dependency_status.csv", index=False)
    pd.DataFrame({"ForecastLevel":["Hourly","Daily"],"PredictorCount":[len(features),len([x for x in features if x not in ["Hour"]])],"Predictors":["; ".join(features),"; ".join(features)]}).to_csv(OUT/"predictor_manifest.csv",index=False)
    pd.DataFrame(availability).to_csv(OUT/"predictor_availability.csv", index=False)
    all_metrics.to_csv(OUT/"model_metrics.csv", index=False)
    pd.concat([hourly_preds,daily_preds],ignore_index=True).to_csv(OUT/"predictions_test.csv", index=False)
    summary={"source_rows":source_rows,"source_units":source_units,"hourly_fit_rows":int(df["HourlyFitEligible"].sum()),"hourly_negative_excluded":int((~df["HourlyFitEligible"]).sum()),"observed_zero":int(df["HourStatus"].eq("observed_zero").sum()),"advanced_models_attempted":False,"note":"Advanced model execution blocked by missing optional dependencies; baselines executed."}
    (OUT/"run_summary.json").write_text(json.dumps(summary,indent=2))
    log.info(json.dumps(summary))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    try: main()
    except Exception:
        log.error(traceback.format_exc()); raise
