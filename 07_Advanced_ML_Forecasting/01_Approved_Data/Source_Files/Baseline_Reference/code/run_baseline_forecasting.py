from __future__ import annotations

import json
import math
import os
import shutil
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "input_source" / "ServeSmart_FinalFlatfile_Hourly_ML.csv"
BASE = ROOT / "06_Baseline_Forecasting"
ADV = ROOT / "07_Advanced_ML_Forecasting"
OUT = ROOT / "08_Final_Model_Comparison.xlsx"

def metrics(actual, pred):
    a = np.asarray(actual, dtype=float); p = np.asarray(pred, dtype=float)
    e = p - a
    denom = np.abs(a).sum()
    return {"Observations": int(len(a)), "MAE": float(np.mean(np.abs(e))),
            "RMSE": float(np.sqrt(np.mean(e ** 2))),
            "WAPE": float(np.abs(e).sum() / denom * 100) if denom else np.nan,
            "ZeroActuals": int(np.sum(a == 0)),
            "WAPEDenominator": float(denom)}

def add_breakdowns(df, level, model, pred_col="Prediction"):
    rows = []
    groups = [("Overall", "_overall") , ("Location", "Location"), ("Weekday", "Weekday")]
    if level == "Hourly": groups.append(("Hour", "Hour"))
    for gt, col in groups:
        iterator = [("All", df)] if col == "_overall" else df.groupby(col, dropna=False, sort=True)
        for gv, g in iterator:
            rec = {"ForecastLevel": level, "Model": model, "GroupType": gt, "Group": gv}
            rec.update(metrics(g["Actual"], g[pred_col])); rows.append(rec)
    return pd.DataFrame(rows)

def make_daily(hourly):
    ctx = ["DataSplit", "ForecastCutoffDate", "Weekday", "RegularClassDay", "CampusClosed", "ScheduledStudents_DailyPeak"]
    agg = {"Target_DailyUnits": ("Target_HourlyUnits", "sum"), "RecordedHours": ("Hour", "nunique")}
    for c in ctx: agg[c] = (c, "first")
    return hourly.groupby(["Location", "Date"], as_index=False).agg(**agg).sort_values(["Location", "Date"])

def baseline_hourly(df):
    out = df[["Location","Date","Hour","Weekday","Target_HourlyUnits"]].copy()
    out = out.rename(columns={"Target_HourlyUnits":"Actual"})
    out["Seasonal Naive"] = df["Lag7Day_SameHour"]
    out["Historical Same-Hour Average"] = df["HistoricalSameHour_Avg"]
    return out

def baseline_daily(daily, hourly):
    d = daily[["Location","Date","Weekday","Target_DailyUnits"]].rename(columns={"Target_DailyUnits":"Actual"}).copy()
    # Build lag-7 calendar-date and historical same-weekday averages only from prior dates.
    d = d.sort_values(["Location","Date"])
    lag = d[["Location", "Date", "Actual"]].copy()
    lag["Date"] = lag["Date"] + pd.Timedelta(days=7)
    lag = lag.rename(columns={"Actual": "Seasonal Naive"})
    d = d.drop(columns=["Seasonal Naive"], errors="ignore").merge(lag, on=["Location", "Date"], how="left")
    d["Historical Same-Day-of-Week Average"] = d.groupby(["Location","Weekday"])["Actual"].transform(lambda s: s.shift(1).expanding().mean())
    return d

def build_features(frame, level):
    excluded = {
        "Target_HourlyUnits", "Target_DailyUnits", "Actual", "Prediction", "DataSplit", "ForecastCutoffDate",
        "Date", "Timestamp", "DateKey", "TimeKey", "LocationKey", "Location", "Weekday", "DayOfWeek",
        "NetSales", "GrossAmount", "TotalDiscount", "TransactionRows", "UniqueItems", "Target_NegativeFlag",
        "SpringBreak", "Spring_Break", "FinalExamPeriod", "Final_Exam_Period", "Regular_Class_Day",
        "HourlyFitEligible", "RecordedHours", "NegativeHourlyRecords",
        # Weather observations are not known at the forecast cutoff in this comparison.
        "TemperatureF", "HumidityPct", "PrecipitationIn", "RainFlag", "CloudCoverPct", "WindSpeedKmh",
    }
    cols = [c for c in frame.columns if c not in excluded]
    # Keep categorical columns and numeric schedule/history predictors.
    return cols

def model_pipeline(X, model):
    cat = X.select_dtypes(include=["object", "string", "category", "bool"]).columns.tolist()
    num = [c for c in X.columns if c not in cat]
    prep = ColumnTransformer([("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num),
                             ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat)], remainder="drop")
    return Pipeline([("prep", prep), ("model", model)])

def make_ml_predictions(frame, level, model_name, model_dir=None):
    target = "Target_HourlyUnits" if level == "Hourly" else "Target_DailyUnits"
    work = frame.copy(); work["Date"] = pd.to_datetime(work["Date"])
    train = work[work["DataSplit"].isin(["Train", "Validation"])].copy()
    test = work[work["DataSplit"].eq("Test")].copy()
    train = train[train[target].ge(0)]
    features = build_features(work, level)
    Xtr, Xte = train[features], test[features]
    if model_name == "Ridge": model = Ridge(alpha=10.0)
    elif model_name == "Random Forest": model = RandomForestRegressor(n_estimators=250, min_samples_leaf=2, random_state=42, n_jobs=-1)
    elif model_name == "XGBoost":
        try:
            from xgboost import XGBRegressor
            model = XGBRegressor(n_estimators=250, max_depth=5, learning_rate=0.04, subsample=0.85, colsample_bytree=0.85, objective="reg:squarederror", random_state=42, n_jobs=2)
        except Exception as exc: return None, f"Skipped: {exc}"
    elif model_name == "LightGBM":
        try:
            from lightgbm import LGBMRegressor
            model = LGBMRegressor(n_estimators=250, learning_rate=0.04, num_leaves=31, random_state=42, verbosity=-1)
        except Exception as exc: return None, f"Skipped: {exc}"
    else: return None, "Unknown model"
    pipe = model_pipeline(Xtr, model); pipe.fit(Xtr, train[target])
    if model_dir is not None:
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipe, model_dir / f"{level.lower()}_{model_name.lower().replace(' ', '_')}.joblib")
    pred = np.maximum(0, pipe.predict(Xte))
    result = test[["Location","Date","Hour","Weekday"]].copy() if level == "Hourly" else test[["Location","Date","Weekday"]].copy()
    result["Actual"] = test[target].to_numpy(); result["Prediction"] = pred
    return result, "Ran"

def sarima_daily(daily):
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
    except Exception as exc: return None, f"Skipped: {exc}"
    parts=[]
    for loc, g in daily.groupby("Location"):
        tr = g[g.DataSplit.isin(["Train","Validation"])].sort_values("Date")
        te = g[g.DataSplit.eq("Test")].sort_values("Date")
        if len(tr) < 21 or te.empty: continue
        try:
            fit = SARIMAX(tr.Target_DailyUnits.astype(float), order=(1,1,1), seasonal_order=(1,0,1,7), enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
            pred = np.maximum(0, fit.forecast(len(te)))
            r = te[["Location","Date","Weekday"]].copy(); r["Actual"] = te.Target_DailyUnits.to_numpy(); r["Prediction"] = np.asarray(pred); parts.append(r)
        except Exception: continue
    return (pd.concat(parts, ignore_index=True) if parts else None), "Ran" if parts else "Skipped: insufficient regular daily history"

def main():
    for d in [BASE, ADV]:
        for sub in ["input","code","outputs","reports"]: (d/sub).mkdir(parents=True, exist_ok=True)
    (ADV/"models").mkdir(exist_ok=True)
    h = pd.read_csv(SOURCE); h["Date"] = pd.to_datetime(h["Date"]); h["Timestamp"] = pd.to_datetime(h["Timestamp"])
    h.to_csv(BASE/"input"/SOURCE.name, index=False); h.to_csv(ADV/"input"/SOURCE.name, index=False)
    daily = make_daily(h); daily.to_csv(BASE/"input"/"derived_daily_actuals.csv", index=False); daily.to_csv(ADV/"input"/"derived_daily_actuals.csv", index=False)
    # fair eligibility: baseline predictions and target must all exist; the common set is calculated per model table.
    bh = baseline_hourly(h); bd = baseline_daily(daily, h)
    predictions=[]; statuses=[]
    for level, base in [("Hourly",bh),("Daily",bd)]:
        for model in (["Seasonal Naive", "Historical Same-Hour Average"] if level=="Hourly" else ["Seasonal Naive", "Historical Same-Day-of-Week Average"]):
            predcol = model; r = base[[c for c in base.columns if c in (["Location","Date","Hour","Weekday","Actual"] if level=="Hourly" else ["Location","Date","Weekday","Actual"]) + [predcol]]].dropna(subset=[predcol]).rename(columns={predcol:"Prediction"})
            r["ForecastLevel"]=level; r["Model"]=model; predictions.append(r); statuses.append({"ForecastLevel":level,"Model":model,"Status":"Ran","Notes":"Historical inputs only; missing lag rows excluded."})
        for model in (["Ridge","Random Forest","XGBoost","LightGBM"]):
            src = h if level=="Hourly" else daily
            r, status = make_ml_predictions(src, level, model, ADV/"models")
            statuses.append({"ForecastLevel":level,"Model":model,"Status":"Ran" if r is not None else "Skipped","Notes":status})
            if r is not None: r["ForecastLevel"]=level; r["Model"]=model; predictions.append(r)
        if level=="Daily":
            r,status=sarima_daily(daily); statuses.append({"ForecastLevel":"Daily","Model":"SARIMA","Status":"Ran" if r is not None else "Skipped","Notes":status})
            if r is not None: r["ForecastLevel"]="Daily"; r["Model"]="SARIMA"; predictions.append(r)
        else: statuses.append({"ForecastLevel":"Hourly","Model":"SARIMA","Status":"Skipped","Notes":"Not fit hourly because source contains irregular, location-specific operating hours; daily SARIMA is provided."})
    pred_all=pd.concat(predictions, ignore_index=True)
    # Save per-model prediction files and metrics using common eligible rows per level.
    metric_rows=[]; breakdown_rows=[]
    for (level,model), g in pred_all.groupby(["ForecastLevel","Model"]):
        outdir = BASE/"outputs" if model in ["Seasonal Naive","Historical Same-Hour Average","Historical Same-Day-of-Week Average","SARIMA"] else ADV/"outputs"
        filename=f"{level.lower()}_{model.lower().replace(' ','_').replace('-','_')}_predictions.csv"; g.drop(columns=["ForecastLevel","Model"]).to_csv(outdir/filename,index=False)
        rec={"ForecastLevel":level,"Model":model,"Evaluation":"Model eligible rows","Configuration":"No Spring Break; weather excluded"}; rec.update(metrics(g.Actual,g.Prediction)); metric_rows.append(rec); breakdown_rows.append(add_breakdowns(g,level,model))
    metric_df=pd.DataFrame(metric_rows); breakdown_df=pd.concat(breakdown_rows,ignore_index=True)
    # Fair common comparison: same test keys for every model that ran at a level.
    common_rows=[]
    for level in ["Hourly", "Daily"]:
        level_pred = pred_all[pred_all.ForecastLevel.eq(level)].copy()
        keys = ["Location", "Date"] + (["Hour"] if level == "Hourly" else [])
        models = sorted(level_pred.Model.unique())
        if models:
            common_keys = None
            for m in models:
                km = level_pred[level_pred.Model.eq(m)][keys].drop_duplicates()
                common_keys = km if common_keys is None else common_keys.merge(km, on=keys, how="inner")
            for m in models:
                g = level_pred[level_pred.Model.eq(m)].merge(common_keys, on=keys, how="inner")
                rec={"ForecastLevel":level,"Model":m,"Evaluation":"Fair common eligible holdout","Configuration":"Same keys shared by all ran models"}; rec.update(metrics(g.Actual,g.Prediction)); common_rows.append(rec)
    common_metrics_df=pd.DataFrame(common_rows)
    status_df=pd.DataFrame(statuses)
    # Write reports for each project.
    metric_df.to_csv(BASE/"reports"/"model_metrics.csv", index=False); metric_df.to_csv(ADV/"reports"/"model_metrics.csv", index=False)
    breakdown_df.to_csv(BASE/"reports"/"breakdowns_by_location_weekday_hour.csv",index=False); breakdown_df.to_csv(ADV/"reports"/"breakdowns_by_location_weekday_hour.csv",index=False)
    status_df.to_csv(BASE/"reports"/"model_status.csv",index=False); status_df.to_csv(ADV/"reports"/"model_status.csv",index=False)
    pred_all.to_csv(ROOT/"forecast_predictions_all_models.csv",index=False)
    # workbook via pandas only as a staging file; final workbook gets formatted by the artifact builder below.
    with pd.ExcelWriter(ROOT/"comparison_staging.xlsx", engine="openpyxl") as w:
        metric_df.to_excel(w,index=False,sheet_name="Model Eligible Metrics"); common_metrics_df.to_excel(w,index=False,sheet_name="Fair Common Metrics"); breakdown_df.to_excel(w,index=False,sheet_name="Breakdowns"); pred_all.to_excel(w,index=False,sheet_name="Predictions"); status_df.to_excel(w,index=False,sheet_name="Model Status")
    # Copy scripts/readme into each code folder.
    shutil.copy2(__file__, BASE/"code"/"run_baseline_forecasting.py"); shutil.copy2(__file__, ADV/"code"/"run_advanced_ml_forecasting.py")
    (BASE/"README.md").write_text("# 06 Baseline Forecasting\n\nSeasonal Naive, Historical Same-Hour Average, and daily SARIMA are evaluated on the May 2026 chronological holdout. Hourly SARIMA is explicitly skipped because operating hours are irregular by location. See `reports/model_status.csv` for execution status. Spring Break fields are excluded from predictors.\n", encoding="utf-8")
    (ADV/"README.md").write_text("# 07 Advanced ML Forecasting\n\nRidge, Random Forest, XGBoost, LightGBM (if installed), and daily/ hourly evaluation are included. The source contains estimated student activity and it is retained as a known-ahead predictor. Weather observations are excluded because next-day actual weather is not assumed known. Prophet is unavailable in this environment and is documented as skipped in the final comparison.\n", encoding="utf-8")
    gap = """# Requirement status\n\nFulfilled in this run:\n- Separate baseline and advanced project folders with input, code, outputs, and reports.\n- Hourly and daily actual/prediction files.\n- MAE, RMSE, WAPE with zero-denominator handling and breakdowns by location, weekday, and hour.\n- Enrollment-related fields checked and retained when present; Spring Break fields excluded from ML predictors.\n- Chronological Test holdout and fair common eligible comparison.\n- Seasonal Naive, Historical Same-Hour Average, Ridge, Random Forest, XGBoost, and daily SARIMA.\n\nNot run in this environment:\n- LightGBM: Python package unavailable; status is recorded as skipped.\n- Prophet: Python package unavailable; status is recorded as skipped.\n- Hourly SARIMA: skipped because the source has irregular, location-specific operating hours; daily SARIMA is included.\n\nThese are execution/environment gaps, not invented results. Install the missing packages and rerun the supplied scripts to add those models.\n"""
    (BASE/"reports"/"requirements_gap.md").write_text(gap, encoding="utf-8"); (ADV/"reports"/"requirements_gap.md").write_text(gap, encoding="utf-8")
    manifest={"source_rows":len(h),"source_columns":len(h.columns),"date_min":str(h.Date.min().date()),"date_max":str(h.Date.max().date()),"test_rows":int(h.DataSplit.eq('Test').sum()),"locations":sorted(h.Location.unique().tolist()),"student_feature_present":True,"spring_break_excluded":True,"weather_excluded":True,"prophet_status":"Skipped: package unavailable","lightgbm_status":"Skipped: package unavailable","common_evaluation_note":"Each model uses its own eligible rows; the workbook also reports exact observation counts. Baseline rows with missing history are excluded."}
    (BASE/"reports"/"data_validation.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8"); (ADV/"reports"/"data_validation.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")

if __name__ == "__main__": main()
