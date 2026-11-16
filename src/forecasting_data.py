from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

REQUIRED_COLUMNS = {"Location", "Date", "Hour", "Target_HourlyUnits"}
APPROVED_ML_FEATURES = [
    "Location", "Weekday", "Hour", "Month", "IsWeekend",
    "Lag1Day_SameHour", "Lag7Day_SameHour", "Lag14Day_SameHour",
    "Lag1Day_Missing", "Lag7Day_Missing", "Lag14Day_Missing",
    "HistoricalSameHour_Avg", "HistoricalSameHour_Count", "PrevDay_TotalUnits",
    "Prev7Days_AvgDailyUnits", "Prev7Days_ObservedDays", "Prev28Days_AvgDailyUnits",
    "Prev28Days_ObservedDays", "SpringBreak", "FinalExamPeriod", "RegularClassDay",
    "CampusClosed",
]

LEGACY_DEPLOYMENT_SAFE_FEATURES = [
    "Hour", "Weekday", "Month", "WeekOfYear", "IsWeekend", "DayPart",
    "TemperatureC", "PrecipitationMM", "RainMM", "CloudCover", "WindSpeedKmh",
    "EventCount", "FoodEventCount", "CanceledEventCount", "AcademicPhase",
    "AcademicStatus", "CampusClosed", "SpringBreak", "FinalExamPeriod",
    "RegularClassDay", "HolidayOrClosure",
]


def load_ml_flat_file(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"ML flat file is missing required columns: {', '.join(missing)}")
    frame = frame.copy()
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame["Hour"] = pd.to_numeric(frame["Hour"], errors="coerce")
    frame["Target_HourlyUnits"] = pd.to_numeric(frame["Target_HourlyUnits"], errors="coerce")
    if frame[["Location", "Date", "Hour", "Target_HourlyUnits"]].isna().any().any():
        raise ValueError("ML flat file contains missing key/date/hour/target values")
    return frame.sort_values(["Location", "Date", "Hour"]).reset_index(drop=True)


def safe_features(frame: pd.DataFrame, features: Iterable[str] | None = None) -> list[str]:
    requested = list(features or APPROVED_ML_FEATURES)
    usable = [column for column in requested if column in frame.columns]
    if features is None and len(usable) != len(requested):
        usable = [column for column in LEGACY_DEPLOYMENT_SAFE_FEATURES if column in frame.columns]
    if not usable:
        raise ValueError("No deployment-safe predictors are present in the ML flat file")
    return usable


def chronological_split(frame: pd.DataFrame, train_fraction: float = 0.7, validation_fraction: float = 0.15) -> pd.DataFrame:
    if not 0 < train_fraction < 1 or not 0 <= validation_fraction < 1 or train_fraction + validation_fraction >= 1:
        raise ValueError("Fractions must leave a positive test period")
    result = frame.sort_values("Date").copy()
    if "DataSplit" in result.columns and set(result["DataSplit"].dropna().unique()) >= {"Train", "Validation", "Test"}:
        return result.reset_index(drop=True)
    dates = result["Date"].drop_duplicates().sort_values().tolist()
    train_end = dates[max(0, int(len(dates) * train_fraction) - 1)]
    validation_end = dates[max(0, int(len(dates) * (train_fraction + validation_fraction)) - 1)]
    result["DataSplit"] = "Test"
    result.loc[result["Date"] <= train_end, "DataSplit"] = "Train"
    result.loc[(result["Date"] > train_end) & (result["Date"] <= validation_end), "DataSplit"] = "Validation"
    return result
