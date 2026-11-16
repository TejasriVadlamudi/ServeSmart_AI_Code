from __future__ import annotations

import pandas as pd

DEFAULT_THRESHOLDS = {"high_demand": 0.75, "low_demand": 0.25, "rain_mm": 5.0}


def build_recommendations(frame: pd.DataFrame, thresholds: dict | None = None) -> pd.DataFrame:
    limits = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    data = frame.copy()
    if "ForecastUnits" not in data:
        data["ForecastUnits"] = data.get("Target_HourlyUnits", 0)
    q75 = data["ForecastUnits"].quantile(limits["high_demand"])
    q25 = data["ForecastUnits"].quantile(limits["low_demand"])
    rows = []
    for _, row in data.iterrows():
        forecast = float(row["ForecastUnits"])
        rain = float(row.get("RainMM", 0) or 0)
        if forecast >= q75:
            prep = "High-demand signal: review food-prep readiness and monitor sell-through."
            staff = "Review peak-period coverage and cross-train available staff."
        elif forecast <= q25:
            prep = "Lower-demand signal: consider smaller prep batches to reduce waste."
            staff = "Review whether normal coverage can be maintained without adding shifts."
        else:
            prep = "Moderate-demand signal: follow normal preparation cadence and monitor actuals."
            staff = "Use normal coverage and reassess near the service period."
        inventory = "Review perishables and reorder points against current stock and lead times."
        if rain >= limits["rain_mm"]:
            inventory += " Rain may affect demand; confirm outdoor/event assumptions."
        rows.append({"Location": row.get("Location", "Unknown"), "Date": row.get("Date"), "Hour": row.get("Hour"), "Food preparation": prep, "Inventory": inventory, "Staffing": staff, "Planning note": "Directional guidance only; not an exact quantity or staffing order."})
    return pd.DataFrame(rows)
