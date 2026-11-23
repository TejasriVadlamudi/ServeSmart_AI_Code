from pathlib import Path
import os

import pandas as pd


BASE = Path(__file__).resolve().parent
SOURCE = Path(os.environ.get("SERVESMART_SOURCE", BASE / "ServeSmart Final Dataset.xlsx"))
OUTPUT = Path(os.environ.get("SERVESMART_DIMENSIONAL_OUTPUT", BASE / "outputs_phase1_20261122"))
OUTPUT.mkdir(parents=True, exist_ok=True)


def item_numbers(series):
    return series.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)


def observed_date_hour():
    hourly = pd.read_excel(SOURCE, sheet_name="Hourly Forecast")
    hourly["Date"] = pd.to_datetime(hourly["Date"]).dt.normalize()
    hourly["Hour"] = pd.to_numeric(hourly["Hour"], errors="raise").astype(int)
    return hourly[["Date", "Hour"]].drop_duplicates()


def load_sales_lines():
    register_sheets = {
        "Residential Dining": "Residential Dining Register",
        "Einstein's": "Einstein's Register",
        "Panda Express": "Panda Express Register",
    }
    frames = []
    for location, sheet in register_sheets.items():
        frame = pd.read_excel(SOURCE, sheet_name=sheet, dtype={"ItemNumber": "string"})
        frame["Location"] = location
        frame["SourceSheet"] = sheet
        frame["SourceRowNumber"] = frame.index + 2
        frames.append(frame)
    sales = pd.concat(frames, ignore_index=True)
    sales["ItemNumber"] = item_numbers(sales["ItemNumber"])
    sales["Timestamp"] = pd.to_datetime(sales["Timestamp"], errors="raise")
    sales["Date"] = sales["Timestamp"].dt.normalize()
    sales["Hour"] = sales["Timestamp"].dt.hour
    sales["DateKey"] = sales["Date"].dt.strftime("%Y%m%d").astype(int)
    return sales


def build_core_dimensions(sales):
    hourly = pd.read_excel(SOURCE, sheet_name="Hourly Forecast")
    location_names = sorted(sales["Location"].dropna().unique())
    dim_location = pd.DataFrame({"Location": location_names})
    dim_location.insert(0, "LocationKey", range(1, len(dim_location) + 1))

    calendar = pd.read_excel(SOURCE, sheet_name="Academic Calendar")
    calendar["Date"] = pd.to_datetime(calendar["Date"]).dt.normalize()
    dim_date = calendar.copy()
    dim_date["DateKey"] = dim_date["Date"].dt.strftime("%Y%m%d").astype(int)
    dim_date["Year"] = dim_date["Date"].dt.year
    dim_date["Month"] = dim_date["Date"].dt.month
    dim_date["DayOfMonth"] = dim_date["Date"].dt.day
    dim_date["DayOfWeek"] = dim_date["Date"].dt.day_name()
    dim_date["WeekOfYear"] = dim_date["Date"].dt.isocalendar().week.astype(int)
    dim_date = dim_date.drop_duplicates("DateKey")
    date_columns = [
        "DateKey", "Date", "Year", "Month", "DayOfMonth", "DayOfWeek", "WeekOfYear",
        "Academic_Phase", "Academic_Status", "Spring_Break", "Final_Exam_Period",
        "Regular_Class_Day", "HolidayFlag", "CampusClosedFlag",
    ]
    dim_date = dim_date[date_columns]

    time_source = hourly[["Hour", "DayPart"]].copy()
    time_source["Hour"] = pd.to_numeric(time_source["Hour"], errors="raise").astype(int)
    dim_time = time_source.drop_duplicates().sort_values("Hour").reset_index(drop=True)
    if dim_time["Hour"].duplicated().any():
        raise ValueError("Dim_Time has multiple DayPart values for one hour.")
    dim_time.insert(0, "TimeKey", range(1, len(dim_time) + 1))

    for frame, key in [(dim_location, "LocationKey"), (dim_date, "DateKey"), (dim_time, "TimeKey")]:
        if frame[key].duplicated().any():
            raise ValueError(f"{key} is not unique.")
    return dim_location, dim_date, dim_time


def build_context_dimensions(dim_date, dim_time):
    grid = observed_date_hour()
    grid["DateKey"] = grid["Date"].dt.strftime("%Y%m%d").astype(int)

    grid = grid.merge(dim_time[["Hour", "TimeKey"]], on="Hour", how="left", validate="many_to_one")
    grid = grid.merge(dim_date[["DateKey"]], on="DateKey", how="left", validate="many_to_one")

    weather = pd.read_excel(SOURCE, sheet_name="Weather Hourly")
    weather["Date"] = pd.to_datetime(weather["Date"]).dt.normalize()
    weather["Hour"] = pd.to_numeric(weather["Hour"], errors="raise").astype(int)
    weather = weather[
        ["Date", "Hour", "TemperatureF", "HumidityPct", "PrecipitationIn",
         "RainFlag", "CloudCoverPct", "WindSpeedKmh"]
    ]
    weather = grid.merge(weather, on=["Date", "Hour"], how="left", validate="one_to_one")
    weather = weather.sort_values(["DateKey", "TimeKey"]).reset_index(drop=True)
    weather.insert(0, "WeatherKey", range(1, len(weather) + 1))
    weather["WeatherNaturalKey"] = weather["DateKey"].astype("string") + "|" + weather["TimeKey"].astype("string")
    weather["RainFlag"] = weather["RainFlag"].fillna(0).astype(int)
    weather_dim = weather[
        ["WeatherKey", "WeatherNaturalKey", "DateKey", "TimeKey", "TemperatureF",
         "HumidityPct", "PrecipitationIn", "RainFlag", "CloudCoverPct", "WindSpeedKmh"]
    ]

    events = pd.read_excel(SOURCE, sheet_name="Events Hourly")
    events["Date"] = pd.to_datetime(events["Date"]).dt.normalize()
    events["Hour"] = pd.to_numeric(events["Hour"], errors="raise").astype(int)
    event_columns = [
        "CampusEventCount", "EventOccurrenceCount", "EventFoodCount",
        "AllDayEventCount", "EventTypeCount",
    ]
    events = grid.merge(
        events[["Date", "Hour"] + event_columns],
        on=["Date", "Hour"],
        how="left",
        validate="one_to_one",
    )
    events[event_columns] = events[event_columns].fillna(0)
    events = events.sort_values(["DateKey", "TimeKey"]).reset_index(drop=True)
    events.insert(0, "EventKey", range(1, len(events) + 1))
    events["EventNaturalKey"] = events["DateKey"].astype("string") + "|" + events["TimeKey"].astype("string")
    events["HasEvent"] = events["CampusEventCount"].gt(0)
    event_dim = events[
        ["EventKey", "EventNaturalKey", "DateKey", "TimeKey", "HasEvent"] + event_columns
    ]

    classes = pd.read_excel(SOURCE, sheet_name="Class Weekday Hour")
    classes["Hour"] = pd.to_numeric(classes["Hour"], errors="raise").astype(int)
    calendar = pd.read_excel(SOURCE, sheet_name="Academic Calendar")
    calendar["Date"] = pd.to_datetime(calendar["Date"]).dt.normalize()
    calendar["DayOfWeek"] = calendar["Date"].dt.day_name()
    class_columns = ["ClassesActive", "EstimatedStudentsOnCampus", "ClassesStarting", "ClassesEnding"]
    class_grid = grid.merge(
        calendar[["Date", "DayOfWeek", "Regular_Class_Day"]],
        on="Date",
        how="left",
        validate="many_to_one",
    ).merge(
        classes[["DayOfWeek", "Hour"] + class_columns],
        on=["DayOfWeek", "Hour"],
        how="left",
        validate="many_to_one",
    )
    regular = class_grid["Regular_Class_Day"].astype("string").str.strip().str.lower().eq("yes")
    class_grid.loc[class_grid["Regular_Class_Day"].notna() & ~regular, class_columns] = 0
    class_grid[class_columns] = class_grid[class_columns].fillna(0)
    class_grid = class_grid.sort_values(["DateKey", "TimeKey"]).reset_index(drop=True)
    class_grid.insert(0, "ClassKey", range(1, len(class_grid) + 1))
    class_grid["ClassNaturalKey"] = class_grid["DateKey"].astype("string") + "|" + class_grid["TimeKey"].astype("string")
    class_grid["HasClass"] = class_grid["ClassesActive"].gt(0)
    class_dim = class_grid[
        ["ClassKey", "ClassNaturalKey", "DateKey", "TimeKey", "HasClass", "Regular_Class_Day"] + class_columns
    ]

    for frame, key in [(weather_dim, "WeatherNaturalKey"), (event_dim, "EventNaturalKey"), (class_dim, "ClassNaturalKey")]:
        if frame[key].duplicated().any():
            raise ValueError(f"{key} is not unique.")

    return class_dim, event_dim, weather_dim


def build_sales_line_fact(sales, class_dim, event_dim, weather_dim, dim_location, dim_date, dim_time):
    sales["TimeKey"] = sales["Hour"].map(dim_time.set_index("Hour")["TimeKey"])
    dim_item = sales[["Location", "ItemNumber", "Item"]].drop_duplicates().copy()
    dim_item["Item"] = dim_item["Item"].fillna("").astype("string").str.strip()
    dim_item = dim_item.sort_values(["Location", "ItemNumber"]).reset_index(drop=True)
    dim_item.insert(0, "ItemKey", range(1, len(dim_item) + 1))
    if dim_item.duplicated(["Location", "ItemNumber"]).any():
        raise ValueError("Item number maps to multiple names within a location.")
    dim_item.to_csv(OUTPUT / "Dim_Item.csv", index=False)

    fact = sales.merge(dim_location[["Location", "LocationKey"]], on="Location", how="left", validate="many_to_one")
    fact = fact.merge(dim_item[["Location", "ItemNumber", "ItemKey"]], on=["Location", "ItemNumber"], how="left", validate="many_to_one")
    fact = fact.merge(dim_date[["DateKey"]], on="DateKey", how="left", validate="many_to_one")
    fact = fact.merge(class_dim[["DateKey", "TimeKey", "ClassKey"]], on=["DateKey", "TimeKey"], how="left", validate="many_to_one")
    fact = fact.merge(event_dim[["DateKey", "TimeKey", "EventKey"]], on=["DateKey", "TimeKey"], how="left", validate="many_to_one")
    fact = fact.merge(weather_dim[["DateKey", "TimeKey", "WeatherKey"]], on=["DateKey", "TimeKey"], how="left", validate="many_to_one")

    for key in ["LocationKey", "ItemKey", "DateKey", "TimeKey", "ClassKey", "EventKey", "WeatherKey"]:
        if fact[key].isna().any():
            raise ValueError(f"Fact_Sales_Line has unmatched {key} values.")

    fact["SalesLineKey"] = fact["SourceSheet"].astype("string") + "|row=" + fact["SourceRowNumber"].astype("string")
    fact["NetAmount"] = fact["Amount"] - fact["Discount"]
    fact["IsAdjustment"] = fact["Quantity"].lt(0) | fact["Amount"].lt(0)
    columns = [
        "SalesLineKey", "LocationKey", "DateKey", "TimeKey", "ItemKey", "ClassKey", "EventKey", "WeatherKey",
        "Register", "Timestamp", "ItemNumber", "Item", "Quantity", "Discount", "Amount", "NetAmount",
        "IsAdjustment", "SourceSheet", "SourceRowNumber",
    ]
    fact = fact[columns].sort_values("SalesLineKey").reset_index(drop=True)
    if fact["SalesLineKey"].duplicated().any():
        raise ValueError("SalesLineKey is not unique.")
    return dim_item, fact


def build_aggregate_facts(fact):
    hourly = (
        fact.groupby(["LocationKey", "DateKey", "TimeKey"], as_index=False)
        .agg(
            TotalQuantity=("Quantity", "sum"),
            NetSales=("NetAmount", "sum"),
            GrossAmount=("Amount", "sum"),
            TotalDiscount=("Discount", "sum"),
            TransactionRows=("SalesLineKey", "count"),
            UniqueItems=("ItemKey", "nunique"),
        )
    )
    item_daily = (
        fact.groupby(["LocationKey", "DateKey", "ItemKey"], as_index=False)
        .agg(
            Quantity=("Quantity", "sum"),
            NetSales=("NetAmount", "sum"),
            Discount=("Discount", "sum"),
            TransactionRows=("SalesLineKey", "count"),
        )
    )
    return hourly, item_daily


def build_hour_status_audit(sales, fact, dim_location, dim_date, dim_time):
    """Classify only what the source proves; campus closure is context, not outlet status."""
    source_hours = pd.read_excel(SOURCE, sheet_name="Hourly Forecast")
    source_hours["Date"] = pd.to_datetime(source_hours["Date"]).dt.normalize()
    source_hours["Hour"] = pd.to_numeric(source_hours["Hour"], errors="raise").astype(int)
    source_hours["DateKey"] = source_hours["Date"].dt.strftime("%Y%m%d").astype(int)
    source_hours = source_hours[["Location", "Date", "DateKey", "Hour"]].drop_duplicates()
    source_hours = source_hours.merge(dim_location, on="Location", how="left", validate="many_to_one")
    source_hours = source_hours.merge(dim_time[["Hour", "TimeKey"]], on="Hour", how="left", validate="many_to_one")
    calendar = pd.read_excel(SOURCE, sheet_name="Academic Calendar")
    calendar["Date"] = pd.to_datetime(calendar["Date"]).dt.normalize()
    calendar = calendar[["Date", "CampusClosedFlag", "HolidayFlag", "Regular_Class_Day"]].drop_duplicates()
    source_hours = source_hours.merge(calendar, on="Date", how="left", validate="many_to_one")
    observed = (
        fact.groupby(["LocationKey", "DateKey", "TimeKey"], as_index=False)
        .agg(ItemUnits=("Quantity", "sum"), NetSales=("NetAmount", "sum"), TransactionRows=("SalesLineKey", "count"))
    )
    audit = source_hours.merge(observed, on=["LocationKey", "DateKey", "TimeKey"], how="left")
    audit["TransactionRows"] = audit["TransactionRows"].fillna(0).astype(int)
    audit["ItemUnits"] = audit["ItemUnits"].fillna(0)
    audit["NetSales"] = audit["NetSales"].fillna(0.0)
    audit["CampusClosedFlag"] = audit["CampusClosedFlag"].fillna("").astype("string")
    audit["OutletScheduleEvidence"] = "none_available"
    audit["HourStatus"] = "missing_or_unconfirmed"
    audit.loc[audit["TransactionRows"] > 0, "HourStatus"] = "observed_sales"
    audit.loc[(audit["TransactionRows"] > 0) & audit["ItemUnits"].eq(0) & audit["NetSales"].eq(0), "HourStatus"] = "observed_zero"
    columns = ["Location", "LocationKey", "Date", "DateKey", "Hour", "TimeKey", "HourStatus",
               "ItemUnits", "NetSales", "TransactionRows", "CampusClosedFlag", "HolidayFlag",
               "Regular_Class_Day", "OutletScheduleEvidence"]
    audit = audit[columns].sort_values(["Date", "Location", "Hour"]).reset_index(drop=True)
    audit.to_csv(OUTPUT / "Hour_Status_Audit.csv", index=False)
    return audit


def build_reconciliation_reports(sales, fact, hourly):
    source_total = {"POSLines": len(sales), "ItemUnits": sales["Quantity"].sum(),
                    "GrossSales": sales["Amount"].sum(), "Discount": sales["Discount"].sum(),
                    "NetSales": (sales["Amount"] - sales["Discount"]).sum()}
    fact_total = {"POSLines": len(fact), "ItemUnits": fact["Quantity"].sum(),
                  "GrossSales": fact["Amount"].sum(), "Discount": fact["Discount"].sum(),
                  "NetSales": fact["NetAmount"].sum()}
    rows = []
    for metric in source_total:
        source_value, fact_value = source_total[metric], fact_total[metric]
        rows.append({"Measure": metric, "SourceValue": source_value, "FactValue": fact_value,
                     "Difference": fact_value - source_value,
                     "Status": "PASS" if abs(fact_value - source_value) < 0.005 else "FAIL"})
    pd.DataFrame(rows).to_csv(OUTPUT / "Source_to_Fact_Reconciliation.csv", index=False)
    hourly_total = {"ItemUnits": hourly["TotalQuantity"].sum(), "NetSales": hourly["NetSales"].sum(),
                    "GrossSales": hourly["GrossAmount"].sum(), "TransactionRows": hourly["TransactionRows"].sum()}
    rows = []
    for metric, value in hourly_total.items():
        fact_metric = {"ItemUnits": fact["Quantity"].sum(), "NetSales": fact["NetAmount"].sum(),
                       "GrossSales": fact["Amount"].sum(), "TransactionRows": len(fact)}[metric]
        rows.append({"Measure": metric, "HourlyValue": value, "FactValue": fact_metric,
                     "Difference": value - fact_metric,
                     "Status": "PASS" if abs(value - fact_metric) < 0.005 else "FAIL"})
    pd.DataFrame(rows).to_csv(OUTPUT / "Hourly_Source_to_Fact_Reconciliation.csv", index=False)


def write_scd4_limitation():
    (OUTPUT / "Campus_Context_SCD4_DESIGN_GAP.md").write_text(
        "# Campus Context SCD Type 4 limitation\n\n"
        "The workbook provides a current academic-calendar snapshot but no dated historical versions or effective timestamps. "
        "This revision preserves the current context and does not fabricate prior SCD4 rows. Future loads should capture "
        "EffectiveStartDate, EffectiveEndDate, IsCurrent, load timestamp, and change reason from dated official extracts.\n",
        encoding="utf-8",
    )


def validate_outputs(sales, dimensions, facts):
    checks = []

    def record(name, passed, detail):
        checks.append({"Check": name, "Status": "PASS" if passed else "FAIL", "Details": detail})
        if not passed:
            raise ValueError(f"Validation failed: {name} - {detail}")

    fact = facts["Fact_Sales_Line"]
    record("Sales-line row count", len(fact) == len(sales), f"{len(fact):,} fact rows vs {len(sales):,} source rows")
    record("SalesLineKey unique", fact["SalesLineKey"].is_unique, "No duplicate source-line keys")

    for key, dimension_name in [
        ("LocationKey", "Dim_Location"), ("DateKey", "Dim_Date"), ("TimeKey", "Dim_Time"),
        ("ItemKey", "Dim_Item"), ("ClassKey", "Dim_Class_Context"),
        ("EventKey", "Dim_Event"), ("WeatherKey", "Dim_Weather"),
    ]:
        dimension = dimensions[dimension_name]
        record(f"{dimension_name} key unique", dimension[key].is_unique, f"{len(dimension):,} rows")
        record(f"Fact foreign key {key}", fact[key].notna().all(), "No missing foreign keys")

    hourly = facts["Fact_Hourly_Demand"]
    item_daily = facts["Fact_Item_Daily"]
    record("Hourly grain unique", not hourly.duplicated(["LocationKey", "DateKey", "TimeKey"]).any(), "One row per location/date/hour")
    record("Item-daily grain unique", not item_daily.duplicated(["LocationKey", "DateKey", "ItemKey"]).any(), "One row per location/date/item")
    record("Authoritative POS units", fact["Quantity"].sum() == 63891, f"{fact['Quantity'].sum():,} units")
    record("Authoritative net sales", round(fact["NetAmount"].sum(), 2) == 369586.75, f"${fact['NetAmount'].sum():,.2f}")

    report = pd.DataFrame(checks)
    report.to_csv(OUTPUT / "ETL_Validation_Report.csv", index=False)
    print("ETL validation report:", OUTPUT / "ETL_Validation_Report.csv")


def main():
    print("SERVESMART AI - BUILDING GRANULAR SALES MODEL")
    sales = load_sales_lines()
    dim_location, dim_date, dim_time = build_core_dimensions(sales)
    dim_location.to_csv(OUTPUT / "Dim_Location.csv", index=False)
    dim_date.to_csv(OUTPUT / "Dim_Date.csv", index=False)
    dim_time.to_csv(OUTPUT / "Dim_Time.csv", index=False)
    class_dim, event_dim, weather_dim = build_context_dimensions(dim_date, dim_time)
    dim_item, fact = build_sales_line_fact(sales, class_dim, event_dim, weather_dim, dim_location, dim_date, dim_time)
    fact_hourly, fact_item_daily = build_aggregate_facts(fact)
    class_dim.to_csv(OUTPUT / "Dim_Class_Context.csv", index=False)
    event_dim.to_csv(OUTPUT / "Dim_Event.csv", index=False)
    weather_dim.to_csv(OUTPUT / "Dim_Weather.csv", index=False)
    fact.to_csv(OUTPUT / "Fact_Sales_Line.csv", index=False)
    fact_hourly.to_csv(OUTPUT / "Fact_Hourly_Demand.csv", index=False)
    fact_item_daily.to_csv(OUTPUT / "Fact_Item_Daily.csv", index=False)
    build_reconciliation_reports(sales, fact, fact_hourly)
    audit = build_hour_status_audit(sales, fact, dim_location, dim_date, dim_time)
    jan19 = audit[audit["Date"].eq(pd.Timestamp("2026-01-19")) & audit["TransactionRows"].gt(0)]
    print("Jan 19 observed location-hours:", len(jan19), "units:", jan19["ItemUnits"].sum(), "net sales:", round(jan19["NetSales"].sum(), 2))
    if len(jan19) != 4 or jan19["ItemUnits"].sum() != 8 or round(jan19["NetSales"].sum(), 2) != 95.06:
        raise ValueError("Jan 19 audit did not reconcile to 4 location-hours, 8 units, $95.06.")
    if audit.loc[audit["TransactionRows"].gt(0), "HourStatus"].eq("closed").any():
        raise ValueError("Recorded POS activity was incorrectly classified as closed.")
    write_scd4_limitation()
    dimensions = {
        "Dim_Location": dim_location,
        "Dim_Date": dim_date,
        "Dim_Time": dim_time,
        "Dim_Item": dim_item,
        "Dim_Class_Context": class_dim,
        "Dim_Event": event_dim,
        "Dim_Weather": weather_dim,
    }
    facts = {
        "Fact_Sales_Line": fact,
        "Fact_Hourly_Demand": fact_hourly,
        "Fact_Item_Daily": fact_item_daily,
    }
    validate_outputs(sales, dimensions, facts)
    print("Fact_Sales_Line rows:", len(fact))
    print("Dim_Item rows:", len(dim_item))
    print("Dim_Date rows:", len(dim_date))
    print("Dim_Time rows:", len(dim_time))
    print("Dim_Location rows:", len(dim_location))
    print("Dim_Class_Context rows:", len(class_dim))
    print("Dim_Event rows:", len(event_dim))
    print("Dim_Weather rows:", len(weather_dim))
    print("Fact_Hourly_Demand rows:", len(fact_hourly))
    print("Fact_Item_Daily rows:", len(fact_item_daily))
    print("Granular sales model created successfully.")


if __name__ == "__main__":
    main()
