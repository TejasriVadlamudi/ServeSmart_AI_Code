from __future__ import annotations

import base64
import re
from datetime import datetime, time
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))
from decision_support import build_recommendations  # noqa: E402
from forecasting_data import load_ml_flat_file  # noqa: E402

APPROVED_ML_FILE = ROOT / "data" / "ServeSmart_ML_Flat_File_Updated.csv"
BASELINE_FORECAST_FILE = ROOT / "outputs" / "baselines" / "approved" / "all_hour_forecasts.csv"
BASELINE_COMPARISON_FILE = ROOT / "outputs" / "models" / "model_comparison_common_test.csv"
QUEUE_FILE = ROOT / "data" / "queue_service_times.csv"
CLOSURE_FILE = ROOT / "data" / "campus_closures.csv"
OPERATING_HOURS_FILE = ROOT / "data" / "operating_hours.csv"
LEGACY_DEMO_FILE = ROOT.parent / "05_Baseline_Forecasting" / "outputs" / "ML_Flat_File.csv"
SALES_FILE = ROOT.parent / "04_ETL_Pipeline_Final" / "outputs" / "Fact_Sales_Line.csv"
LOCATION_FILE = ROOT.parent / "04_ETL_Pipeline_Final" / "outputs" / "Dim_Location.csv"
HERO_IMAGE = ROOT / "assets" / "servesmart-hero.png"
HOME_IMAGE = ROOT / "assets" / "servesmart-home.png"
DINING_COMMONS_IMAGE = ROOT / "assets" / "dining-commons.png"
EINSTEINS_IMAGE = ROOT / "assets" / "einsteins-bagels.png"
PANDAS_IMAGE = ROOT / "assets" / "pandas-food.png"
LOCAL_TZ = ZoneInfo("America/Los_Angeles")
STATION_NAMES = {"Einstein's": "Einstein's", "Panda Express": "Panda's", "Residential Dining": "Dining Commons"}
SUGGESTED_MANAGER_QUESTIONS = [
    "What are the predicted sales at 10 a.m. at Panda's?",
    "What are the predicted sales at 2 p.m. at Einstein's?",
    "Which items sell most on Mondays?",
]


def load_demo_data() -> pd.DataFrame:
    source_file = APPROVED_ML_FILE if APPROVED_ML_FILE.exists() else LEGACY_DEMO_FILE
    if source_file.exists():
        frame = load_ml_flat_file(source_file)
        frame["ForecastUnits"] = frame["Target_HourlyUnits"]
        frame["Date"] = pd.to_datetime(frame["Date"])
        frame["Station"] = frame["Location"].map(STATION_NAMES).fillna(frame["Location"])
        if BASELINE_FORECAST_FILE.exists():
            baseline = pd.read_csv(BASELINE_FORECAST_FILE, low_memory=False)
            baseline = baseline.rename(columns={"location": "Location", "date": "Date", "hour": "Hour"})
            baseline["Date"] = pd.to_datetime(baseline["Date"])
            baseline = baseline[["Location", "Date", "Hour", "seasonal_naive", "historical_hour_mean"]]
            frame = frame.merge(baseline, on=["Location", "Date", "Hour"], how="left")
            frame["ForecastUnits"] = frame["seasonal_naive"].fillna(frame["historical_hour_mean"]).fillna(frame["ForecastUnits"])
        return frame
    dates = pd.date_range("2026-01-01", periods=14, freq="D")
    rows = []
    station_bases = {"Einstein's": 24, "Panda Express": 30, "Residential Dining": 38}
    for date in dates:
        for location, base in station_bases.items():
            for hour in range(7, 22):
                demand = max(4, base + ((hour - 12) ** 2 // 4) * -1 + (date.dayofweek % 3) * 2)
                rows.append({"Location": location, "Station": STATION_NAMES[location], "Date": date, "Hour": hour, "Target_HourlyUnits": demand, "ForecastUnits": demand})
    return pd.DataFrame(rows)


def load_item_sales() -> pd.DataFrame:
    if not SALES_FILE.exists():
        rows = []
        demo_items = {"Einstein's": ["Bagel", "Coffee", "Muffin"], "Panda Express": ["Orange Chicken", "Fried Rice", "Spring Roll"], "Residential Dining": ["Pizza", "Salad", "Pasta"]}
        for day in pd.date_range("2026-01-01", periods=14, freq="D"):
            for location, items in demo_items.items():
                for hour in [8, 10, 12, 17]:
                    for rank, item in enumerate(items):
                        rows.append({"LocationKey": 1, "Location": location, "Timestamp": day + pd.Timedelta(hours=hour), "Date": day.date(), "Hour": hour, "Weekday": day.day_name(), "Item": item, "Quantity": max(1, 18 - rank * 4 + (day.dayofweek % 3)), "NetAmount": float(4 + rank)})
        return pd.DataFrame(rows)
    sales = pd.read_csv(SALES_FILE, low_memory=False)
    sales["Timestamp"] = pd.to_datetime(sales["Timestamp"], errors="coerce")
    sales["Date"] = sales["Timestamp"].dt.date
    sales["Hour"] = sales["Timestamp"].dt.hour
    sales["Weekday"] = sales["Timestamp"].dt.day_name()
    sales["Quantity"] = pd.to_numeric(sales["Quantity"], errors="coerce").fillna(0)
    sales["NetAmount"] = pd.to_numeric(sales["NetAmount"], errors="coerce").fillna(0)
    if LOCATION_FILE.exists():
        locations = pd.read_csv(LOCATION_FILE)
        sales = sales.merge(locations[["LocationKey", "Location"]], on="LocationKey", how="left")
    return sales.dropna(subset=["Timestamp", "Item"])


def load_queue_service_times() -> pd.DataFrame:
    """Load optional operational wait data; empty means no real queue feed is connected."""
    if not QUEUE_FILE.exists():
        return pd.DataFrame(columns=["Date", "Hour", "Station", "WaitMinutes"])
    queue = pd.read_csv(QUEUE_FILE)
    queue["Date"] = pd.to_datetime(queue["Date"], errors="coerce")
    queue["Hour"] = pd.to_numeric(queue["Hour"], errors="coerce")
    queue["WaitMinutes"] = pd.to_numeric(queue["WaitMinutes"], errors="coerce")
    return queue.dropna(subset=["Date", "Hour", "Station", "WaitMinutes"])


def load_closure_calendar() -> pd.DataFrame:
    """Load optional future campus closure dates."""
    if not CLOSURE_FILE.exists():
        return pd.DataFrame(columns=["Date", "Reason"])
    closures = pd.read_csv(CLOSURE_FILE)
    closures["Date"] = pd.to_datetime(closures["Date"], errors="coerce")
    return closures.dropna(subset=["Date"])


def load_operating_hours() -> pd.DataFrame:
    if not OPERATING_HOURS_FILE.exists():
        return pd.DataFrame(
            [
                {"Station": "Dining Commons", "OpenHour": 7, "CloseHour": 20},
                {"Station": "Einstein's", "OpenHour": 7, "CloseHour": 17},
                {"Station": "Panda's", "OpenHour": 10, "CloseHour": 16},
            ]
        )
    hours = pd.read_csv(OPERATING_HOURS_FILE)
    hours["OpenHour"] = pd.to_numeric(hours["OpenHour"], errors="coerce")
    hours["CloseHour"] = pd.to_numeric(hours["CloseHour"], errors="coerce")
    return hours.dropna(subset=["Station", "OpenHour", "CloseHour"])


def estimated_wait_minutes(forecast: float, reference: pd.Series) -> int:
    """Calibrated directional rule; real queue/service-time data is still required for production accuracy."""
    clean_reference = pd.to_numeric(reference, errors="coerce").dropna()
    if clean_reference.empty:
        return 15
    percentile = float((clean_reference.le(forecast)).mean())
    wait = 10 + 35 * percentile
    return int(max(5, min(45, round(wait / 5) * 5)))


def station_operating_status(data: pd.DataFrame, station: str, forecast_date: pd.Timestamp, service_hour: int, closures: pd.DataFrame, operating_hours: pd.DataFrame) -> str:
    """Return Open, Closed, or No forecast using campus flags and historical operating coverage."""
    if forecast_date.dayofweek == 6:
        return "Closed"
    if not closures.empty and closures["Date"].dt.normalize().eq(forecast_date.normalize()).any():
        return "Closed"
    schedule = operating_hours[operating_hours["Station"].eq(station)]
    if not schedule.empty:
        opening = int(schedule["OpenHour"].iloc[0])
        closing = int(schedule["CloseHour"].iloc[0])
        if service_hour < opening or service_hour >= closing:
            return "Closed"
    station_data = data[data["Station"].eq(station)]
    selected = station_data[station_data["Date"].dt.normalize().eq(forecast_date.normalize())]
    if "CampusClosed" in selected and not selected.empty and selected["CampusClosed"].fillna(0).astype(float).max() == 1:
        return "Closed"
    if "HolidayFlag" in selected and not selected.empty and selected["HolidayFlag"].fillna(0).astype(float).max() == 1:
        return "Closed"
    same_weekday = station_data[station_data["Date"].dt.dayofweek.eq(forecast_date.dayofweek)]
    if same_weekday.empty:
        return "Closed"
    if service_hour not in set(same_weekday["Hour"].astype(int)):
        return "Closed"
    if selected.empty:
        if forecast_date.normalize() > data["Date"].dt.normalize().max():
            return "Open"
        return "No forecast"
    if selected[selected["Hour"].eq(service_hour)].empty:
        return "No forecast"
    return "Open"


def forecast_for_date(data: pd.DataFrame, forecast_date: pd.Timestamp) -> tuple[pd.DataFrame, str]:
    """Use recorded baseline values when available, otherwise forecast by recent weekday/hour history."""
    exact = data[data["Date"].dt.normalize().eq(forecast_date.normalize())]
    if not exact.empty:
        return exact[["Station", "Hour", "ForecastUnits"]].copy(), "approved baseline output"

    rows = []
    weekday = forecast_date.dayofweek
    for station, station_data in data.groupby("Station"):
        prior = station_data[station_data["Date"].lt(forecast_date)]
        for hour in range(7, 22):
            same_pattern = prior[prior["Date"].dt.dayofweek.eq(weekday) & prior["Hour"].eq(hour)]
            if not same_pattern.empty:
                history = same_pattern.sort_values("Date").tail(4)["Target_HourlyUnits"]
            else:
                history = prior[prior["Hour"].eq(hour)].sort_values("Date").tail(8)["Target_HourlyUnits"]
            if history.empty:
                continue
            rows.append({"Station": station, "Hour": hour, "ForecastUnits": float(history.mean())})
    return pd.DataFrame(rows), "recent same-weekday/hour history"


def queue_wait_minutes(queue: pd.DataFrame, station: str, forecast_date: pd.Timestamp, service_hour: int) -> int | None:
    if queue.empty:
        return None
    match = queue[
        queue["Station"].eq(station)
        & queue["Date"].dt.normalize().eq(forecast_date.normalize())
        & queue["Hour"].eq(service_hour)
    ]
    if match.empty:
        return None
    return int(round(float(match["WaitMinutes"].iloc[-1])))


def forecast_value_at_time(forecast_data: pd.DataFrame, station: str, selected_time: time) -> float | None:
    """Interpolate between hourly forecasts so the customer control responds to minutes too."""
    decimal_hour = selected_time.hour + selected_time.minute / 60
    station_forecast = forecast_data[forecast_data["Station"].eq(station)].copy()
    if station_forecast.empty:
        return None
    station_forecast = station_forecast.sort_values("Hour")
    lower = station_forecast[station_forecast["Hour"].le(decimal_hour)].tail(1)
    upper = station_forecast[station_forecast["Hour"].ge(decimal_hour)].head(1)
    if lower.empty and upper.empty:
        return None
    if lower.empty:
        return float(upper["ForecastUnits"].iloc[0])
    if upper.empty:
        return float(lower["ForecastUnits"].iloc[0])
    low_hour = float(lower["Hour"].iloc[0])
    high_hour = float(upper["Hour"].iloc[0])
    low_value = float(lower["ForecastUnits"].iloc[0])
    high_value = float(upper["ForecastUnits"].iloc[0])
    if low_hour == high_hour:
        return low_value
    fraction = (decimal_hour - low_hour) / (high_hour - low_hour)
    return low_value + fraction * (high_value - low_value)


def rush_label(forecast: float, reference: pd.Series) -> str:
    high = float(reference.quantile(0.75))
    low = float(reference.quantile(0.25))
    return "High" if forecast >= high else "Low" if forecast <= low else "Moderate"


def answer_manager_query(query: str, sales: pd.DataFrame) -> pd.DataFrame:
    if sales.empty:
        return pd.DataFrame()
    result = sales.copy()
    text = query.lower().strip()
    if "peak" in text and "hour" in text:
        if "panda" in text:
            result = result[result["Location"].astype(str).str.lower().str.contains("panda")]
        peak = result.groupby("Hour", as_index=False)["Quantity"].sum().sort_values("Quantity", ascending=False).head(5)
        return peak.rename(columns={"Hour": "Hour of day", "Quantity": "Units sold"})
    hours = re.findall(r"\b([0-2]?\d)\s*(?:a\.m\.?|am|p\.m\.?|pm)\b", text)
    if hours:
        hour = int(hours[0])
        if "pm" in text or "p.m" in text:
            hour = 12 if hour == 12 else hour + 12
        else:
            hour = 0 if hour == 12 else hour
        result = result[result["Hour"].eq(hour)]
    weekdays = [day for day in ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"] if day in text]
    if weekdays:
        result = result[result["Weekday"].str.lower().eq(weekdays[0])]
    for location in sorted(result["Location"].dropna().unique(), key=len, reverse=True):
        if str(location).lower() in text:
            result = result[result["Location"].eq(location)]
            break
    item_matches = [item for item in sales["Item"].dropna().astype(str).unique() if item.lower() in text]
    if item_matches:
        result = result[result["Item"].isin(item_matches)]
    return result


def answer_predicted_sales_query(
    query: str,
    data: pd.DataFrame,
    forecast_date: pd.Timestamp,
    closures: pd.DataFrame,
    operating_hours: pd.DataFrame,
) -> pd.DataFrame:
    """Answer a predicted-sales question for the explicitly selected date."""
    text = query.lower().strip()
    real_today = pd.Timestamp(datetime.now(LOCAL_TZ).date())
    if "today" in text:
        forecast_date = real_today
    elif "tomorrow" in text:
        forecast_date = real_today + pd.Timedelta(days=1)
    elif "yesterday" in text:
        forecast_date = real_today - pd.Timedelta(days=1)
    forecast_data, _ = forecast_for_date(data, forecast_date)
    hours = re.findall(r"\b([0-2]?\d)\s*(?:a\.m\.?|am|p\.m\.?|pm)\b", text)
    hour = int(hours[0]) if hours else 11
    if "pm" in text or "p.m" in text:
        hour = 12 if hour == 12 else hour + 12
    else:
        hour = 0 if hour == 12 else hour
    stations = data["Station"].dropna().unique()
    station_filter = None
    for station in sorted(stations, key=len, reverse=True):
        if str(station).lower() in text:
            station_filter = station
            break
    if station_filter is not None:
        stations = [station_filter]
    rows = []
    for station in stations:
        status = station_operating_status(data, station, forecast_date, hour, closures, operating_hours)
        if status == "Closed":
            rows.append({
                "Date": forecast_date.strftime("%b %-d, %Y"),
                "Station": station,
                "Time": f"{hour}:00",
                "Predicted sales": "—",
                "Status": "Closed",
            })
            continue
        result = forecast_data[forecast_data["Station"].eq(station) & forecast_data["Hour"].eq(hour)]
        if result.empty or status == "No forecast":
            rows.append({
                "Date": forecast_date.strftime("%b %-d, %Y"),
                "Station": station,
                "Time": f"{hour}:00",
                "Predicted sales": "—",
                "Status": "Forecast unavailable",
            })
            continue
        forecast = float(result["ForecastUnits"].mean())
        rows.append({
            "Date": forecast_date.strftime("%b %-d, %Y"),
            "Station": station,
            "Time": f"{hour}:00",
            "Predicted sales": round(forecast),
            "Status": "Open",
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def customer_portal(st, data: pd.DataFrame, queue: pd.DataFrame, closures: pd.DataFrame, operating_hours: pd.DataFrame) -> None:
    st.markdown(
        """
        <div class="customer-hero">
          <div class="customer-brand">🍴 <span>Serve</span>Smart</div>
          <div class="customer-kicker">Customer-first dining guidance for your next visit.</div>
          <div class="customer-title">Customer <span>Portal</span></div>
          <div class="customer-benefits">◷ Check Wait Time &nbsp;&nbsp;|&nbsp;&nbsp; ◫ Plan Your Visit</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    plan_col, wait_col = st.columns([0.8, 2.2], gap="large")
    with plan_col:
        st.markdown("### Plan Your Visit")
        today = pd.Timestamp(datetime.now(LOCAL_TZ).date())
        forecast_date = pd.Timestamp(st.date_input("Forecast date", value=today.date(), key="customer_date"))
        # Offer every minute so Use Current Time can reflect the exact local minute.
        time_options = [time(hour, minute) for hour in range(24) for minute in range(60)]
        current_time = datetime.now(LOCAL_TZ).replace(second=0, microsecond=0).time()
        if "customer_time_value" not in st.session_state:
            nearest_index = min(
                range(len(time_options)),
                key=lambda index: abs(
                    (time_options[index].hour * 60 + time_options[index].minute)
                    - (current_time.hour * 60 + current_time.minute)
                ),
            )
            st.session_state["customer_time_value"] = time_options[nearest_index]
        if "customer_time_reset" not in st.session_state:
            st.session_state["customer_time_reset"] = 0
        if st.button("◉  Use Current Time", use_container_width=True):
            nearest_index = min(
                range(len(time_options)),
                key=lambda index: abs(
                    (time_options[index].hour * 60 + time_options[index].minute)
                    - (current_time.hour * 60 + current_time.minute)
                ),
            )
            # This runs before the dropdown is created, so Streamlit can safely
            # update the widget selection before rerunning the page.
            st.session_state["customer_time_value"] = time_options[nearest_index]
            st.session_state["customer_time_reset"] += 1
            st.rerun()
        time_widget_key = f"customer_time_select_{st.session_state['customer_time_reset']}"
        time_input = st.selectbox(
            "View time",
            options=time_options,
            index=time_options.index(st.session_state["customer_time_value"]),
            format_func=lambda value: value.strftime("%I:%M %p").lstrip("0"),
            key=time_widget_key,
        )
        st.session_state["customer_time_value"] = time_input
        auto_refresh = st.checkbox("Refresh automatically every minute", value=True)
        if auto_refresh:
            try:
                from streamlit_autorefresh import st_autorefresh
                st_autorefresh(interval=60_000, key="customer_minute_refresh")
            except ImportError:
                st.caption("Automatic refresh will be available after installing the optional refresh package.")
    service_hour = time_input.hour
    latest_date = pd.to_datetime(data["Date"]).max().normalize()
    forecast_data, forecast_source = forecast_for_date(data, forecast_date)
    rows = []
    for station, station_data in data.groupby("Station"):
        profile = forecast_data[forecast_data["Station"].eq(station) & forecast_data["Hour"].eq(service_hour)]
        status = station_operating_status(data, station, forecast_date, service_hour, closures, operating_hours)
        # For future dates, a generated same-hour fallback is valid evidence of
        # demand coverage even when that exact hour was not observed on the
        # same weekday. Explicit Sundays/closures still take precedence.
        schedule = operating_hours[operating_hours["Station"].eq(station)]
        schedule_allows = schedule.empty or (
            int(schedule["OpenHour"].iloc[0]) <= service_hour < int(schedule["CloseHour"].iloc[0])
        )
        if forecast_date.normalize() > latest_date and not profile.empty and forecast_date.dayofweek != 6 and not (
            not closures.empty and closures["Date"].dt.normalize().eq(forecast_date.normalize()).any()
        ) and schedule_allows:
            status = "Open"
        if status == "Closed":
            rows.append({"Station": station, "Estimated wait": "Closed"})
            continue
        interpolated_forecast = forecast_value_at_time(forecast_data, station, time_input)
        if status == "No forecast" or profile.empty or interpolated_forecast is None:
            rows.append({"Station": station, "Estimated wait": "Unavailable"})
            continue
        forecast = float(interpolated_forecast)
        live_wait = queue_wait_minutes(queue, station, forecast_date, service_hour)
        wait_label = f"{live_wait} minutes" if live_wait is not None else f"{estimated_wait_minutes(forecast, station_data['ForecastUnits'])} minutes"
        rows.append({"Station": station, "Estimated wait": wait_label})
    with wait_col:
        st.markdown(f"### Current Estimated Wait Time · {time_input.strftime('%I:%M %p').lstrip('0')}")
        st.caption(
            f"Showing estimates for {forecast_date.strftime('%B %-d, %Y')} at the selected hour. "
            f"Updated {datetime.now(LOCAL_TZ).strftime('%I:%M %p').lstrip('0')}"
        )
        station_images = {
            "Dining Commons": DINING_COMMONS_IMAGE,
            "Einstein's": EINSTEINS_IMAGE,
            "Panda's": PANDAS_IMAGE,
        }
        station_icons = {"Dining Commons": "🍽️", "Einstein's": "🥯", "Panda's": "🐼"}
        card_columns = st.columns(len(rows), gap="small")
        for column, row in zip(card_columns, rows):
            station = row["Station"]
            with column:
                image_path = station_images.get(station)
                if image_path and image_path.exists():
                    st.image(str(image_path), use_container_width=True)
                wait_text = row["Estimated wait"]
                prefix = "~ " if wait_text not in {"Closed", "Unavailable"} else ""
                st.markdown(
                    f"<div class='wait-card'><div class='wait-name'>{station_icons.get(station, '🍽️')} {station}</div><div class='wait-value'>{prefix}{wait_text}</div></div>",
                    unsafe_allow_html=True,
                )


def manager_portal(st, data: pd.DataFrame, sales: pd.DataFrame, closures: pd.DataFrame, operating_hours: pd.DataFrame) -> None:
    st.markdown(
        """
        <div class="manager-hero">
          <div class="manager-brand">▥ <span>Serve</span>Smart</div>
          <div class="manager-kicker">Decision support for campus dining teams.</div>
          <div class="manager-title">Manager <span>Portal</span></div>
          <div class="manager-benefits">▣ Predict Sales &nbsp;&nbsp;|&nbsp;&nbsp; ⌕ Ask Data Questions &nbsp;&nbsp;|&nbsp;&nbsp; ◫ Plan Operations</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Use the date and time controls for the snapshot. Use the search below when you need a specific answer.")
    plan_col, snapshot_col = st.columns([0.8, 2.2], gap="large")
    with plan_col:
        st.markdown("### Plan Your View")
        today = pd.Timestamp(datetime.now(LOCAL_TZ).date())
        forecast_date = pd.Timestamp(st.date_input("Forecast date", value=today.date(), key="manager_date"))
        forecast_time = st.time_input("Forecast time", value=datetime.now(LOCAL_TZ).replace(second=0, microsecond=0).time(), step=60, key="manager_time")
    selected_hour = forecast_time.hour
    with plan_col:
        st.caption(f"Snapshot: {forecast_date.strftime('%B %-d, %Y')} at {forecast_time.strftime('%I:%M %p').lstrip('0')}. Past times remain estimates for that date; they do not switch to tomorrow.")
    forecast_data, _ = forecast_for_date(data, forecast_date)
    snapshot_rows = []
    for station in data["Station"].dropna().unique():
        status = station_operating_status(data, station, forecast_date, selected_hour, closures, operating_hours)
        match = forecast_data[forecast_data["Station"].eq(station) & forecast_data["Hour"].eq(selected_hour)]
        snapshot_rows.append({
            "Station": station,
            "Predicted sales": "—" if status != "Open" or match.empty else round(float(match["ForecastUnits"].mean())),
            "Status": status,
        })
    with snapshot_col:
        st.markdown(f"### Predicted Sales · {forecast_time.strftime('%I:%M %p').lstrip('0')}")
        st.caption(f"Selected date: {forecast_date.strftime('%B %-d, %Y')}")
        snapshot_columns = st.columns(len(snapshot_rows), gap="small")
        for column, row in zip(snapshot_columns, snapshot_rows):
            with column:
                st.markdown(
                    f"<div class='wait-card'><div class='wait-name'>{row['Station']}</div><div class='wait-value'>{row['Predicted sales']}</div><div>{row['Status']}</div></div>",
                    unsafe_allow_html=True,
                )
        st.dataframe(pd.DataFrame(snapshot_rows), hide_index=True, use_container_width=True)
    st.subheader("Try asking:")
    question_columns = st.columns(len(SUGGESTED_MANAGER_QUESTIONS))
    for column, question in zip(question_columns, SUGGESTED_MANAGER_QUESTIONS):
        if column.button(question, use_container_width=True):
            st.session_state["manager_query_input"] = question
    answer_area = st.container()
    with st.form("manager_question_form", clear_on_submit=False):
        query = st.text_input("Ask a question", placeholder="Ask about predicted sales or top-selling items…", key="manager_query_input")
        submitted = st.form_submit_button("Submit Question", use_container_width=True)
    if submitted:
        st.session_state["last_manager_query"] = query
    query_to_answer = st.session_state.get("last_manager_query", "")
    with answer_area:
        if query_to_answer:
            if "predicted" in query_to_answer.lower() or "forecast" in query_to_answer.lower():
                predicted = answer_predicted_sales_query(query_to_answer, data, forecast_date, closures, operating_hours)
                if predicted.empty:
                    st.warning("No predicted-sales match was found. Try a station and time, such as Panda's at 10 a.m.")
                else:
                    st.subheader("Answer")
                    st.dataframe(predicted, hide_index=True, use_container_width=True)
            else:
                result = answer_manager_query(query_to_answer, sales)
                if result.empty:
                    st.warning("No matching item-level sales were found. Try a location, item, weekday, or time.")
                else:
                    st.subheader("Answer")
                    st.success(f"Found {len(result):,} matching sales rows.")
                    summary = result.groupby("Item", as_index=False).agg(Units_Sold=("Quantity", "sum"), Sales=("NetAmount", "sum")).sort_values("Units_Sold", ascending=False)
                    st.dataframe(summary.head(20), use_container_width=True)
                    st.bar_chart(summary.head(10).set_index("Item")["Units_Sold"])


def analytics_portal(st, data: pd.DataFrame, sales: pd.DataFrame) -> None:
    st.header("Analytics")
    st.caption("Use these charts to understand demand patterns and item performance.")
    c1, c2 = st.columns(2)
    c1.subheader("Demand by hour")
    c1.bar_chart(data.groupby("Hour")["ForecastUnits"].mean())
    c2.subheader("Demand by location")
    c2.bar_chart(data.groupby("Location")["ForecastUnits"].mean())
    if not sales.empty:
        st.subheader("Top-selling items")
        top_items = sales.groupby("Item", as_index=False)["Quantity"].sum().sort_values("Quantity", ascending=False).head(15)
        st.bar_chart(top_items.set_index("Item"))
        st.dataframe(top_items.rename(columns={"Quantity": "Units sold"}), use_container_width=True)
    else:
        st.info("Item-level sales data is not available in this demo environment.")


def main() -> None:
    import streamlit as st
    st.set_page_config(page_title="ServeSmart", page_icon="🍽️", layout="wide")
    hero_background = ""
    if HERO_IMAGE.exists():
        encoded_hero = base64.b64encode(HERO_IMAGE.read_bytes()).decode("ascii")
        hero_background = f"data:image/png;base64,{encoded_hero}"
    st.markdown(
        f"""
        <style>
        .stApp {{ background: radial-gradient(circle at 15% 20%, #143b34 0, #0d191c 34%, #151329 100%); }}
        .welcome-wrap {{ min-height: 520px; padding: 2.8rem 4rem 2rem; border-radius: 28px; background: linear-gradient(180deg, rgba(9, 18, 25, .28), rgba(9, 18, 25, .84)), url('{hero_background}'); background-size: cover; background-position: center; box-shadow: 0 20px 55px rgba(0,0,0,.28); text-align: center; }}
        .brand-mark {{ font-size: 1.05rem; letter-spacing: .12em; text-transform: uppercase; color: #ffd27a; font-weight: 700; }}
        .brand-mark span {{ color: #ff9c23; }}
        .welcome-kicker {{ margin-top: .6rem; color: #e6e9ee; font-size: 1.05rem; letter-spacing: .04em; }}
        .welcome-title {{ margin: 1.2rem 0 .25rem; color: #fff; font-size: clamp(2.7rem, 6vw, 5.2rem); line-height: 1; font-weight: 850; letter-spacing: -.055em; }}
        .welcome-title span {{ color: #ff9d21; }}
        .welcome-caption {{ font-size: 1.35rem; color: #f4f6f7; margin: .8rem 0 2rem; }}
        .benefits {{ display: flex; justify-content: center; gap: 1rem; flex-wrap: wrap; }}
        .benefit {{ min-width: 185px; padding: .8rem 1.2rem; border: 1px solid rgba(255,196,106,.55); border-radius: 999px; background: rgba(10, 28, 29, .72); color: #fff; font-weight: 700; }}
        .portal-card {{ min-height: 245px; padding: 1.75rem 2rem; border-radius: 24px; border: 1px solid rgba(255,255,255,.18); box-shadow: 0 12px 30px rgba(0,0,0,.18); }}
        .customer-card {{ background: linear-gradient(135deg, rgba(255,251,239,.96), rgba(218,242,216,.86)); color: #0d503c; }}
        .manager-card {{ background: linear-gradient(135deg, rgba(251,248,255,.96), rgba(224,211,250,.9)); color: #41206f; }}
        .portal-icon {{ font-size: 2.2rem; }}
        .portal-title {{ font-size: 1.75rem; font-weight: 800; margin: .55rem 0 .4rem; }}
        .portal-text {{ line-height: 1.45; min-height: 58px; }}
        .location-heading {{ text-align: center; color: #fff; font-size: 1.35rem; font-weight: 750; margin: 1.7rem 0 .8rem; }}
        .locations {{ display: flex; justify-content: center; gap: .9rem; flex-wrap: wrap; }}
        .location {{ background: rgba(255,255,255,.94); color: #20222a; padding: .75rem 1.25rem; border-radius: 999px; font-weight: 750; min-width: 165px; text-align: center; }}
        .stButton > button {{ border-radius: 999px; font-weight: 750; min-height: 2.7rem; }}
        div[data-testid="column"]:has(.customer-card) .stButton > button {{ background: #075c43; color: white; border: 0; }}
        div[data-testid="column"]:has(.manager-card) .stButton > button {{ background: #4b2584; color: white; border: 0; }}
        .customer-hero {{ padding: 2rem 2.4rem 1.7rem; border-radius: 26px; color: white; background: linear-gradient(120deg, #064b39, #0d6b4d 58%, #12372d); box-shadow: 0 15px 35px rgba(0,0,0,.2); }}
        .customer-brand {{ font-size: 1.15rem; font-weight: 800; letter-spacing: .08em; text-transform: uppercase; }}
        .customer-brand span, .customer-title span {{ color: #ffad19; }}
        .customer-kicker {{ margin-top: 1rem; font-size: 1.05rem; color: #d9f0e3; }}
        .customer-title {{ margin-top: .35rem; font-size: clamp(2.4rem, 5vw, 4.5rem); line-height: 1; font-weight: 850; letter-spacing: -.04em; }}
        .customer-benefits {{ margin-top: 1.6rem; color: #fff1b5; font-weight: 700; font-size: 1rem; }}
        .wait-card {{ text-align: center; background: #f8f1e4; color: #123f32; padding: .75rem .5rem 1rem; border-radius: 0 0 18px 18px; min-height: 78px; }}
        .wait-name {{ font-size: 1.05rem; font-weight: 800; }}
        .wait-value {{ display: inline-block; margin-top: .55rem; padding: .45rem 1.1rem; border-radius: 999px; background: #08704e; color: white; font-weight: 800; }}
        .manager-hero {{ padding: 2rem 2.4rem 1.7rem; border-radius: 26px; color: white; background: linear-gradient(120deg, #34215f, #5d318a 58%, #1d244b); box-shadow: 0 15px 35px rgba(0,0,0,.2); }}
        .manager-brand {{ font-size: 1.15rem; font-weight: 800; letter-spacing: .08em; text-transform: uppercase; }}
        .manager-brand span, .manager-title span {{ color: #ffad19; }}
        .manager-kicker {{ margin-top: 1rem; font-size: 1.05rem; color: #eee3ff; }}
        .manager-title {{ margin-top: .35rem; font-size: clamp(2.4rem, 5vw, 4.5rem); line-height: 1; font-weight: 850; letter-spacing: -.04em; }}
        .manager-benefits {{ margin-top: 1.6rem; color: #ffe5a9; font-weight: 700; font-size: 1rem; }}
        .manager-panel {{ background: rgba(255,255,255,.07); border: 1px solid rgba(255,255,255,.12); border-radius: 20px; padding: 1.25rem; }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    data = load_demo_data()
    sales = load_item_sales()
    queue = load_queue_service_times()
    closures = load_closure_calendar()
    operating_hours = load_operating_hours()
    st.sidebar.title("Portal")
    portal_options = ["Home", "Customer Portal", "Manager Portal", "Analytics"]
    portal = st.sidebar.radio("Choose a view", portal_options, index=portal_options.index(st.session_state.get("portal_choice", "Home")))
    if portal == "Home":
        st.markdown(
            """
            <div class="welcome-wrap">
              <div class="brand-mark">🍴 <span>Serve</span>Smart</div>
              <div class="welcome-kicker">AI-Powered Campus Dining Forecasting</div>
              <div class="welcome-title">Welcome to <span>ServeSmart</span></div>
              <div class="welcome-caption">Smarter dining decisions for every visit.</div>
              <div class="benefits">
                <div class="benefit">📊 Predict Demand</div>
                <div class="benefit">👥 Reduce Wait Time</div>
                <div class="benefit">🌿 Optimize Operations</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.markdown(
                """
                <div class="portal-card customer-card">
                  <div class="portal-icon">♧</div>
                  <div class="portal-title">Customer Portal</div>
                  <div class="portal-text">Check estimated wait times at your favorite dining locations before you visit.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("Open Customer Portal  →", key="home_customer_portal", use_container_width=True):
                st.session_state["portal_choice"] = "Customer Portal"
                st.rerun()
        with c2:
            st.markdown(
                """
                <div class="portal-card manager-card">
                  <div class="portal-icon">▥</div>
                  <div class="portal-title">Manager Portal</div>
                  <div class="portal-text">View today’s predicted sales, analyze trends, and search manager questions.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("Open Manager Portal  →", key="home_manager_portal", use_container_width=True):
                st.session_state["portal_choice"] = "Manager Portal"
                st.rerun()
        st.markdown(
            """
            <div class="location-heading">Our Dining Locations at <span style="color:#ff9d21">CSUEB</span></div>
            <div class="locations">
              <div class="location">🍽️ Dining Commons</div>
              <div class="location">🥯 Einstein Bros. Bagels</div>
              <div class="location">🐼 Panda Express</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif portal == "Customer Portal":
        st.title("ServeSmart")
        customer_portal(st, data, queue, closures, operating_hours)
    elif portal == "Manager Portal":
        st.title("ServeSmart")
        manager_portal(st, data, sales, closures, operating_hours)
    else:
        st.title("ServeSmart")
        if BASELINE_COMPARISON_FILE.exists():
            st.info("BASELINE MODE: Seasonal Naive is the current selected comparison winner.")
        else:
            st.warning("MODEL PREVIEW MODE: forecast values remain preview estimates until the refreshed baseline comparison selects the final model.")
        analytics_portal(st, data, sales)


if __name__ == "__main__":
    main()
