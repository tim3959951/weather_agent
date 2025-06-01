import os
import requests
import joblib
import time
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from dateutil import parser as date_parser
from typing import Union
from geopy.geocoders import Nominatim
from timezonefinder import TimezoneFinder
from tabulate import tabulate
import pandas as pd
import matplotlib.pyplot as plt
from langchain.schema import AIMessage
try:
    from langchain_openai import ChatOpenAI
except ImportError:
    from langchain.chat_models import ChatOpenAI


os.environ["OPENAI_API_KEY"] = "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"
os.environ["WEATHER_API_KEY"] = "b9b44f4c0e8949bb95a90524250204"
#os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
#os.environ["WEATHER_API_KEY"] = os.getenv("WEATHER_API_KEY")
try:
    llm_gpt4 = ChatOpenAI(model="gpt-4o-mini", temperature=0)
except Exception:
    try:
        llm_gpt4 = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    except Exception as e:
        print(f"LangChain initialization failed: {e}")
        llm_gpt4 = None
        
def location_to_timezone(location: str) -> str:
    try:
        geo = Nominatim(user_agent="time_agent_demo")
        loc = geo.geocode(location)
        if not loc:
            return "Europe/London"
        tf = TimezoneFinder()
        return tf.timezone_at(lng=loc.longitude, lat=loc.latitude) or "Europe/London"
    except Exception:
        return "Europe/London"

def get_time_tool2(query: str) -> tuple[datetime, int, str]:
    try:
        location_prompt = f"""
        You are a location extractor. Given a user's query about time or date, return the location mentioned in it.
        If not found, return "London".

        Query: "{query}"
        """
        location_response = llm_gpt4.invoke(location_prompt)
        location = location_response.content.strip() if isinstance(location_response, AIMessage) else str(location_response).strip()

        #print(f"[DEBUG] Extracted Location: {location}")
        tz_str = location_to_timezone(location)
        #print(f"[DEBUG] Timezone: {tz_str}")
        now = datetime.now(ZoneInfo(tz_str))
        #print(f"[DEBUG] Local Time at {location}: {now}")

        examples = [
            # Pure hourly relative expression
            ("five hours later",         f"START_TIME: {now.strftime('%Y-%m-%d %H:%M')}\nDURATION_HOURS: 5"),
            ("later",                    f"START_TIME: {now.strftime('%Y-%m-%d %H:%M')}\nDURATION_HOURS: 2"),
            ("soon",                     f"START_TIME: {now.strftime('%Y-%m-%d %H:%M')}\nDURATION_HOURS: 1"),
            ("shortly",                  f"START_TIME: {now.strftime('%Y-%m-%d %H:%M')}\nDURATION_HOURS: 1"),
            ("after a while",            f"START_TIME: {now.strftime('%Y-%m-%d %H:%M')}\nDURATION_HOURS: 1"),

            # Expressions throughout the day (no specific time)
            ("today",                    f"START_TIME: {now.replace(hour=0,minute=0).strftime('%Y-%m-%d %H:%M')}\nDURATION_HOURS: 24"),
            ("tomorrow",                 f"START_TIME: {(now + timedelta(days=1)).replace(hour=0,minute=0).strftime('%Y-%m-%d %H:%M')}\nDURATION_HOURS: 24"),
            ("yesterday",                f"START_TIME: {(now - timedelta(days=1)).replace(hour=0,minute=0).strftime('%Y-%m-%d %H:%M')}\nDURATION_HOURS: 24"),
            ("the day before yesterday", f"START_TIME: {(now - timedelta(days=2)).replace(hour=0,minute=0).strftime('%Y-%m-%d %H:%M')}\nDURATION_HOURS: 24"),
            ("the day after tomorrow",   f"START_TIME: {(now + timedelta(days=2)).replace(hour=0,minute=0).strftime('%Y-%m-%d %H:%M')}\nDURATION_HOURS: 24"),
            ("昨天",                     f"START_TIME: {(now - timedelta(days=1)).replace(hour=0,minute=0).strftime('%Y-%m-%d %H:%M')}\nDURATION_HOURS: 24"),
            ("今天",                     f"START_TIME: {now.replace(hour=0,minute=0).strftime('%Y-%m-%d %H:%M')}\nDURATION_HOURS: 24"),
            ("明天",                     f"START_TIME: {(now + timedelta(days=1)).replace(hour=0,minute=0).strftime('%Y-%m-%d %H:%M')}\nDURATION_HOURS: 24"),
            ("前天",                     f"START_TIME: {(now - timedelta(days=2)).replace(hour=0,minute=0).strftime('%Y-%m-%d %H:%M')}\nDURATION_HOURS: 24"),

            # Specific time point (single point)
            ("tomorrow at 3pm",          f"START_TIME: {(now + timedelta(days=1)).replace(hour=15,minute=0).strftime('%Y-%m-%d %H:%M')}\nDURATION_HOURS: 1"),
            ("tomorrow at 10am",         f"START_TIME: {(now + timedelta(days=1)).replace(hour=10,minute=0).strftime('%Y-%m-%d %H:%M')}\nDURATION_HOURS: 1"),
            ("yesterday at 5pm",         f"START_TIME: {(now - timedelta(days=1)).replace(hour=17,minute=0).strftime('%Y-%m-%d %H:%M')}\nDURATION_HOURS: 1"),
            ("today at 6pm",             f"START_TIME: {now.replace(hour=18,minute=0).strftime('%Y-%m-%d %H:%M')}\nDURATION_HOURS: 1"),
            ("the day after tomorrow at 10am", f"START_TIME: {(now + timedelta(days=2)).replace(hour=10,minute=0).strftime('%Y-%m-%d %H:%M')}\nDURATION_HOURS: 1"),
            ("昨天下午五點",               f"START_TIME: {(now - timedelta(days=1)).replace(hour=17,minute=0).strftime('%Y-%m-%d %H:%M')}\nDURATION_HOURS: 1"),
            ("昨天早上八點",               f"START_TIME: {(now - timedelta(days=1)).replace(hour=8,minute=0).strftime('%Y-%m-%d %H:%M')}\nDURATION_HOURS: 1"),
            ("下週一下午三點",             f"START_TIME: {(now + timedelta(days=(7 - now.weekday() + 0) % 7)).replace(hour=15,minute=0).strftime('%Y-%m-%d %H:%M')}\nDURATION_HOURS: 1"),
            ("last Monday 9am",          f"START_TIME: {(now - timedelta(days=(now.weekday() + 7))).replace(hour=9,minute=0).strftime('%Y-%m-%d %H:%M')}\nDURATION_HOURS: 1"),

            # Day of the week (all day)
            ("next Monday",              f"START_TIME: {(now + timedelta(days=(7 - now.weekday()))).replace(hour=0,minute=0).strftime('%Y-%m-%d %H:%M')}\nDURATION_HOURS: 24"),
            ("last Friday",              f"START_TIME: {(now - timedelta(days=(now.weekday() - 4 + 7) % 7)).replace(hour=0,minute=0).strftime('%Y-%m-%d %H:%M')}\nDURATION_HOURS: 24"),
            ("next Friday",              f"START_TIME: {(now + timedelta(days=(4 - now.weekday() + 7) % 7)).replace(hour=0,minute=0).strftime('%Y-%m-%d %H:%M')}\nDURATION_HOURS: 24"),
            ("this Monday",              f"START_TIME: {(now - timedelta(days=now.weekday())).replace(hour=0,minute=0).strftime('%Y-%m-%d %H:%M')}\nDURATION_HOURS: 24"),
            ("this Sunday",              f"START_TIME: {(now + timedelta(days=(6 - now.weekday()))).replace(hour=0,minute=0).strftime('%Y-%m-%d %H:%M')}\nDURATION_HOURS: 24"),

            # X days before/after (all day)
            ("5 days ago",               f"START_TIME: {(now - timedelta(days=5)).replace(hour=0,minute=0).strftime('%Y-%m-%d %H:%M')}\nDURATION_HOURS: 24"),
            ("3 days ago",               f"START_TIME: {(now - timedelta(days=3)).replace(hour=0,minute=0).strftime('%Y-%m-%d %H:%M')}\nDURATION_HOURS: 24"),
            ("in 5 days",                f"START_TIME: {(now + timedelta(days=5)).replace(hour=0,minute=0).strftime('%Y-%m-%d %H:%M')}\nDURATION_HOURS: 24"),

            # Multi-day range
            ("past 3 days",              f"START_TIME: {(now - timedelta(days=2)).strftime('%Y-%m-%d %H:%M')}\nDURATION_HOURS: 72"),
            ("last 5 days",              f"START_TIME: {(now - timedelta(days=4)).strftime('%Y-%m-%d %H:%M')}\nDURATION_HOURS: 120"),
            ("next 7 days",              f"START_TIME: {now.strftime('%Y-%m-%d %H:%M')}\nDURATION_HOURS: 168"),
            ("past 3 days",              f"START_TIME: {(now - timedelta(days=3)).strftime('%Y-%m-%d %H:%M')}\nDURATION_HOURS: 72"),
            ("last 5 days",              f"START_TIME: {(now - timedelta(days=5)).strftime('%Y-%m-%d %H:%M')}\nDURATION_HOURS: 120"),
            ("last 10 days",             f"START_TIME: {(now - timedelta(days=10)).strftime('%Y-%m-%d %H:%M')}\nDURATION_HOURS: 240"),

            # Weekly/Monthly Range
            ("this week",                f"START_TIME: {(now - timedelta(days=now.weekday())).replace(hour=0,minute=0).strftime('%Y-%m-%d %H:%M')}\nDURATION_HOURS: 168"),
            ("last week",                f"START_TIME: {(now - timedelta(days=now.weekday() + 7)).replace(hour=0,minute=0).strftime('%Y-%m-%d %H:%M')}\nDURATION_HOURS: 168"),
            ("next week",                f"START_TIME: {(now + timedelta(days=(7 - now.weekday()))).replace(hour=0,minute=0).strftime('%Y-%m-%d %H:%M')}\nDURATION_HOURS: 168"),
            ("last month",               f"START_TIME: {(now - timedelta(days=30)).replace(hour=0,minute=0).strftime('%Y-%m-%d %H:%M')}\nDURATION_HOURS: 720"),
            ("本週",                     f"START_TIME: {(now - timedelta(days=now.weekday())).replace(hour=0,minute=0).strftime('%Y-%m-%d %H:%M')}\nDURATION_HOURS: 168"),
            ("上週",                     f"START_TIME: {(now - timedelta(days=now.weekday() + 7)).replace(hour=0,minute=0).strftime('%Y-%m-%d %H:%M')}\nDURATION_HOURS: 168"),

            # Past X hours
            ("過去 24 小時",               f"START_TIME: {(now - timedelta(hours=24)).strftime('%Y-%m-%d %H:%M')}\nDURATION_HOURS: 24"),
            ("past 48 hours",            f"START_TIME: {(now - timedelta(hours=48)).strftime('%Y-%m-%d %H:%M')}\nDURATION_HOURS: 48"),
            ("last 12 hours",            f"START_TIME: {(now - timedelta(hours=12)).strftime('%Y-%m-%d %H:%M')}\nDURATION_HOURS: 12"),

            # The next X hours
            ("in 10 hours",              f"START_TIME: {now.strftime('%Y-%m-%d %H:%M')}\nDURATION_HOURS: 10"),
            ("in 2 hours",               f"START_TIME: {now.strftime('%Y-%m-%d %H:%M')}\nDURATION_HOURS: 2"),
            ("in one hour",              f"START_TIME: {now.strftime('%Y-%m-%d %H:%M')}\nDURATION_HOURS: 1"),
            ("next 2 hours",             f"START_TIME: {now.strftime('%Y-%m-%d %H:%M')}\nDURATION_HOURS: 2"),
            ("next 8 hours",             f"START_TIME: {now.strftime('%Y-%m-%d %H:%M')}\nDURATION_HOURS: 8"),

            # A few minutes of expression (regarded as one hour)
            ("in 30 minutes",            f"START_TIME: {now.strftime('%Y-%m-%d %H:%M')}\nDURATION_HOURS: 1"),
            ("in a few minutes",         f"START_TIME: {now.strftime('%Y-%m-%d %H:%M')}\nDURATION_HOURS: 1"),

            # Special Segment
            ("later this evening",       f"START_TIME: {now.replace(hour=20,minute=0).strftime('%Y-%m-%d %H:%M')}\nDURATION_HOURS: 1"),
            ("this weekend",             f"START_TIME: {(now + timedelta(days=(5 - now.weekday()) % 7)).replace(hour=0,minute=0).strftime('%Y-%m-%d %H:%M')}\nDURATION_HOURS: 48"),
            ("next weekend",             f"START_TIME: {(now + timedelta(days=((5 - now.weekday()) % 7) + 7)).replace(hour=0,minute=0).strftime('%Y-%m-%d %H:%M')}\nDURATION_HOURS: 48"),
            ("tonight",                  f"START_TIME: {now.replace(hour=20,minute=0).strftime('%Y-%m-%d %H:%M')}\nDURATION_HOURS: 4"),
            ("this morning",             f"START_TIME: {now.replace(hour=6,minute=0).strftime('%Y-%m-%d %H:%M')}\nDURATION_HOURS: 6"),
            ("this afternoon",           f"START_TIME: {now.replace(hour=12,minute=0).strftime('%Y-%m-%d %H:%M')}\nDURATION_HOURS: 6"),

            # Current Time
            ("now",                      f"START_TIME: {now.strftime('%Y-%m-%d %H:%M')}\nDURATION_HOURS: 1"),
            ("right now",                f"START_TIME: {now.strftime('%Y-%m-%d %H:%M')}\nDURATION_HOURS: 1"),
            ("現在",                      f"START_TIME: {now.strftime('%Y-%m-%d %H:%M')}\nDURATION_HOURS: 1"),

            # Scope of Expression
            ("from today 15:00 to 20:00",  f"START_TIME: {now.replace(hour=15,minute=0).strftime('%Y-%m-%d %H:%M')}\nDURATION_HOURS: 5"),
            ("從今天下午 3 點到晚上 8 點",    f"START_TIME: {now.replace(hour=15,minute=0).strftime('%Y-%m-%d %H:%M')}\nDURATION_HOURS: 5"),
            ("from tomorrow 14:00 to tomorrow 18:00", f"START_TIME: {(now + timedelta(days=1)).replace(hour=14,minute=0).strftime('%Y-%m-%d %H:%M')}\nDURATION_HOURS: 4"),
            ("from 3pm today to 2am tomorrow", f"START_TIME: {now.replace(hour=15,minute=0).strftime('%Y-%m-%d %H:%M')}\nDURATION_HOURS: 11"),
            ("from 4pm to 8pm today",    f"START_TIME: {now.replace(hour=16,minute=0).strftime('%Y-%m-%d %H:%M')}\nDURATION_HOURS: 4"),
            ("Provide a  6-day weather summary for Tokyo ending today.", f"START_TIME: {(now - timedelta(days=5)).replace(hour=0,minute=0).strftime('%Y-%m-%d %H:%M')}\nDURATION_HOURS: 144"),
            
        ]

        # few‐shot prompt
        examples_header = f"""Assume the current local time in {location} is exactly:
**{now.strftime('%Y-%m-%d %H:%M')}** (timezone: {tz_str})

Use this exact time to reason all examples below.
"""
        examples_str = "\n".join([f'User Query: "{q}"\n→ {out}' for q, out in examples])

        time_query_prompt = f"""
You are a timezone-aware time reasoner. Based on the user's query, calculate:
1. START_TIME (format: YYYY-MM-DD HH:MM) in the local timezone.
2. DURATION_HOURS (integer hours) for how many hours this query spans.

Examples:
{examples_str}

Now process:
User Query: "{query}"
→
"""

        time_response = llm_gpt4.invoke(time_query_prompt)
        resp_lines = time_response.content.strip().splitlines()

        # # Parse START_TIME and DURATION_HOURS from the return.
        start_time = now
        duration_hours = 1
        for line in resp_lines:
            if line.startswith("START_TIME:"):
                t_str = line.split(":", 1)[1].strip()
                try:
                    start_time = datetime.strptime(t_str, "%Y-%m-%d %H:%M")
                    start_time = start_time.replace(tzinfo=ZoneInfo(tz_str))
                except:
                    pass
            elif line.startswith("DURATION_HOURS:"):
                try:
                    duration_hours = int(line.split(":", 1)[1].strip())
                except:
                    pass

        return start_time, duration_hours, location

    except Exception as e:
        # If parsing fails, fallback to "current time + 1 hour".
        tz_str_fallback = "Europe/London"
        try:
            tz_str_fallback = location_to_timezone(location)
        except:
            pass
        now = datetime.now(ZoneInfo(tz_str_fallback))
        return now, 1, location


def render_chart(df: pd.DataFrame, location: str, title: str = "Weather Forecast") -> str:   
    try:
        chart_path = "/tmp/weather_chart.png"
        if os.path.exists(chart_path):
            os.remove(chart_path)
        # Converting time fields to datetime objects
        if "time" in df.columns:
            df_plot = df.copy()
            df_plot["datetime"] = pd.to_datetime(df_plot["time"])
            
            # Create Submap
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f"{title}", fontsize=16)
            
            # Temperature Chart
            if "temp_c" in df_plot.columns:
                axes[0, 0].plot(df_plot["datetime"], df_plot["temp_c"], 
                               marker='o', color='red', linewidth=2, markersize=4)
                axes[0, 0].set_title("Temperature (°C)")
                axes[0, 0].set_ylabel("Temperature (°C)")
                axes[0, 0].grid(True, alpha=0.3)
                axes[0, 0].tick_params(axis='x', rotation=45)
            
            # Humidity Chart
            if "humidity" in df_plot.columns:
                axes[0, 1].plot(df_plot["datetime"], df_plot["humidity"], 
                               marker='s', color='blue', linewidth=2, markersize=4)
                axes[0, 1].set_title("Humidity (%)")
                axes[0, 1].set_ylabel("Humidity (%)")
                axes[0, 1].grid(True, alpha=0.3)
                axes[0, 1].tick_params(axis='x', rotation=45)
            
            # Wind Speed Chart
            if "wind_kph" in df_plot.columns:
                axes[1, 0].plot(df_plot["datetime"], df_plot["wind_kph"], 
                               marker='^', color='green', linewidth=2, markersize=4)
                axes[1, 0].set_title("Wind Speed (kph)")
                axes[1, 0].set_ylabel("Wind Speed (kph)")
                axes[1, 0].grid(True, alpha=0.3)
                axes[1, 0].tick_params(axis='x', rotation=45)
            
            # Rainfall Probability Chart
            if "chance_of_rain" in df_plot.columns:
                axes[1, 1].plot(df_plot["datetime"], df_plot["chance_of_rain"], 
                               marker='d', color='purple', linewidth=2, markersize=4)
                axes[1, 1].set_title("Chance of Rain (%)")
                axes[1, 1].set_ylabel("Chance of Rain (%)")
                axes[1, 1].grid(True, alpha=0.3)
                axes[1, 1].tick_params(axis='x', rotation=45)
            
            
            for ax in axes.flat:
                if len(df_plot) > 1:
                    # Adjust the timeline format according to the data range
                    time_span = (df_plot["datetime"].iloc[-1] - df_plot["datetime"].iloc[0]).total_seconds() / 3600
                    
                    if time_span <= 24:  # In 24 hours, show hours
                        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                        ax.xaxis.set_major_locator(mdates.HourLocator(interval=max(1, int(time_span/6))))
                    elif time_span <= 168:  # Within one week, date and hour are displayed
                        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H'))
                        ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
                    else:  # More than one week, only the date is displayed
                        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
                        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
            
            timestamp = int(time.time())
            plt.tight_layout()
            chart_path = "/tmp/weather_chart.png"  
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            print(f" Chart saved to: {chart_path}")
            plt.close()
            
            return chart_path
            
        else:
            # Returns an error if there is no time field
            return "Error: No time column found in DataFrame"
            
    except Exception as e:
        plt.close()  # Ensure closure charts
        return f"Chart generation error: {str(e)}"


def render_table(df: pd.DataFrame) -> str:
    try:
        if df.empty:
            return "No data available"
        
        # Create a more readable table format
        display_df = df.copy()
        
        # Rename fields to more friendly names
        column_mapping = {
            "time": "Time",
            "temp_c": "Temp (°C)",
            "feelslike_c": "Feels Like (°C)",
            "humidity": "Humidity (%)",
            "condition": "Condition",
            "chance_of_rain": "Rain (%)",
            "chance_of_snow": "Snow (%)",
            "wind_kph": "Wind (kph)",
            "uv": "UV Index",
            "cloud": "Cloud (%)",
            "vis_km": "Visibility (km)"
        }
        
        # Rename existing columns
        for old_name, new_name in column_mapping.items():
            if old_name in display_df.columns:
                display_df = display_df.rename(columns={old_name: new_name})
        
        # Retain 1 decimal place in the numeric field
        numeric_columns = display_df.select_dtypes(include=['float64', 'float32']).columns
        for col in numeric_columns:
            display_df[col] = display_df[col].round(1)
        
        # 使用 tabulate 創建完美對齊的表格
        return tabulate(
            display_df, 
            headers='keys', 
            tablefmt='grid',
            showindex=False,
            numalign="center",
            stralign="center"
        )
        
    except Exception as e:
        return f"Table generation error: {str(e)}"

def render_text_summary(df: pd.DataFrame, location: str, time_type: str) -> str:
    try:
        lines = [f"Weather {time_type} summary for {location}:"]
        
        for _, row in df.iterrows():
            if "time" in row and "temp_c" in row and "humidity" in row:
                time_str = row["time"]
                temp = row["temp_c"]
                humidity = row["humidity"]
                condition = row.get("condition", "N/A")
                
                lines.append(f"{time_str}: {temp}°C, {humidity}% humidity, {condition}")
        
        return "\n".join(lines)
        
    except Exception as e:
        return f"Text summary generation error: {str(e)}"

def predict_weather_fallback(location: str, target_dt: datetime) -> dict:
    try:
        # Load previously trained model and metadata 
        all_models = joblib.load("weather_multi_parameter_models.joblib")
        forecast_horizons = joblib.load("weather_forecast_horizons.joblib")
        
        # Get the actual weather of the location as input to the model
        weather_api_key = os.environ.get("WEATHER_API_KEY")
        current_url = f"http://api.weatherapi.com/v1/current.json?key={weather_api_key}&q={location}"
        current_data = requests.get(current_url).json()
        current = current_data["current"]
        
        # Prepare X_input: a DataFrame that is consistent with the "features" of the training
        X_input = pd.DataFrame([{
            "temp": current["temp_c"],
            "humidity": current["humidity"],
            "pressure": current["pressure_mb"],
            "wind": current["wind_kph"] / 3.6, # km/h → m/s
            "hour": target_dt.hour,
            "day": target_dt.day,
            "month": target_dt.month,
            "day_of_week": target_dt.weekday(),
            "is_weekend": 1 if target_dt.weekday() >= 5 else 0
        }])
        
        # fix the offset-naive vs offset-aware bug 
        # Use location to get the timezone string, then take the now_aware of the current "offset-aware"
        tz_str_model = location_to_timezone(location)
        now_aware = datetime.now(ZoneInfo(tz_str_model))
        hours_ahead = int((target_dt - now_aware).total_seconds() / 3600)
        
        # Find the nearest trained model to hours_ahead horizon
        closest_horizon = min(forecast_horizons, key=lambda h: abs(h - hours_ahead))
        
        # The corresponding sub-models are used to make their own predictions
        predictions = {}
        
        if closest_horizon in all_models["temperature"]:
            predictions["temp_c"] = float(all_models["temperature"][closest_horizon].predict(X_input)[0])
        else:
            # If the model doesn't have the horizon you want, fallback to 0 or some other default value
            predictions["temp_c"] = current["temp_c"]
        
        if closest_horizon in all_models["humidity"]:
            predictions["humidity"] = float(all_models["humidity"][closest_horizon].predict(X_input)[0])
        else:
            predictions["humidity"] = current["humidity"]
        
        if closest_horizon in all_models["pressure"]:
            predictions["pressure"] = float(all_models["pressure"][closest_horizon].predict(X_input)[0])
        else:
            predictions["pressure"] = current["pressure_mb"]
        
        if closest_horizon in all_models["wind"]:
            pred_wind_ms = float(all_models["wind"][closest_horizon].predict(X_input)[0])
            predictions["wind_kph"] = pred_wind_ms * 3.6   # m/s → km/h
        else:
            predictions["wind_kph"] = current["wind_kph"]
        
        # Use predicted temperature and humidity to extrapolate other parameters (rain or shine, cloud cover etc.) 
        temp = predictions["temp_c"]
        humidity = predictions["humidity"]
        
        if humidity > 80:
            condition = "Cloudy" if temp < 25 else "Partly cloudy"
            chance_of_rain = 60 if humidity > 90 else 40
        elif temp > 30:
            condition = "Sunny"
            chance_of_rain = 5
        else:
            condition = "Clear" if humidity < 60 else "Partly cloudy"
            chance_of_rain = 20
        
        # Assembles the complete dictionary to be returned to the Agent
        return {
            "temp_c": round(predictions["temp_c"], 1),
            "feelslike_c": round(predictions["temp_c"] - 2, 1),
            "humidity": round(predictions["humidity"]),
            "pressure": round(predictions["pressure"]),
            "wind_kph": round(predictions["wind_kph"], 1),
            "condition": condition,
            "chance_of_rain": chance_of_rain,
            "uv": 5,                 # fix
            "cloud": round(humidity * 0.8),
            "vis_km": 10 if humidity < 80 else 5
        }
        
    except Exception as e:
        print(f"ML prediction error: {e}")
        # If the model or other link throws an exception, it returns a simple default prediction
        return {
            "temp_c": 25,
            "feelslike_c": 23,
            "humidity": 60,
            "pressure": 1013,
            "wind_kph": 15,
            "condition": "Partly cloudy",
            "chance_of_rain": 20,
            "uv": 5,
            "cloud": 40,
            "vis_km": 10
        }
    
def weather_agent_tool(query: str) -> str:
    try:
        weather_api_key = os.environ.get("WEATHER_API_KEY")
        if not weather_api_key:
            return "Weather API key not found. Please set WEATHER_API_KEY env variable."
            
        # Use get_time_tool2 to get (start_dt, duration_hours, location)
        time_result = get_time_tool2(query)
        if not isinstance(time_result, tuple) or len(time_result) != 3:
            return "Error in retrieving time information."
        start_dt, duration_hours, location = time_result

        tz_str = location_to_timezone(location)
        start_dt = start_dt.replace(tzinfo=ZoneInfo(tz_str))
        now = datetime.now(ZoneInfo(tz_str))
        end_dt = start_dt + timedelta(hours=duration_hours)

        if start_dt < now - timedelta(days=7):
            return "Only supports up to 7 days of historical data."
        if end_dt > now + timedelta(days=13):
            return "Only supports up to 13 days of future forecast."

        weather_data = []
        current_time = start_dt
        while current_time <= end_dt:
            time_diff_hours = (current_time - now).total_seconds() / 3600
    
            if time_diff_hours > 72:
                # Over the next 3 days, using ML modelling
                try:
                    model_result = predict_weather_fallback(location, current_time)
                    weather_point = {
                        "time": current_time.strftime('%Y-%m-%d %H:%M'),
                        "condition": model_result["condition"],
                        "temp_c": model_result["temp_c"],
                        "feelslike_c": model_result["feelslike_c"],
                        "humidity": model_result["humidity"],
                        "chance_of_rain": model_result.get("chance_of_rain", 0),
                        "chance_of_snow": model_result.get("chance_of_snow", 0),
                        "wind_kph": model_result.get("wind_kph", 0),
                        "uv": model_result.get("uv", 0),
                        "cloud": model_result.get("cloud", 0),
                        "vis_km": model_result.get("vis_km", 0)
                    }
                except Exception as e:
                    # ML model failing fallback
                    weather_point = {
                        "time": current_time.strftime('%Y-%m-%d %H:%M'),
                        "condition": "Partly cloudy",
                        "temp_c": 20,
                        "feelslike_c": 18,
                        "humidity": 60,
                        "chance_of_rain": 20,
                        "chance_of_snow": 0,
                        "wind_kph": 15,
                        "uv": 5,
                        "cloud": 40,
                        "vis_km": 10
                    }
            else:
                # Use WeatherAPI within API support.
                try:
                    if time_diff_hours < -168:  
                        current_time += timedelta(hours=1)
                        continue
                    elif time_diff_hours < -24:  # Over 1 day ago with historical APIs
                        url = f"http://api.weatherapi.com/v1/history.json?key={weather_api_key}&q={location}&dt={current_time.strftime('%Y-%m-%d')}"
                    else:  # Use the Predictive API for everything within 1 day (including now)
                        url = f"http://api.weatherapi.com/v1/forecast.json?key={weather_api_key}&q={location}&days=3&aqi=no&alerts=no"
                    
                    data = requests.get(url).json()
                    # print(f"[DEBUG] Processing time: {current_time}, time difference: {time_diff_hours:.1f}hour")
                    
                    # Collect all available hourly data
                    forecast_hours = []
                    if "forecast" in data:
                        for day in data["forecast"]["forecastday"]:
                            for hour in day["hour"]:
                                forecast_hours.append(hour)
                    
                    # Find the closest hour
                    min_diff = float("inf")
                    closest_hour = None
                    for hour_data in forecast_hours:
                        hour_dt = date_parser.parse(hour_data["time"]).replace(tzinfo=ZoneInfo(tz_str))
                        diff = abs((hour_dt - current_time).total_seconds())
                        if diff < min_diff:
                            min_diff = diff
                            closest_hour = hour_data
                    
                    if closest_hour:
                        #print(f"[DEBUG] best match: Objectives{current_time.strftime('%H:%M')}, Selected{date_parser.parse(closest_hour['time']).strftime('%H:%M')}, Temperature{closest_hour['temp_c']}°C")
                        weather_point = {
                            "time": current_time.strftime('%Y-%m-%d %H:%M'),
                            "condition": closest_hour["condition"]["text"],
                            "temp_c": closest_hour["temp_c"],
                            "feelslike_c": closest_hour["feelslike_c"],
                            "humidity": closest_hour["humidity"],
                            "chance_of_rain": closest_hour.get("chance_of_rain", 0),
                            "chance_of_snow": closest_hour.get("chance_of_snow", 0),
                            "wind_kph": closest_hour.get("wind_kph", 0),
                            "uv": closest_hour.get("uv", 0),
                            "cloud": closest_hour.get("cloud", 0),
                            "vis_km": closest_hour.get("vis_km", 0)
                        }
                    else:
                        # API fallback when no data is available
                        weather_point = {
                            "time": current_time.strftime('%Y-%m-%d %H:%M'),
                            "condition": "No data",
                            "temp_c": 20,
                            "feelslike_c": 18,
                            "humidity": 60,
                            "chance_of_rain": 0,
                            "chance_of_snow": 0,
                            "wind_kph": 0,
                            "uv": 0,
                            "cloud": 0,
                            "vis_km": 0
                        }
                except Exception as e:
                    # fallback when the API fails
                    weather_point = {
                        "time": current_time.strftime('%Y-%m-%d %H:%M'),
                        "condition": "API Error",
                        "temp_c": 20,
                        "feelslike_c": 18,
                        "humidity": 60,
                        "chance_of_rain": 0,
                        "chance_of_snow": 0,
                        "wind_kph": 0,
                        "uv": 0,
                        "cloud": 0,
                        "vis_km": 0
                    }
            
            weather_data.append(weather_point)
            
            # Hourly Sampling
            current_time += timedelta(hours=1)

        # Formatted as a DataFrame
        df = pd.DataFrame(weather_data)

        # Step 5：Variables for Summary Prompt
        if duration_hours == 1:
            # Single point enquiry
            time_description = f"at a specific time: {start_dt.strftime('%Y-%m-%d %H:%M')} in {location}"
            
            if len(weather_data) > 0:
                wd = weather_data[0]
                weather_data_text = f"""Location: {location}
Time: {start_dt.strftime('%Y-%m-%d')} at {start_dt.strftime('%H:%M')}
Condition: {wd['condition']}
Temperature: {wd['temp_c']}°C (Feels like {wd['feelslike_c']}°C)
Humidity: {wd['humidity']}%
Chance of rain: {wd['chance_of_rain']}%
Chance of snow: {wd['chance_of_snow']}%
Wind speed: {wd['wind_kph']} kph
UV index: {wd['uv']}
Cloud cover: {wd['cloud']}%
Visibility: {wd['vis_km']} km"""
            else:
                weather_data_text = "No weather data available."
        else:
            # Range Enquiry - Using Tabular Format
            time_description = f"from {start_dt.strftime('%Y-%m-%d %H:%M')} to {end_dt.strftime('%Y-%m-%d %H:%M')} in {location}"
            weather_data_text = f"Location: {location}\n\nWeather Data Table:\n{df.to_string(index=False)}"

        summary_prompt = f"""
You are a helpful weather reasoning assistant with intelligent output selection.

The user wants to know about the weather conditions {time_description}.
Use the data below to answer their question. This may refer to the past, present, or future — do not assume it is the current weather.

Based on the following weather data and the user's question, think step-by-step to extract the most relevant information, and give a natural, friendly, and cautious answer in British English.

Avoid being overly confident — never say "Yes, it will..." or "Definitely." Instead, use expressions like:
- "It is very likely that..."
- "There is a high chance of..."
- "Based on the available data, it seems that..."
- "There may be..."

Also, after answering the question, include a short weather summary and a useful suggestion.

**Do not use markdown formatting such as `*`, `**`, or list symbols.**

--- Weather Data ---
{weather_data_text}

--- User Question ---
{query}

--- Final Answer ---
First, provide your weather analysis and recommendations.

Then, intelligently decide if the user would benefit from visual aids:

**Add "chart: true" if:**
- The query involves trends, changes over time, or comparisons
- Multiple time periods are mentioned (e.g., "next 3 days", "this week")
- The user asks about patterns, variations, or forecasts
- Weather data spans several hours/days
- Questions like "how will it change", "show me", "what's the trend"

**Add "chart: true" if:**
- Weather data spans MORE than 1 hour (time series visualization helpful)
- ANY time range query (next 2 hours, today, tomorrow, this week, etc.)
- Multiple time points are involved (even implicit ranges)
- Trend analysis would be useful for the user
- DEFAULT: If duration > 1 hour → ALWAYS add chart: true

**Add "table: true" if:**
- User wants comprehensive details, precise values, or reference data
- Multiple weather parameters need exact numbers
- Detailed breakdown is specifically requested

**Single point queries (1 hour or specific moment):**
- "Will it rain at 3pm tomorrow?" → neither (just text answer)
- "Temperature right now in London" → neither (single value)

**Time range queries (>1 hour):**
- "Weather today" → chart: true (shows daily trend)
- "Next 2 hours" → chart: true (shows progression) 
- "This weekend" → chart: true (trend visualization)
- "Tomorrow" → chart: true (daily pattern)

**Remember: Humans prefer visual information. When in doubt about time range, lean towards providing charts.**

Think about what would be most helpful for the user, even if they didn't explicitly ask.
"""
        response = llm_gpt4.invoke(summary_prompt)
        response_text = response.content.strip() if isinstance(response, AIMessage) else str(response)

        # Capture output mode (chart / table)
        def extract_output_mode(text: str) -> list[str]:
            modes = []
            lower_text = text.lower()
            if "chart: true" in lower_text:
                modes.append("chart")
            if "table: true" in lower_text:
                modes.append("table")
            return modes

        output_modes = extract_output_mode(response_text)

        # Remove the chart/table prompt string and keep only the narrative.
        clean_response = response_text.split("chart:")[0].split("table:")[0].strip()
        final_response = clean_response
        

        # Inserting a Chart or Table (as indicated by LLM)
        if "chart" in output_modes:
            try:
                chart_path = render_chart(df, location, f"Weather Data for {location}")
                final_response += f"\n\nHere is your chart:\n{chart_path}"
            except Exception as e:
                final_response += f"\n\nChart generation failed: {e}"

        if "table" in output_modes:
            try:
                table_text = render_table(df)
                final_response += f"\n\nHere is your table:\n{table_text}"
            except Exception as e:
                final_response += f"\n\nTable generation failed: {e}"

        return final_response

    except Exception as e:
        return f"Weather Agent Error: {e}"
    
if __name__ == "__main__":
    test_queries = [
        "Will it rain in Tokyo tomorrow at 3pm?",

        "Show me the weather for the next 2 hours in London.",

        "What's the temperature in New York from 3pm today to 2am tomorrow?",

        "Generate a 3-day weather chart for Paris starting next Wednesday.",

        "Give me a table of average humidity for the next 5 days in Sydney.",

        "What was the average temperature in Kaohsiung 5 days ago?",

        "Provide a  6-day weather summary for Tokyo ending today.",

        "Will it rain in Taipei from 4pm to 8pm today?",

        "What is the average temperature in London for the next 8 days?",

        "How hot was it in New York yesterday at 5pm?"
    ]

    for q in test_queries:
        print("─" * 60)
        print("Query:", q)
        print("Response:")
        print(weather_agent_tool(q))
        print("\n")   
