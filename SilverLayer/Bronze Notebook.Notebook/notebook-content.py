# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "synapse_pyspark"
# META   },
# META   "dependencies": {
# META     "lakehouse": {
# META       "default_lakehouse": "41d2946a-a6f1-4fb7-936b-40732a1a2b0e",
# META       "default_lakehouse_name": "USWeatherForecast",
# META       "default_lakehouse_workspace_id": "3c9c33cd-8103-42e6-900e-4a2a8da0aafe",
# META       "known_lakehouses": [
# META         {
# META           "id": "41d2946a-a6f1-4fb7-936b-40732a1a2b0e"
# META         }
# META       ]
# META     }
# META   }
# META }

# CELL ********************

import requests
from pyspark.sql import Row
from pyspark.sql.functions import *
from pyspark.sql.types import *
# OR explicitly import with alias
from pyspark.sql.functions import (
    col, to_timestamp, to_date, dayofweek,
    current_timestamp, round as spark_round, when
)

HEADERS = {
    "User-Agent": "MyWeatherApp/1.0 (contact@example.com)",  # Required by weather.gov
    "Accept": "application/geo+json"
}

def get_grid_info(lat: float, lon: float) -> dict:
    """Step 1: Resolve lat/lon to NWS grid info"""
    url = f"https://api.weather.gov/points/{lat},{lon}"
    resp = requests.get(url, headers=HEADERS)
    props = resp.json()["properties"]
    return {
        "wfo":    props["gridId"],
        "grid_x": props["gridX"],
        "grid_y": props["gridY"],
        "city":   props["relativeLocation"]["properties"]["city"],
        "state":  props["relativeLocation"]["properties"]["state"]
    }

def get_forecast(wfo: str, x: int, y: int) -> list:
    """Step 2: Fetch 7-day forecast periods"""
    url = f"https://api.weather.gov/gridpoints/{wfo}/{x},{y}/forecast"
    resp = requests.get(url, headers=HEADERS)
    return resp.json()["properties"]["periods"]

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

#fetch from multiple locations

locations = [
    {"name": "New York",     "lat": 40.7128, "lon": -74.0060},
    {"name": "Chicago",      "lat": 41.8781, "lon": -87.6298},
    {"name": "Los Angeles",  "lat": 34.0522, "lon": -118.2437},
    {"name": "Miami",        "lat": 25.7617, "lon": -80.1918},
    {"name": "Seattle",      "lat": 47.6062, "lon": -122.3321},
]

rows = []

for loc in locations:
    grid = get_grid_info(loc["lat"], loc["lon"])
    periods = get_forecast(grid["wfo"], grid["grid_x"], grid["grid_y"])
    
    for period in periods:
        rows.append(Row(
            location_name   = loc["name"],
            city            = grid["city"],
            state           = grid["state"],
            period_name     = period["name"],               # "Monday", "Monday Night"
            is_daytime      = bool(period["isDaytime"]),
            temp_f          = int(period["temperature"]),
            temp_unit       = period["temperatureUnit"],
            wind_speed      = period["windSpeed"],          # e.g. "10 to 15 mph"
            wind_direction  = period["windDirection"],
            short_forecast  = period["shortForecast"],
            detailed_forecast = period["detailedForecast"],
            start_time      = period["startTime"],
            end_time        = period["endTime"],
        ))

df_raw = spark.createDataFrame(rows)
display(df_raw)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

#Define Schema Explicitly
forecast_schema = StructType([
    StructField("location_name",    StringType(),  True),
    StructField("city",             StringType(),  True),
    StructField("state",            StringType(),  True),
    StructField("period_name",      StringType(),  True),
    StructField("is_daytime",       BooleanType(), True),
    StructField("temp_f",           IntegerType(), True),
    StructField("temp_unit",        StringType(),  True),
    StructField("wind_speed",       StringType(),  True),
    StructField("wind_direction",   StringType(),  True),
    StructField("short_forecast",   StringType(),  True),
    StructField("detailed_forecast",StringType(),  True),
    StructField("start_time",       StringType(),  True),
    StructField("end_time",         StringType(),  True),
])

df_raw = spark.createDataFrame(rows, schema=forecast_schema)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

#Transform with Spark
df_clean = df_raw \
    .withColumn("start_time", to_timestamp(col("start_time"))) \
    .withColumn("end_time",   to_timestamp(col("end_time"))) \
    .withColumn("temp_c",     spark_round((col("temp_f") - 32) * 5/5, 1)) \
    .withColumn("temp_c",     spark_round((col("temp_f") - 32) * 5 / 9, 1)) \
    .withColumn("day_of_week",dayofweek(col("start_time"))) \
    .withColumn("forecast_date", to_date(col("start_time"))) \
    .withColumn("ingested_at", current_timestamp()) \
    .withColumn("heat_index",
        when(col("temp_f") >= 90, "Hot")
        .when(col("temp_f") >= 70, "Warm")
        .when(col("temp_f") >= 50, "Cool")
        .otherwise("Cold")
    )

display(df_clean)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Parallelize with RDD for Scale
locations_broadcast = spark.sparkContext.broadcast(locations)
headers_broadcast   = spark.sparkContext.broadcast(HEADERS)

def fetch_location_forecast(loc):
    import requests
    headers = headers_broadcast.value
    
    # Step 1: grid info
    pts = requests.get(
        f"https://api.weather.gov/points/{loc['lat']},{loc['lon']}",
        headers=headers
    ).json()["properties"]
    
    wfo, x, y = pts["gridId"], pts["gridX"], pts["gridY"]
    city  = pts["relativeLocation"]["properties"]["city"]
    state = pts["relativeLocation"]["properties"]["state"]
    
    # Step 2: forecast
    periods = requests.get(
        f"https://api.weather.gov/gridpoints/{wfo}/{x},{y}/forecast",
        headers=headers
    ).json()["properties"]["periods"]
    
    return [
        (loc["name"], city, state,
         p["name"], bool(p["isDaytime"]),
         int(p["temperature"]), p["windSpeed"],
         p["windDirection"], p["shortForecast"],
         p["startTime"], p["endTime"])
        for p in periods
    ]

locations_rdd = spark.sparkContext.parallelize(locations, numSlices=len(locations))
results_rdd   = locations_rdd.flatMap(fetch_location_forecast)  # flatMap since each loc → many rows

df_parallel = spark.createDataFrame(results_rdd, schema=[
    "location_name", "city", "state", "period_name",
    "is_daytime", "temp_f", "wind_speed",
    "wind_direction", "short_forecast",
    "start_time", "end_time"
])

display(df_parallel)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Write forecast data
df_clean.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .partitionBy("location_name", "forecast_date") \
    .saveAsTable("weather_nws_forecast")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
