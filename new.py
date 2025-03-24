from flask import Flask, render_template, jsonify
import requests
import pandas as pd
import joblib
from datetime import datetime, timedelta
import numpy as np
import psycopg2
from psycopg2.extras import execute_values


app = Flask(__name__)

# Load the trained model and scaler 
model = joblib.load(r"C:\Users\Sarah\Desktop\trend_analysis\model\random_forest_model.pkl")
scaler = joblib.load(r"C:\Users\Sarah\Desktop\trend_analysis\model\scaler.pkl")

API_URL = "https://neptune.kyogojo.com/api/statistics/get-multiple?stations=BSLG-002"

def fetch_data():
    """Fetch real-time water pressure data from API."""
    response = requests.get(API_URL)
    if response.status_code == 200:
        data = response.json().get("payload", {}).get("data", [])
        if data:
            return data[0]["pressure"]
    return []  # Return empty list if no data

def preprocess_data(data):
    """Convert API data into structured DataFrame with extracted time features."""
    if not data:
        return pd.DataFrame() 

    df = pd.DataFrame(data)
    df["time"] = df["time"] // 1000  # convert milliseconds to seconds
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df["time"] = df["time"] + timedelta(hours=8)
    df["value"] = df["value"].interpolate(method='linear', limit_direction='both')

    # Extract time-based features
    df["year"] = df["time"].dt.year
    df["month"] = df["time"].dt.month
    df["day"] = df["time"].dt.day
    df["hour"] = df["time"].dt.hour
    df["minute"] = df["time"].dt.minute
    df["second"] = df["time"].dt.second
    df["millisecond"] = df["time"].dt.microsecond // 1000  # Convert microseconds to milliseconds

    return df


@app.route("/")
def index():
    """Render the frontend."""
    return render_template("index.html")

def save_to_db(data):
    """Store the actual and predicted pressure data in PostgreSQL with converted time (UTC -> PST)."""
    try:
        conn = psycopg2.connect(
            host="localhost",
            database="water",
            user="postgres",
            password="123",
            port=5432
        )
        cur = conn.cursor()

        insert_query = """
        INSERT INTO water_pressure (time, actual_pressure, predicted_pressure, upper_bound, lower_bound)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (time) DO UPDATE
        SET actual_pressure = EXCLUDED.actual_pressure,
            predicted_pressure = EXCLUDED.predicted_pressure,
            upper_bound = EXCLUDED.upper_bound,
            lower_bound = EXCLUDED.lower_bound;
        """

        for row in data:
            # Skip actual pressure is null
            if row["Actual"] is None:
                continue

            utc_time = pd.to_datetime(row["time"])
            # UTC time to PST - subtract 8 hours
            pst_time = utc_time - timedelta(hours=8)

            cur.execute(insert_query, (
                pst_time, 
                row["Actual"], 
                row["Predicted"], 
                row["Upper_Bound"], 
                row["Lower_Bound"]
            ))

        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Database insertion error: {e}")

@app.route('/data')
def get_data():
    # Fetch real-time data
    api_data = fetch_data()
    if not api_data:
        return jsonify({"error": "No data available"}), 500

    df = preprocess_data(api_data)
    if df.empty:
        return jsonify({"error": "Failed to process data"}), 500

    # If the 'value' column does not exist, create it and set as None
    if "value" not in df.columns:
        df["value"] = None  # if no real values are available, assign None 

    df = df.rename(columns={"value": "Actual"}) 

    # correct feature order for the model
    feature_columns = ["year", "month", "day", "hour", "minute", "second", "millisecond"]
    X = df[feature_columns]

    # Scale features
    try:
        X_scaled = scaler.transform(X)
    except Exception as e:
        return jsonify({"error": f"Scaler transform failed: {str(e)}"}), 500

    # Predict pressure values
    try:
        df["Predicted"] = model.predict(X_scaled)
    except Exception as e:
        return jsonify({"error": f"Model prediction failed: {str(e)}"}), 500

    # Ensure no NaN values in predictions
    df["Predicted"] = df["Predicted"].fillna(0)

    # sort data by time (latest first)
    df = df.sort_values(by="time", ascending=False).reset_index(drop=True)

    # get latest and previous (5 min ago) pressure
    latest_pressure = df.iloc[0]["Actual"] if not df.empty else None
    previous_pressure = None

    if not df.empty:
        latest_time = df.iloc[0]["time"]
        prev_row = df[df["time"] <= latest_time - timedelta(minutes=5)]
        if not prev_row.empty:
            previous_pressure = prev_row.iloc[0]["Actual"]

    # generate future timestamps (every 5 minutes for the next 30 minutes)
    last_time = df["time"].max()
    future_times = [last_time + timedelta(minutes=5 * i) for i in range(1, 7)]

    # create future feature dataframe
    future_df = pd.DataFrame({
        "year": [t.year for t in future_times],
        "month": [t.month for t in future_times],
        "day": [t.day for t in future_times],
        "hour": [t.hour for t in future_times],
        "minute": [t.minute for t in future_times],
        "second": [0] * len(future_times),
        "millisecond": [0] * len(future_times)
    })

    # Scale future features
    future_scaled = scaler.transform(future_df)
    future_df["Predicted"] = model.predict(future_scaled)
    future_df["time"] = future_times

    # set "Actual" values for future data to None
    future_df["Actual"] = None  

    # ensure df is not empty before concatenation 
    if not df.empty:
        combined_df = pd.concat([
            df[["time", "Actual", "Predicted"]], 
            future_df[["time", "Actual", "Predicted"]]
        ], ignore_index=True)
    else:
        combined_df = future_df[["time", "Actual", "Predicted"]]

    # compute Upper & Lower Bound Lines (Margin ±5)
    combined_df["Upper_Bound"] = combined_df["Predicted"] + 5
    combined_df["Lower_Bound"] = combined_df["Predicted"] - 5

    # convert DataFrame to JSON and explicitly replace NaN values
    combined_json = combined_df.replace({np.nan: None}).to_dict(orient="records")
    save_to_db(combined_json)


    # return latest & previous pressure along with data
    return jsonify({
        "latest_pressure": latest_pressure,
        "previous_pressure": previous_pressure,
        "data": combined_json
    })

def get_headloss():
    """Calculate headloss based on the water pressure using pressure today and yesterday."""
    pass

if __name__ == "__main__":
    app.run(debug=True)

