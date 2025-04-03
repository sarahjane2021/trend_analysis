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
model = joblib.load(r"C:\Users\Sarah\Desktop\trend_analysis\model\random_forest_model2.pkl")
scaler = joblib.load(r"C:\Users\Sarah\Desktop\trend_analysis\model\scaler.pkl")

API_URL = "https://neptune.kyogojo.com/api/statistics/get-multiple?stations=BSLG-002&days"
API_URL2 = "https://neptune.kyogojo.com/api/statistics/get-multiple?stations=BSLG-003&days"

def fetch_data(api_url):
    """Fetch real-time water pressure data from the given API and return a structured DataFrame."""
    response = requests.get(api_url)
    if response.status_code == 200:
        data = response.json().get("payload", {}).get("data", [])
        if data and "pressure" in data[0]:
            pressure_data = data[0]["pressure"]

            # Convert API response into a DataFrame
            df = pd.DataFrame([
                {"time": (datetime.utcfromtimestamp(e["time"] / 1000) + timedelta(hours=8)).strftime("%Y-%m-%d %H:%M:%S"),
                 "value": e["value"]}
                for e in pressure_data
            ])
            return df  # Return structured DataFrame
    
    return pd.DataFrame()  # Return empty DataFrame if no data

####################
@app.route("/")
def index():
    """Render the frontend."""
    return render_template("index.html")

####################

def get_headloss():
    """Fetch head loss values for 24 hours ago and store the calculation timestamp."""
    conn = None
    cur = None
    try:
        conn = psycopg2.connect(host="localhost", database="water", user="postgres", password="123", port=5432)
        cur = conn.cursor()

        query = """
        SELECT 
            ROUND(CAST((y.actual_pressure - t.actual_pressure) / 9810.0 AS NUMERIC), 6) AS head_loss_24h
        FROM 
            (SELECT actual_pressure FROM water_pressure 
            WHERE time <= NOW() - INTERVAL '1 day' 
            ORDER BY time DESC LIMIT 1) y,
            (SELECT actual_pressure FROM water_pressure 
            ORDER BY time DESC LIMIT 1) t;
        """
        
        cur.execute(query)
        result = cur.fetchone()
        
        # Get current timestamp
        head_loss_time = datetime.now().strftime("%B %d, %Y %I:%M %p")  # Example: April 1, 2025 9:45 AM
        
        return {
            "head_loss_24h": result[0] if result else None,
            "calculation_time": head_loss_time
        }

    except Exception as e:
        print(f"Database error: {e}")
        return {"head_loss_24h": None, "calculation_time": None}

    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()

####################
def save_to_db(data, head_loss_24h, anomaly_value):
    """Store actual, predicted pressure, and head loss values in PostgreSQL."""
    try:
        conn = psycopg2.connect(
            host="localhost",
            database="water",
            user="postgres",
            password="123",
            port=5432
        )
        cur = conn.cursor()

        # Ensure table has the new column for anomaly_value
        cur.execute("""
        CREATE TABLE IF NOT EXISTS water_pressure (
            time TIMESTAMP PRIMARY KEY,
            actual_pressure NUMERIC,
            predicted_pressure NUMERIC,
            upper_bound NUMERIC,
            lower_bound NUMERIC,
            head_loss_24h NUMERIC,
            anomaly_value NUMERIC  -- New column for anomaly
        );
        """)

        insert_query = """
        INSERT INTO water_pressure (time, actual_pressure, predicted_pressure, upper_bound, lower_bound, head_loss_24h, anomaly_value)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (time) DO NOTHING;
        """

        cur.execute("SELECT time FROM water_pressure")
        existing_timestamps = {row[0] for row in cur.fetchall()}  # Store existing timestamps to prevent duplicates

        for row in data:
            if row.get("Actual") is None:
                continue  # Skip if no actual pressure data

            utc_time = pd.to_datetime(row["time"])

            if utc_time in existing_timestamps:
                print(f"⏩ Skipping duplicate timestamp: {utc_time}")
                continue  # Skip duplicate timestamps

            print(f"✅ Inserting: {utc_time}, {row['Actual']}, {row['Predicted']}, {head_loss_24h}, {anomaly_value}")

            cur.execute(insert_query, (
                utc_time,
                row["Actual"],
                row["Predicted"],
                row["Upper_Bound"],
                row["Lower_Bound"],
                head_loss_24h, 
                row["anomaly_value"] if row["is_anomaly"] else None
            ))

        conn.commit()
        cur.close()
        conn.close()

        print("🚀 Data successfully saved!")

    except Exception as e:
        print(f"❌ Database error: {e}")

####################
@app.route('/data')
def get_data():
    # Fetch data from both APIs
    df_bslg_002 = fetch_data(API_URL)  # For BSLG-002
    df_bslg_003 = fetch_data(API_URL2)  # For BSLG-003

    if df_bslg_002.empty and df_bslg_003.empty:
        return jsonify({"error": "No data available from both sources"}), 500

    # Process data for BSLG-002
    df_bslg_002["time"] = pd.to_datetime(df_bslg_002["time"])
    df_bslg_002["year"] = df_bslg_002["time"].dt.year
    df_bslg_002["month"] = df_bslg_002["time"].dt.month
    df_bslg_002["day"] = df_bslg_002["time"].dt.day
    df_bslg_002["hour"] = df_bslg_002["time"].dt.hour
    df_bslg_002["minute"] = df_bslg_002["time"].dt.minute
    df_bslg_002["second"] = df_bslg_002["time"].dt.second
    df_bslg_002["millisecond"] = df_bslg_002["time"].dt.microsecond // 1000
    df_bslg_002 = df_bslg_002.rename(columns={"value": "Actual_BSLG_002"})

    # Process data for BSLG-003
    df_bslg_003["time"] = pd.to_datetime(df_bslg_003["time"])
    df_bslg_003["year"] = df_bslg_003["time"].dt.year
    df_bslg_003["month"] = df_bslg_003["time"].dt.month
    df_bslg_003["day"] = df_bslg_003["time"].dt.day
    df_bslg_003["hour"] = df_bslg_003["time"].dt.hour
    df_bslg_003["minute"] = df_bslg_003["time"].dt.minute
    df_bslg_003["second"] = df_bslg_003["time"].dt.second
    df_bslg_003["millisecond"] = df_bslg_003["time"].dt.microsecond // 1000
    df_bslg_003 = df_bslg_003.rename(columns={"value": "Actual_BSLG_003"})

    # Feature extraction and prediction (for both sources)
    feature_columns = ["year", "month", "day", "hour", "minute", "second", "millisecond"]
    X_bslg_002 = df_bslg_002[feature_columns]
    X_bslg_003 = df_bslg_003[feature_columns]

    try:
        # Predict for both BSLG-002 and BSLG-003
        X_bslg_002_scaled = scaler.transform(X_bslg_002)
        df_bslg_002["Predicted_BSLG_002"] = model.predict(X_bslg_002_scaled)

        X_bslg_003_scaled = scaler.transform(X_bslg_003)
        df_bslg_003["Predicted_BSLG_003"] = model.predict(X_bslg_003_scaled)
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    # Handle anomalies (for both sources)
    df_bslg_002["Upper_Bound_BSLG_002"] = df_bslg_002["Predicted_BSLG_002"] + 5
    df_bslg_002["Lower_Bound_BSLG_002"] = df_bslg_002["Predicted_BSLG_002"] - 5
    df_bslg_003["Upper_Bound_BSLG_003"] = df_bslg_003["Predicted_BSLG_003"] + 5
    df_bslg_003["Lower_Bound_BSLG_003"] = df_bslg_003["Predicted_BSLG_003"] - 5

    df_bslg_002["is_anomaly_BSLG_002"] = df_bslg_002.apply(
        lambda row: (row["Actual_BSLG_002"] is not None) and 
                    ((row["Actual_BSLG_002"] > row["Upper_Bound_BSLG_002"]) or 
                     (row["Actual_BSLG_002"] < row["Lower_Bound_BSLG_002"])), axis=1
    )

    df_bslg_003["is_anomaly_BSLG_003"] = df_bslg_003.apply(
        lambda row: (row["Actual_BSLG_003"] is not None) and 
                    ((row["Actual_BSLG_003"] > row["Upper_Bound_BSLG_003"]) or 
                     (row["Actual_BSLG_003"] < row["Lower_Bound_BSLG_003"])), axis=1
    )

    # Extract latest anomaly (if any) for both sources
    latest_anomaly_bslg_002 = df_bslg_002[df_bslg_002["is_anomaly_BSLG_002"]].sort_values(by="time", ascending=False).head(1)
    latest_anomaly_bslg_003 = df_bslg_003[df_bslg_003["is_anomaly_BSLG_003"]].sort_values(by="time", ascending=False).head(1)

    anomaly_info_bslg_002 = {
        "value": latest_anomaly_bslg_002["Actual_BSLG_002"].values[0] if not latest_anomaly_bslg_002.empty else None,
        "timestamp": latest_anomaly_bslg_002["time"].dt.strftime("%B %d, %Y %I:%M %p").values[0] if not latest_anomaly_bslg_002.empty else None
    }

    anomaly_info_bslg_003 = {
        "value": latest_anomaly_bslg_003["Actual_BSLG_003"].values[0] if not latest_anomaly_bslg_003.empty else None,
        "timestamp": latest_anomaly_bslg_003["time"].dt.strftime("%B %d, %Y %I:%M %p").values[0] if not latest_anomaly_bslg_003.empty else None
    }

    # Extract latest pressure and previous pressure for both sources
    latest_pressure_bslg_002 = df_bslg_002.iloc[0]["Actual_BSLG_002"] if not df_bslg_002.empty else None
    previous_pressure_bslg_002 = None
    if not df_bslg_002.empty:
        latest_time_bslg_002 = df_bslg_002.iloc[0]["time"]
        prev_row_bslg_002 = df_bslg_002[df_bslg_002["time"] <= latest_time_bslg_002 - timedelta(minutes=5)]
        if not prev_row_bslg_002.empty:
            previous_pressure_bslg_002 = prev_row_bslg_002.iloc[0]["Actual_BSLG_002"]

    latest_pressure_bslg_003 = df_bslg_003.iloc[0]["Actual_BSLG_003"] if not df_bslg_003.empty else None
    previous_pressure_bslg_003 = None
    if not df_bslg_003.empty:
        latest_time_bslg_003 = df_bslg_003.iloc[0]["time"]
        prev_row_bslg_003 = df_bslg_003[df_bslg_003["time"] <= latest_time_bslg_003 - timedelta(minutes=5)]
        if not prev_row_bslg_003.empty:
            previous_pressure_bslg_003 = prev_row_bslg_003.iloc[0]["Actual_BSLG_003"]

    # Calculate head_loss_24h and calculation_time for both sources (you can keep your existing head_loss calculation)
    head_loss_values_bslg_002 = get_headloss()  # Assuming this is a common function
    head_loss_values_bslg_003 = get_headloss()

    # Return both sources' data in the same response
    return jsonify({
        "data_bslg_002": {
            "latest_pressure": latest_pressure_bslg_002,
            "previous_pressure": previous_pressure_bslg_002,
            "head_loss_24h": head_loss_values_bslg_002["head_loss_24h"],
            "calculation_time": head_loss_values_bslg_002["calculation_time"],
            "anomaly_value": anomaly_info_bslg_002["value"],
            "anomaly_time": anomaly_info_bslg_002["timestamp"],
            "data": df_bslg_002.replace({np.nan: None}).to_dict(orient="records")
        },
        "data_bslg_003": {
            "latest_pressure": latest_pressure_bslg_003,
            "previous_pressure": previous_pressure_bslg_003,
            "head_loss_24h": head_loss_values_bslg_003["head_loss_24h"],
            "calculation_time": head_loss_values_bslg_003["calculation_time"],
            "anomaly_value": anomaly_info_bslg_003["value"],
            "anomaly_time": anomaly_info_bslg_003["timestamp"],
            "data": df_bslg_003.replace({np.nan: None}).to_dict(orient="records")
        }
    })

if __name__ == "__main__":
    app.run(debug=True)
