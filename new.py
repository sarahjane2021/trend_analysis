from flask import Flask, render_template, jsonify
import requests
import pandas as pd
import joblib
from datetime import datetime, timedelta
import numpy as np
import psycopg2
from psycopg2.extras import execute_values

app = Flask(__name__)

# Load the trained models and scalers for both APIs
model_bslg_002 = joblib.load(r"C:\Users\Sarah\Desktop\trend_analysis\model\random_forest_model.pkl")
scaler_bslg_002 = joblib.load(r"C:\Users\Sarah\Desktop\trend_analysis\model\scaler.pkl")

model_bslg_003 = joblib.load(r"C:\Users\Sarah\Desktop\trend_analysis\model\random_forest_model2.pkl")
scaler_bslg_003 = joblib.load(r"C:\Users\Sarah\Desktop\trend_analysis\model\scaler.pkl")

API_URL = "https://neptune.kyogojo.com/api/statistics/get-multiple?stations=BSLG-002&days"
API_URL2 = "https://neptune.kyogojo.com/api/statistics/get-multiple?stations=BSLG-003&days"

def get_model_and_scaler(station):
    """Return the appropriate model and scaler based on the station ID."""
    if station == "BSLG-002":
        return model_bslg_002, scaler_bslg_002
    elif station == "BSLG-003":
        return model_bslg_003, scaler_bslg_003
    else:
        return None, None
######################
def fetch_data(api_url):
    """Fetch real-time water pressure data from the given API and return a structured DataFrame."""
    response = requests.get(api_url)
    if response.status_code == 200:
        data = response.json().get("payload", {}).get("data", [])
        if data and "pressure" in data[0]:
            pressure_data = data[0]["pressure"]

            # API response into a DataFrame
            df = pd.DataFrame([
                {"time": (datetime.utcfromtimestamp(e["time"] / 1000) + timedelta(hours=8)).strftime("%Y-%m-%d %H:%M:%S"),
                 "value": e["value"]}
                for e in pressure_data
            ])
            return df  
    
    return pd.DataFrame()
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
    df_bslg_002 = fetch_data(API_URL)
    df_bslg_003 = fetch_data(API_URL2)
    
    if df_bslg_002.empty and df_bslg_003.empty:
        return jsonify({"error": "No data available from both sources"}), 500

    def process_data(df, station):
        model, scaler = get_model_and_scaler(station)
        if model is None or scaler is None:
            return None, f"Model or scaler not found for station: {station}"

        df["time"] = pd.to_datetime(df["time"])
        df["year"] = df["time"].dt.year
        df["month"] = df["time"].dt.month
        df["day"] = df["time"].dt.day
        df["hour"] = df["time"].dt.hour
        df["minute"] = df["time"].dt.minute
        df["second"] = df["time"].dt.second
        df["millisecond"] = df["time"].dt.microsecond // 1000

        # Renaming the column dynamically based on the station
        df = df.rename(columns={"value": f"Actual_{station}"})

        # Ensure the renaming worked before proceeding
        if f"Actual_{station}" not in df.columns:
            return None, f"Column 'Actual_{station}' not found after renaming in station: {station}"

        # Feature extraction
        feature_columns = ["year", "month", "day", "hour", "minute", "second", "millisecond"]
        X_scaled = scaler.transform(df[feature_columns])
        df[f"Predicted_{station}"] = model.predict(X_scaled)

        #============================================
        df[f"Upper_Bound_{station}"] = df[f"Predicted_{station}"] + 5
        df[f"Lower_Bound_{station}"] = df[f"Predicted_{station}"] - 5

        # Detect anomalies
        df[f"is_anomaly_{station}"] = df.apply(
            lambda row: (row[f"Actual_{station}"] is not None) and 
                        ((row[f"Actual_{station}"] > row[f"Upper_Bound_{station}"]) or 
                         (row[f"Actual_{station}"] < row[f"Lower_Bound_{station}"])), axis=1
        )
        latest_anomaly = df[df[f"is_anomaly_{station}"]].sort_values(by="time", ascending=False).head(1)    # Extract latest anomaly (if any) for the station

        # anomaly details
        anomaly_info = {
            "value": latest_anomaly[f"Actual_{station}"].values[0] if not latest_anomaly.empty else None,
            "timestamp": latest_anomaly["time"].dt.strftime("%B %d, %Y %I:%M %p").values[0] if not latest_anomaly.empty else None
        }

        df = df.sort_values(by="time", ascending=False)
        latest_pressure = df.iloc[0][f"Actual_{station}"] if not df.empty else None

        current_time = df["time"].max()
        five_minutes_ago = current_time - timedelta(minutes=5)
        previous_pressure = df[df["time"] <= five_minutes_ago].iloc[0][f"Actual_{station}"] if not df[df["time"] <= five_minutes_ago].empty else None

        return anomaly_info, df, latest_pressure, previous_pressure  # return the anomaly info, full data, and pressure values

    anomaly_bslg_002, df_bslg_002_processed, latest_pressure_bslg_002, previous_pressure_bslg_002 = process_data(df_bslg_002, "BSLG-002")
    anomaly_bslg_003, df_bslg_003_processed, latest_pressure_bslg_003, previous_pressure_bslg_003 = process_data(df_bslg_003, "BSLG-003")
    
    # return errors from process_data
    if anomaly_bslg_002 is None:
        return jsonify({"error": "Error processing BSLG-002 data"}), 500
    if anomaly_bslg_003 is None:
        return jsonify({"error": "Error processing BSLG-003 data"}), 500
        #============================================
        
    return jsonify({
        "anomaly_bslg_002": anomaly_bslg_002,
        "anomaly_bslg_003": anomaly_bslg_003,
        "latest_pressure_bslg_002": latest_pressure_bslg_002,
        "previous_pressure_bslg_002": previous_pressure_bslg_002,
        "latest_pressure_bslg_003": latest_pressure_bslg_003,
        "previous_pressure_bslg_003": previous_pressure_bslg_003,
        "data_bslg_002": df_bslg_002_processed.replace({np.nan: None}).to_dict(orient="records"),
        "data_bslg_003": df_bslg_003_processed.replace({np.nan: None}).to_dict(orient="records")
    })

if __name__ == "__main__":
    app.run(debug=True)
