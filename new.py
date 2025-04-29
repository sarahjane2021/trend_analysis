from flask import Flask, render_template, jsonify
import requests
import pandas as pd
import joblib
from datetime import datetime, timedelta
import numpy as np
import psycopg2

app = Flask(__name__)

# Load the trained models and scalers for both APIs
model_bslg_002 = joblib.load(r"C:\Users\Sarah\Desktop\trend_analysis\model\random_forest_model.pkl")
scaler_bslg_002 = joblib.load(r"C:\Users\Sarah\Desktop\trend_analysis\model\scaler_002.pkl")

model_bslg_003 = joblib.load(r"C:\Users\Sarah\Desktop\trend_analysis\model\random_forest_bslg_003.pkl")
scaler_bslg_003 = joblib.load(r"C:\Users\Sarah\Desktop\trend_analysis\model\scaler_003.pkl")

API_URL = "https://neptune.kyogojo.com/api/statistics/get-multiple?stations=BSLG-002&days="
API_URL2 = "https://neptune.kyogojo.com/api/statistics/get-multiple?stations=BSLG-003&days="
 
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
@app.route('/get_headloss')
def get_headloss():
    try:
        conn = psycopg2.connect(
            host="209.38.56.184",
            database="postgres",
            user="admin",
            password="password",
            port=5432
        )
        cursor = conn.cursor()
        
        # Get latest head loss data for each station
        cursor.execute("""
            SELECT station_id, latest_time, head_loss
            FROM head_loss_view
            ORDER BY latest_time DESC
        """)
        
        rows = cursor.fetchall()
        conn.close()
        
        # Structure the result
        result = {}
        for station_id, latest_time, head_loss in rows:
            result[station_id] = {
                'head_loss': head_loss,
                'latest_time': latest_time.strftime("%Y-%m-%d %H:%M:%S") if latest_time else None
            }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)})

####################
def save_to_db(data, station, conn):
    """Create table if not exists and save valid pressure data to the PostgreSQL database."""
    cur = conn.cursor()

    # Create table if not exists
    cur.execute("""
        CREATE TABLE IF NOT EXISTS water_pressure (
            id SERIAL PRIMARY KEY,
            station_id TEXT,
            time TIMESTAMP UNIQUE,
            actual_pressure DOUBLE PRECISION,
            predicted_pressure DOUBLE PRECISION,
            upper_bound DOUBLE PRECISION,
            lower_bound DOUBLE PRECISION,
            is_anomaly BOOLEAN
        );
    """)

    inserted_count = 0

    # Insert records safely
    for record in data:
        actual = record.get(f"Actual_{station}")
        if actual is not None:
            try:
                cur.execute("""
                    INSERT INTO water_pressure (
                        station_id, time, actual_pressure, predicted_pressure,
                        upper_bound, lower_bound, is_anomaly
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (time) DO NOTHING;
                """, (
                    station,
                    datetime.strptime(record["time"], "%a, %d %b %Y %H:%M:%S GMT"),
                    actual,
                    record.get(f"Predicted_{station}"),
                    record.get(f"Upper_Bound_{station}"),
                    record.get(f"Lower_Bound_{station}"),
                    record.get(f"is_anomaly_{station}")
                ))

                if cur.rowcount == 1:  # row was inserted
                    inserted_count += 1

            except Exception as e:
                print(f"[ERROR] Failed insert for {record['time']} - {station}: {e}")

    conn.commit()
    cur.close()

    # ✅ Final summary log for this station
    print(f"[SUMMARY] Station: {station} → Inserted: {inserted_count} → Time: {record['time']}")

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
            return None, f"Model or scaler not found for station: {station}", None, None, None

        df["time"] = pd.to_datetime(df["time"])
        df["year"] = df["time"].dt.year
        df["month"] = df["time"].dt.month
        df["day"] = df["time"].dt.day
        df["hour"] = df["time"].dt.hour
        df["minute"] = df["time"].dt.minute
        df["second"] = df["time"].dt.second
        df["millisecond"] = df["time"].dt.microsecond // 1000

        df = df.rename(columns={"value": f"Actual_{station}"})
        if f"Actual_{station}" not in df.columns:
            return None, f"Column rename failed for {station}", None, None, None

        # Model Prediction
        features = ["year", "month", "day", "hour", "minute", "second", "millisecond"]
        X_scaled = scaler.transform(df[features])
        df[f"Predicted_{station}"] = model.predict(X_scaled)
        df[f"Upper_Bound_{station}"] = df[f"Predicted_{station}"] + 5
        df[f"Lower_Bound_{station}"] = df[f"Predicted_{station}"] - 5

        # Anomaly Detection
        df[f"is_anomaly_{station}"] = df.apply(
            lambda row: (row[f"Actual_{station}"] is not None) and (
                row[f"Actual_{station}"] > row[f"Upper_Bound_{station}"] or
                row[f"Actual_{station}"] < row[f"Lower_Bound_{station}"]
            ), axis=1
        )

        # Anomaly Info
        latest_anomaly = df[df[f"is_anomaly_{station}"]].sort_values("time", ascending=False).head(1)
        anomaly_info = {
            "value": latest_anomaly[f"Actual_{station}"].values[0] if not latest_anomaly.empty else None,
            "timestamp": latest_anomaly["time"].dt.strftime("%B %d, %Y %I:%M %p").values[0] if not latest_anomaly.empty else None
        }

        # Historical Pressures
        df = df.sort_values("time", ascending=False)
        latest_pressure = df.iloc[0][f"Actual_{station}"] if not df.empty else None
        current_time = df["time"].max()
        five_minutes_ago = current_time - timedelta(minutes=5)
        prev_row = df[df["time"] <= five_minutes_ago]
        previous_pressure = prev_row.iloc[0][f"Actual_{station}"] if not prev_row.empty else None

        # Predict 30 Minutes Ahead
        future_predictions = []
        for i in range(1, 31):
            future_time = current_time + timedelta(minutes=i)
            future_feat = [
                future_time.year, future_time.month, future_time.day,
                future_time.hour, future_time.minute, future_time.second,
                future_time.microsecond // 1000
            ]
            X_future = scaler.transform([future_feat])
            prediction = model.predict(X_future)[0]
            future_predictions.append({
                f"Actual_{station}": None,
                f"Predicted_{station}": prediction,
                f"Upper_Bound_{station}": prediction + 5,
                f"Lower_Bound_{station}": prediction - 5,
                "is_anomaly_" + station: False,
                "time": future_time.strftime("%a, %d %b %Y %H:%M:%S GMT"),
                "year": future_time.year,
                "month": future_time.month,
                "day": future_time.day,
                "hour": future_time.hour,
                "minute": future_time.minute,
                "second": future_time.second,
                "millisecond": future_time.microsecond // 1000
            })

        df["time"] = df["time"].dt.strftime("%a, %d %b %Y %H:%M:%S GMT")
        df_json = df.replace({np.nan: None}).to_dict(orient="records")

        # Combine real data and future prediction
        combined_data = future_predictions + df_json

        return anomaly_info, combined_data, latest_pressure, previous_pressure

    # Process Both Stations
    anomaly_002, data_002, latest_002, prev_002 = process_data(df_bslg_002, "BSLG-002")[:4]
    anomaly_003, data_003, latest_003, prev_003 = process_data(df_bslg_003, "BSLG-003")[:4]

    # Save valid data to DB
    try:
        conn = psycopg2.connect(host="209.38.56.184", database="postgres", user="admin", password="password", port=5432)
        save_to_db(data_002, "BSLG-002", conn)
        save_to_db(data_003, "BSLG-003", conn)
    except Exception as e:
        print(f"Error saving to DB: {e}")
    finally:
        if conn:
            conn.close()

    # Error Handling
    if anomaly_002 is None:
        return jsonify({"error": "Error processing BSLG-002 data"}), 500
    if anomaly_003 is None:
        return jsonify({"error": "Error processing BSLG-003 data"}), 500

    return jsonify({
        "anomaly_bslg_002": anomaly_002,
        "anomaly_bslg_003": anomaly_003,
        "latest_pressure_bslg_002": latest_002,
        "previous_pressure_bslg_002": prev_002,
        "latest_pressure_bslg_003": latest_003,
        "previous_pressure_bslg_003": prev_003,
        "data_bslg_002": data_002,
        "data_bslg_003": data_003,
    })

if __name__ == '__main__':
    app.run(debug=True)