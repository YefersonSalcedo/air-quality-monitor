import numpy as np
import json
from datetime import datetime
import logging

# Randomized prediction and automatic air quality alerts

# Configure alerts log
logging.basicConfig(
    filename="../../data/alerts.log",
    level=logging.INFO,
    format="%(message)s",
    encoding="utf-8"
)

# Alert thresholds - EPA Standards (US)
THRESHOLDS = {
    "PM25": 35.0,   # µg/m³ - EPA 24h standard
    "NO2": 100.0    # µg/m³ - EPA 1h standard
}

def predict_from_series(series, sims=500):
    """Predicts future values using random simulations."""
    if not series:
        return None
    arr = np.array(series, dtype=float)
    n = len(arr)
    samples = []
    for _ in range(sims):
        bs = np.random.choice(arr, size=n, replace=True)
        noise = np.random.normal(0, np.std(bs) * 0.05)
        samples.append(np.mean(bs) + noise)
    return float(np.mean(samples))

def predict_sensor(hash_module, sensor_id):
    """Predicts PM25 and NO2 for a sensor using recent data."""
    records = hash_module.get_sensor_records(sensor_id)
    if not records:
        return None
    recent = records[-10:]  # Last 10 readings
    pm25_series = [r["airQualityData"]["PM25"] for r in recent]
    no2_series = [r["airQualityData"]["NO2"] for r in recent]
    return {
        "sensor_id": sensor_id,
        "PM25_pred": predict_from_series(pm25_series),
        "NO2_pred": predict_from_series(no2_series)
    }

def check_and_alert(hash_module, sensor_id):
    """Generates alert if predictions exceed limits."""
    pred = predict_sensor(hash_module, sensor_id)
    if not pred:
        return None
    alerts = []
    for contaminant, limit in THRESHOLDS.items():
        value = pred[f"{contaminant}_pred"]
        if value and value > limit:
            alert = {
                "sensor_id": sensor_id,
                "contaminant": contaminant,
                "predicted_value": value,
                "threshold": limit,
                "time": datetime.utcnow().isoformat() + "Z"
            }
            logging.info(json.dumps(alert, ensure_ascii=False))
            alerts.append(alert)
    return alerts if alerts else None

def run_all_and_alert(hash_module):
    """Runs predictions and alerts for all sensors."""
    sensor_ids = hash_module.get_all_sensor_ids()
    all_alerts = []
    for sid in sensor_ids:
        result = check_and_alert(hash_module, sid)
        if result:
            all_alerts.extend(result)
    return all_alerts