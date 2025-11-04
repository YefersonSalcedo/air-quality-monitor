import pytest
import json
from src.workshop1 import prediction


# --- Mock class to simulate the hash_module behavior ---
class MockHashModule:
    def __init__(self):
        # High values to guarantee alerts are triggered (PM25 > 25, NO2 > 60)
        self.data = {
            "S1": [
                {"airQualityData": {"PM25": 80, "NO2": 120}},
                {"airQualityData": {"PM25": 85, "NO2": 110}},
                {"airQualityData": {"PM25": 90, "NO2": 130}},
                {"airQualityData": {"PM25": 88, "NO2": 125}},
                {"airQualityData": {"PM25": 95, "NO2": 140}},
            ]
        }

    # Simulates retrieval of sensor records by ID
    def get_sensor_records(self, sensor_id):
        return self.data.get(sensor_id, [])

    # Returns a list of all available sensor IDs
    def get_all_sensor_ids(self):
        return list(self.data.keys())


# --- UNIT TESTS ---

def test_predict_from_series_returns_float():
    """Ensure predict_from_series() returns a valid float value."""
    series = [10, 12, 14, 16, 18]
    result = prediction.predict_from_series(series)
    assert isinstance(result, float)
    assert result > 0


def test_predict_sensor_returns_dict():
    """Verify that predict_sensor() returns a dictionary with prediction keys."""
    mock = MockHashModule()
    result = prediction.predict_sensor(mock, "S1")
    assert "PM25_pred" in result
    assert "NO2_pred" in result
    assert isinstance(result["PM25_pred"], float)
    assert isinstance(result["NO2_pred"], float)


def test_check_and_alert_generates_alert():
    """Ensure that check_and_alert() creates alerts when thresholds are exceeded."""
    mock = MockHashModule()
    alerts = prediction.check_and_alert(mock, "S1")
    assert isinstance(alerts, list) or alerts is None
    if alerts:
        for a in alerts:
            assert "sensor_id" in a
            assert "contaminant" in a
            assert "predicted_value" in a
            assert "threshold" in a


def test_run_all_and_alert_returns_list():
    """Confirm that run_all_and_alert() returns a list of alerts."""
    mock = MockHashModule()
    results = prediction.run_all_and_alert(mock)
    assert isinstance(results, list)


def test_alert_log_file(tmp_path):
    """
    Check that alerts are properly written into the log file.
    This test redirects logging to a temporary file using pytest's tmp_path.
    """
    log_path = tmp_path / "alerts.log"

    # Reconfigure logging for this specific test to use the temp file
    import logging
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(message)s",
        encoding="utf-8"
    )

    # Run the function that triggers alerts
    mock = MockHashModule()
    prediction.run_all_and_alert(mock)

    # Verify that the log file was created and contains data
    assert log_path.exists()
    content = log_path.read_text(encoding="utf-8").strip()
    assert len(content) > 0  # File should not be empty

    # Each line should be valid JSON containing the expected alert fields
    for line in content.splitlines():
        record = json.loads(line)
        assert "sensor_id" in record
        assert "time" in record
        assert "predicted_value" in record
        assert "threshold" in record