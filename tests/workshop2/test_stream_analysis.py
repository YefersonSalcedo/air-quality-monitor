import unittest
import numpy as np
from datetime import datetime, timedelta

from src.workshop2.stream_analysis import DGIM, StreamAnalyzer

# Unit tests for DGIM algorithm and stream analysis


class TestDGIM(unittest.TestCase):
    """Tests for DGIM bit counting algorithm."""

    def test_add_and_estimate_simple(self):
        """Verifies correct counting in a simple bit sequence."""
        dgim = DGIM(window_size=10)
        bits = [1, 0, 1, 1, 0, 0, 1, 0, 1, 1]
        for b in bits:
            dgim.add_bit(b)
        estimate = dgim.estimate_count(10)
        expected = sum(bits)
        self.assertTrue(
            abs(estimate - expected) <= 1,
            f"Incorrect DGIM estimation: {estimate}, expected: {expected}"
        )

    def test_window_expiration(self):
        """Verifies that old buckets are expired correctly."""
        dgim = DGIM(window_size=5)
        for _ in range(10):
            dgim.add_bit(1)
        estimate = dgim.estimate_count(5)
        self.assertTrue(
            estimate <= 6,
            "Old buckets were not removed correctly."
        )


class TestStreamAnalyzer(unittest.TestCase):
    """Tests for PM2.5 stream analyzer with DGIM."""

    def setUp(self):
        """Initializes analyzer with artificial PM2.5 stream data."""
        self.analyzer = StreamAnalyzer(window_size=100, threshold=35.0)
        self.timestamps = [
            datetime(2025, 11, 12, 10, 0) + timedelta(minutes=i)
            for i in range(200)
        ]
        pm25_values = np.concatenate([
            np.random.normal(20, 3, 80),
            np.linspace(20, 60, 40),
            np.random.normal(55, 3, 40),
            np.linspace(60, 25, 40)
        ])
        self.pm25_values = pm25_values

    def test_alert_conversion_and_estimation(self):
        """Verifies PM2.5 readings convert to bits and estimate correctly."""
        for t, v in zip(self.timestamps, self.pm25_values):
            self.analyzer.add_reading(t, "sensor1", v)
        exact = self.analyzer.exact_alerts(100)
        estimate = self.analyzer.estimate_alerts(100)
        self.assertTrue(
            abs(exact - estimate) <= 30,
            f"DGIM should approximate alert count: exact={exact}, est={estimate}"
        )

    def test_trend_detection_increasing(self):
        """Simulates sustained pollution increase (positive trend)."""
        an = StreamAnalyzer(window_size=200, threshold=35.0)
        for i in range(200):
            val = 20 + 0.3 * i
            an.add_reading(datetime.now(), "sensorA", val)
        result = an.detect_trend(window_length=100, step=10, lookback_windows=5)
        self.assertEqual(
            result['trend'], 'increasing',
            f"Expected trend: increasing, got: {result}"
        )

    def test_trend_detection_decreasing(self):
        """Simulates sustained air quality improvement (negative trend)."""
        an = StreamAnalyzer(window_size=200, threshold=35.0)
        for i in range(200):
            val = 60 - 0.3 * i
            an.add_reading(datetime.now(), "sensorB", val)
        result = an.detect_trend(window_length=100, step=10, lookback_windows=5)
        self.assertIn(
            result['trend'], ['decreasing', 'stable'],
            f"Expected trend: decreasing or stable, got: {result}"
        )

    def test_trend_detection_stable(self):
        """Simulates stable conditions without notable changes."""
        an = StreamAnalyzer(window_size=200, threshold=35.0)
        for i in range(200):
            val = 30 + np.random.normal(0, 1)
            an.add_reading(datetime.now(), "sensorC", val)
        result = an.detect_trend(window_length=100, step=10, lookback_windows=5)
        self.assertEqual(
            result['trend'], 'stable',
            f"Expected trend: stable, got: {result}"
        )

    def test_prediction_output(self):
        """Verifies prediction function returns positive and reasonable value."""
        for t, v in zip(self.timestamps, self.pm25_values):
            self.analyzer.add_reading(t, "sensor1", v)
        prediction = self.analyzer.predict_counts(window_length=50, lookahead=1)
        self.assertIsInstance(prediction, float)
        self.assertTrue(
            prediction >= 0,
            "Count prediction must not be negative."
        )