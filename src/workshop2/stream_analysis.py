import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import List

# Stream analysis using DGIM algorithm for air quality monitoring

# DGIM bucket structure
@dataclass
class Bucket:
    """Represents a bucket of bits in DGIM summary."""
    size: int
    timestamp: int


class DGIM:
    """
    DGIM algorithm implementation for approximate counting of 1s
    (alert events) in a fixed-length sliding window.
    """

    def __init__(self, window_size: int, max_buckets_per_size: int = 2):
        self.window_size = window_size
        self.max_buckets_per_size = max_buckets_per_size
        self.buckets: List[Bucket] = []
        self.current_index = 0

    def add_bit(self, bit: int):
        """Adds a new bit to the stream (1 = alert, 0 = no alert)."""
        self.current_index += 1
        if bit == 0:
            self._expire_old_buckets()
            return
        new_bucket = Bucket(size=1, timestamp=self.current_index)
        self.buckets.insert(0, new_bucket)
        self._compress()
        self._expire_old_buckets()

    def estimate_count(self, k: int) -> int:
        """Estimates the number of 1s in the last k bits."""
        if k <= 0:
            return 0
        if k > self.window_size:
            k = self.window_size
        cutoff_index = self.current_index - k
        total = 0.0
        for b in self.buckets:
            if b.timestamp <= cutoff_index:
                break
            bucket_start = b.timestamp - b.size + 1
            if bucket_start > cutoff_index:
                total += b.size
            else:
                total += b.size / 2.0
                break
        return int(round(total))

    def _expire_old_buckets(self):
        """Removes buckets that fall outside the sliding window."""
        cutoff = self.current_index - self.window_size
        while self.buckets and self.buckets[-1].timestamp <= cutoff:
            self.buckets.pop()

    def _compress(self):
        """Maintains DGIM invariant: max 2 buckets per size."""
        i = 0
        while i < len(self.buckets):
            size_i = self.buckets[i].size
            j = i
            while j < len(self.buckets) and self.buckets[j].size == size_i:
                j += 1
            count = j - i
            if count > self.max_buckets_per_size:
                oldest1 = self.buckets.pop(j - 1)
                oldest2 = self.buckets.pop(j - 2)
                merged = Bucket(size=oldest1.size * 2, timestamp=oldest1.timestamp)
                self.buckets.insert(j - 2, merged)
                i = 0
                continue
            i = j


class StreamAnalyzer:
    """
    Converts PM2.5 readings into a binary stream (alerts) and applies DGIM
    to monitor pollution trends in real-time.
    """

    def __init__(self, window_size=1024, threshold=35.0, max_buckets_per_size=2):
        self.window_size = window_size
        self.threshold = threshold
        self.dgim = DGIM(window_size=window_size, max_buckets_per_size=max_buckets_per_size)
        self.exact_buffer = deque(maxlen=window_size)
        self.timestamps = deque(maxlen=window_size)
        self.index = 0

    def add_reading(self, timestamp, sensor_id, pm25_value, **metadata):
        """Adds a new PM2.5 sensor reading to the stream."""
        bit = 1 if pm25_value >= self.threshold else 0
        self.index += 1
        self.dgim.add_bit(bit)
        self.exact_buffer.append(bit)
        self.timestamps.append(timestamp)

    def estimate_alerts(self, k=None):
        """Returns estimated number of alerts in the last k readings."""
        if k is None:
            k = self.window_size
        return self.dgim.estimate_count(k)

    def exact_alerts(self, k=None):
        """Returns exact count for evaluation or debugging."""
        if k is None:
            k = len(self.exact_buffer)
        return sum(list(self.exact_buffer)[-k:])

    def detect_trend(self, window_length, step=1, lookback_windows=5, change_threshold=0.2):
        """Detects pollution trend based on DGIM estimates."""
        counts = []
        for i in range(lookback_windows):
            k = window_length - i * step
            if k <= 0:
                break
            counts.append(self.estimate_alerts(k))
        counts = list(reversed(counts))
        y = np.array(counts, dtype=float)
        if len(y) < 2:
            return {'counts': counts, 'trend': 'stable', 'percent_change': 0.0, 'sustained': False}
        if not hasattr(self, "_last_nonzero"):
            self._last_nonzero = 0
        if np.any(y > 0):
            self._last_nonzero = float(np.max(y))
        A = np.vstack([np.arange(len(y)), np.ones(len(y))]).T
        m, _ = np.linalg.lstsq(A, y, rcond=None)[0]
        percent_change = (y[-1] - y[0]) / (abs(y[0]) + 1e-9)
        if np.all(y == 0):
            trend = 'decreasing' if self._last_nonzero > 0 else 'stable'
        elif m > 0 and percent_change > change_threshold:
            trend = 'increasing'
        elif m < 0 and abs(percent_change) > change_threshold:
            trend = 'decreasing'
        else:
            trend = 'stable'
        diffs = np.diff(y)
        signs = np.sign(diffs)
        sustained = (np.all(signs >= 0) or np.all(signs <= 0)) and (abs(percent_change) > change_threshold / 2)
        return {
            'counts': counts,
            'trend': trend,
            'percent_change': percent_change,
            'sustained': sustained
        }

    def predict_counts(self, window_length, lookahead=1):
        """Predicts future alert counts using simple linear regression."""
        lookback = min(5, max(2, int(self.window_size / window_length)))
        counts = [self.estimate_alerts(window_length) for _ in range(lookback)]
        x = np.arange(len(counts))
        y = np.array(counts, dtype=float)
        if len(y) < 2 or np.all(y == y[0]):
            return float(y[-1]) if len(y) > 0 else 0.0
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        pred_x = len(x) + (lookahead - 1)
        pred = m * pred_x + c
        return max(0.0, float(pred))
