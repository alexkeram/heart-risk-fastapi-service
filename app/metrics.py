# app/metrics.py
from __future__ import annotations
from collections import deque
import threading
import math
from time import monotonic

class Metrics:
    """
    Stream-safe metrics:
        - requests_total
        - p95 latency based on the window of the last N values (default is 500)
        - p95 = None if observations < 20
    """
    def __init__(self, window_size: int = 500) -> None:
        self._lock = threading.Lock()
        self._lat_ms = deque(maxlen=window_size)
        self._requests_total = 0
        self._started_at = monotonic()
        self._window_size = window_size

    def observe(self, dt_ms: int | float) -> None:
        with self._lock:
            self._requests_total += 1
            self._lat_ms.append(float(dt_ms))

    def snapshot(self) -> dict:
        with self._lock:
            n = len(self._lat_ms)
            p95 = None
            if n >= 20:
                arr = sorted(self._lat_ms)  # short copy (<=500)
                # position on nearest-rank: ceil(0.95*n) - 1 (0-index)
                idx = max(0, int(math.ceil(0.95 * n)) - 1)
                p95 = round(arr[idx], 3)
            return {
                "requests_total": self._requests_total,
                "latency_p95_ms": p95,
                "window_size": self._window_size,
                "n_samples": n,
                # uptime:
                # "uptime_s": int(monotonic() - self._started_at),
            }
