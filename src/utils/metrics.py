"""
Metrics tracking and performance monitoring utilities.
"""
import threading
from collections import deque
import numpy as np
from src.models.schemas import Metrics


class MetricsTracker:
    """
    Tracker for performance metrics in API operations.
    Stores latency metrics for training and inference operations.
    """

    def __init__(self, max_samples: int = 10000):
        """
        Initialize metrics tracker.

        Args:
            max_samples: Maximum number of samples to store per metric type.
                        When limit is reached, oldest samples are discarded.
                        Default: 10000 samples
        """
        self._training_latencies: deque = deque(maxlen=max_samples)
        self._inference_latencies: deque = deque(maxlen=max_samples)

        self._lock = threading.Lock()

    def record_training_latency(self, latency_ms: float):
        """
        Record a training operation latency.

        Args:
            latency_ms: Latency in milliseconds
        """
        with self._lock:
            self._training_latencies.append(latency_ms)

    def record_inference_latency(self, latency_ms: float):
        """
        Record an inference operation latency.

        Args:
            latency_ms: Latency in milliseconds
        """
        with self._lock:
            self._inference_latencies.append(latency_ms)

    def get_training_metrics(self) -> Metrics:
        """
        Get training latency metrics.

        Returns:
            Metrics object with avg and p95 values, or None values if no data
        """
        with self._lock:
            if not self._training_latencies:
                return Metrics(avg=None, p95=None)

            snapshot = list(self._training_latencies)

        return Metrics(
            avg=float(np.mean(snapshot)),
            p95=float(np.percentile(snapshot, 95))
        )

    def get_inference_metrics(self) -> Metrics:
        """
        Get inference latency metrics.

        Returns:
            Metrics object with avg and p95 values, or None values if no data
        """
        with self._lock:
            if not self._inference_latencies:
                return Metrics(avg=None, p95=None)

            snapshot = list(self._inference_latencies)

        return Metrics(
            avg=float(np.mean(snapshot)),
            p95=float(np.percentile(snapshot, 95))
        )

    def reset(self):
        """
        Clear all recorded metrics.

        """
        with self._lock:
            self._training_latencies.clear()
            self._inference_latencies.clear()
