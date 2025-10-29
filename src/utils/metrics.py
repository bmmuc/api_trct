"""
Metrics tracking and performance monitoring utilities.
"""
import time
from typing import List, Dict
from collections import defaultdict
from functools import wraps
import numpy as np
from src.models.schemas import Metrics


class MetricsTracker:
    """
    Tracks performance metrics for API operations.
    """

    def __init__(self):
        """Initialize metrics tracker."""
        self._training_latencies: List[float] = []
        self._inference_latencies: List[float] = []

    def record_training_latency(self, latency_ms: float):
        """Record a training operation latency."""
        self._training_latencies.append(latency_ms)

    def record_inference_latency(self, latency_ms: float):
        """Record an inference operation latency."""
        self._inference_latencies.append(latency_ms)

    def get_training_metrics(self) -> Metrics:
        """
        Get training latency metrics.

        Returns:
            Metrics object with avg and p95 values
        """
        if not self._training_latencies:
            return Metrics(avg=None, p95=None)

        return Metrics(
            avg=float(np.mean(self._training_latencies)),
            p95=float(np.percentile(self._training_latencies, 95))
        )

    def get_inference_metrics(self) -> Metrics:
        """
        Get inference latency metrics.

        Returns:
            Metrics object with avg and p95 values
        """
        if not self._inference_latencies:
            return Metrics(avg=None, p95=None)

        return Metrics(
            avg=float(np.mean(self._inference_latencies)),
            p95=float(np.percentile(self._inference_latencies, 95))
        )

    def reset(self):
        """Clear all recorded metrics."""
        self._training_latencies.clear()
        self._inference_latencies.clear()
