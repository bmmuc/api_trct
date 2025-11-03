"""
Métricas em memória (implementação original).
"""
import json
import threading
from collections import deque
import numpy as np
from src.models.schemas import Metrics
from src.utils.base_metrics import BaseMetricsExporter


class MemoryMetricsExporter(BaseMetricsExporter):
    """Store and export metrics in memory."""

    def __init__(self, max_samples: int = 10000):
        self._training_latencies: deque = deque(maxlen=max_samples)
        self._inference_latencies: deque = deque(maxlen=max_samples)
        self._lock = threading.Lock()

    def record_training_latency(self, latency_ms: float):
        with self._lock:
            self._training_latencies.append(latency_ms)

    def record_inference_latency(self, latency_ms: float):
        with self._lock:
            self._inference_latencies.append(latency_ms)

    def get_training_metrics(self) -> Metrics:
        with self._lock:
            if not self._training_latencies:
                return Metrics(avg=None, p95=None)
            snapshot = list(self._training_latencies)

        return Metrics(
            avg=float(np.mean(snapshot)),
            p95=float(np.percentile(snapshot, 95))
        )

    def get_inference_metrics(self) -> Metrics:
        with self._lock:
            if not self._inference_latencies:
                return Metrics(avg=None, p95=None)
            snapshot = list(self._inference_latencies)

        return Metrics(
            avg=float(np.mean(snapshot)),
            p95=float(np.percentile(snapshot, 95))
        )

    def export(self) -> str:
        """Exporta métricas em formato JSON."""
        training = self.get_training_metrics()
        inference = self.get_inference_metrics()

        return json.dumps({
            "training": {
                "avg_latency_ms": training.avg,
                "p95_latency_ms": training.p95
            },
            "inference": {
                "avg_latency_ms": inference.avg,
                "p95_latency_ms": inference.p95
            }
        }, indent=2)

    def reset(self):
        with self._lock:
            self._training_latencies.clear()
            self._inference_latencies.clear()
