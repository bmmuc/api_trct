"""
Mock de exportador de métricas para Prometheus.
"""
from src.models.schemas import Metrics
from src.utils.base_metrics import BaseMetricsExporter


class PrometheusMetricsExporter(BaseMetricsExporter):
    """Mock de exporter para formato Prometheus."""

    def __init__(self, namespace: str = "anomaly_detection"):
        self.namespace = namespace
        # Mock: from prometheus_client import Counter, Histogram
        # Mock: self.training_latency_hist = Histogram(f'{namespace}_training_latency_ms', ...)
        # Mock: self.inference_latency_hist = Histogram(f'{namespace}_inference_latency_ms', ...)

    def record_training_latency(self, latency_ms: float):
        """Simula registro no histogram do Prometheus."""
        # Mock: self.training_latency_hist.observe(latency_ms)
        pass

    def record_inference_latency(self, latency_ms: float):
        """Simula registro no histogram do Prometheus."""
        # Mock: self.inference_latency_hist.observe(latency_ms)
        pass

    def get_training_metrics(self) -> Metrics:
        """Pega métricas agregadas (do Prometheus)."""
        # Mock: buscar do prometheus client
        return Metrics(avg=None, p95=None)

    def get_inference_metrics(self) -> Metrics:
        """Pega métricas agregadas (do Prometheus)."""
        # Mock: buscar do prometheus client
        return Metrics(avg=None, p95=None)

    def export(self) -> str:
        """Exporta no formato Prometheus text."""
        # Mock: from prometheus_client import generate_latest
        # Mock: return generate_latest().decode('utf-8')
        return """# HELP anomaly_detection_training_latency_ms Training latency
# TYPE anomaly_detection_training_latency_ms histogram
anomaly_detection_training_latency_ms_bucket{le="10.0"} 5
anomaly_detection_training_latency_ms_bucket{le="50.0"} 20
anomaly_detection_training_latency_ms_bucket{le="+Inf"} 25
anomaly_detection_training_latency_ms_count 25
anomaly_detection_training_latency_ms_sum 750.5
"""

    def reset(self):
        """Prometheus geralmente não reseta métricas."""
        # Mock: registries não costumam ter reset
        pass
