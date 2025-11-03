"""
Factory to create different metrics exporters.
"""
from src.utils.base_metrics import BaseMetricsExporter
from src.utils.memory_metrics import MemoryMetricsExporter
from src.utils.prometheus_metrics import PrometheusMetricsExporter


class MetricsFactory:
    """Creates metrics exporter based on configuration."""

    @classmethod
    def create(cls, metrics_type: str, **kwargs) -> BaseMetricsExporter:
        """Instantiates exporter by type."""
        if metrics_type == "memory":
            return MemoryMetricsExporter(**kwargs)
        elif metrics_type == "prometheus":
            return PrometheusMetricsExporter(**kwargs)
        else:
            raise ValueError(f"Metrics type '{metrics_type}' not supported")
