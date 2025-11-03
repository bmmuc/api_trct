"""
Base interface for different metrics exporters.
"""
from abc import ABC, abstractmethod
from src.models.schemas import Metrics


class BaseMetricsExporter(ABC):
    """Interface for exporting metrics in different ways."""

    @abstractmethod
    def record_training_latency(self, latency_ms: float):
        """Records training latency."""

    @abstractmethod
    def record_inference_latency(self, latency_ms: float):
        """Records inference latency."""

    @abstractmethod
    def get_training_metrics(self) -> Metrics:
        """Returns aggregated training metrics."""

    @abstractmethod
    def get_inference_metrics(self) -> Metrics:
        """Returns aggregated inference metrics."""

    @abstractmethod
    def export(self) -> str:
        """Exports metrics in specific format (JSON, Prometheus, etc)."""

    @abstractmethod
    def reset(self):
        """Clears recorded metrics."""
