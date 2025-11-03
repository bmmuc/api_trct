"""
Interface base para diferentes exportadores de métricas.
"""
from abc import ABC, abstractmethod
from src.models.schemas import Metrics


class BaseMetricsExporter(ABC):
    """Interface para exportar métricas de diferentes formas."""

    @abstractmethod
    def record_training_latency(self, latency_ms: float):
        """Registra latência de treino."""
        pass

    @abstractmethod
    def record_inference_latency(self, latency_ms: float):
        """Registra latência de inferência."""
        pass

    @abstractmethod
    def get_training_metrics(self) -> Metrics:
        """Retorna métricas agregadas de treino."""
        pass

    @abstractmethod
    def get_inference_metrics(self) -> Metrics:
        """Retorna métricas agregadas de inferência."""
        pass

    @abstractmethod
    def export(self) -> str:
        """Exporta métricas em formato específico (JSON, Prometheus, etc)."""
        pass

    @abstractmethod
    def reset(self):
        """Limpa métricas registradas."""
        pass
