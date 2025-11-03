"""
Base interface for anomaly detection models.
"""
from abc import ABC, abstractmethod
from src.models.schemas import TimeSeries, DataPoint


class BaseAnomalyModel(ABC):
    """Interface for different model types."""

    @abstractmethod
    def fit(self, data: TimeSeries) -> "BaseAnomalyModel":
        """Trains the model with historical data."""

    @abstractmethod
    def predict(self, data_point: DataPoint) -> bool:
        """Detects if a point is anomalous."""

    @abstractmethod
    def save(self) -> bytes:
        """Serializes the model (JSON, pickle, ONNX, etc)."""

    @abstractmethod
    def load(self, data: bytes) -> "BaseAnomalyModel":
        """Loads the serialized model."""

    @abstractmethod
    def is_fitted(self) -> bool:
        """Checks if the model was trained."""

    @abstractmethod
    def get_model_type(self) -> str:
        """Returns the model type for persistence."""
