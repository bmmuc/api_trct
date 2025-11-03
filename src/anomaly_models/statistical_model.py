"""
Statistical anomaly detection model (original implementation).
"""
import json
import numpy as np
from src.models.schemas import TimeSeries, DataPoint
from src.anomaly_models.base_model import BaseAnomalyModel


class StatisticalAnomalyModel(BaseAnomalyModel):
    """Detects anomalies using mean + N standard deviations."""

    def __init__(self, threshold: float = 3.0):
        self.threshold = threshold
        self.mean: float | None = None
        self.std: float | None = None
        self._is_fitted: bool = False

    def fit(self, data: TimeSeries) -> "StatisticalAnomalyModel":
        """Trains the model on the data."""
        values_stream = [d.value for d in data.data]

        if len(values_stream) == 0:
            raise ValueError("Cannot train on empty time series")

        self.mean = np.mean(values_stream)
        self.std = np.std(values_stream)
        self._is_fitted = True

        return self

    def predict(self, data_point: DataPoint) -> bool:
        """Checks if the point is outside the configured threshold."""
        return data_point.value > self.mean + self.threshold * self.std

    def save(self) -> bytes:
        """Serializes to JSON."""
        if not self._is_fitted:
            raise ValueError("Cannot serialize an unfitted model")

        data = {
            "model_type": "statistical",
            "threshold": self.threshold,
            "mean": self.mean,
            "std": self.std
        }
        return json.dumps(data).encode('utf-8')

    def load(self, data: bytes) -> "StatisticalAnomalyModel":
        """Loads from JSON."""
        model_data = json.loads(data.decode('utf-8'))
        self.threshold = model_data.get("threshold", 3.0)
        self.mean = model_data["mean"]
        self.std = model_data["std"]
        self._is_fitted = True
        return self

    def is_fitted(self) -> bool:
        """Checks if it was trained."""
        return self._is_fitted

    def get_model_type(self) -> str:
        """Returns model type."""
        return "statistical"
