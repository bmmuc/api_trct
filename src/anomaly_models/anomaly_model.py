"""
Anomaly Detection Model implementation.
"""
from typing import Dict
import numpy as np
from src.models.schemas import TimeSeries, DataPoint


class AnomalyDetectionModel:
    """
    Simple anomaly detection model based on statistical thresholds.
    Detects anomalies using mean + 3 standard deviations rule.
    """

    def __init__(self):
        self.mean: float | None = None
        self.std: float | None = None
        self._is_fitted: bool = False

    def fit(self, data: TimeSeries) -> "AnomalyDetectionModel":
        """
        Train the model on time series data.

        Args:
            data: TimeSeries object containing training data

        Returns:
            Self for method chaining
        """
        values_stream = [d.value for d in data.data]

        if len(values_stream) == 0:
            raise ValueError("Cannot train on empty time series")

        self.mean = np.mean(values_stream)
        self.std = np.std(values_stream)
        self._is_fitted = True

        return self

    def predict(self, data_point: DataPoint) -> bool:
        """
        Predict if a data point is anomalous.

        Args:
            data_point: DataPoint to check for anomaly

        Returns:
            True if the point is anomalous, False otherwise
        """
        return data_point.value > self.mean + 3 * self.std

    def to_dict(self) -> Dict:
        """
        Serialize the model to a dictionary.

        Returns:
            Dictionary representation of the model
        """
        if not self._is_fitted:
            raise ValueError("Cannot serialize an unfitted model")

        return {
            "mean": self.mean,
            "std": self.std
        }

    def from_dict(self, data: Dict) -> "AnomalyDetectionModel":
        """
        Load the model from a dictionary.

        Args:
            data: Dictionary containing model parameters

        Returns:
            Self for method chaining
        """
        self.mean = data["mean"]
        self.std = data["std"]
        self._is_fitted = True
        return self
