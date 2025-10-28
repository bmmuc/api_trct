"""
Service layer for anomaly detection business logic.
"""
from typing import Optional
from models.anomaly_model import AnomalyDetectionModel
from models.schemas import TrainData, DataPoint, TrainResponse, PredictResponse
from storage.model_store import ModelStore


class AnomalyDetectionService:
    """
    Service layer handling business logic for anomaly detection.
    """

    def __init__(self, model_store: ModelStore):
        """
        Initialize the service.

        Args:
            model_store: ModelStore instance for persistence
        """
        self.model_store = model_store


    def train_model(self, series_id: str, train_data: TrainData) -> TrainResponse:
        """
        Train a new anomaly detection model for a series.

        Args:
            series_id: Identifier for the time series
            train_data: Training data

        Returns:
            TrainResponse with training details
        """
        pass

    def predict_anomaly(
        self,
        series_id: str,
        data_point: DataPoint,
        version: Optional[str] = None
    ) -> PredictResponse:
        """
        Predict if a data point is anomalous.

        Args:
            series_id: Identifier for the time series
            data_point: DataPoint to check
            version: Optional model version, uses latest if not provided

        Returns:
            PredictResponse with prediction and version used

        Raises:
            FileNotFoundError: If no model exists for series_id
        """
        pass

    def get_trained_series_count(self) -> int:
        """
        Get the number of series with trained models.

        Returns:
            Count of unique series_ids with models
        """
        pass