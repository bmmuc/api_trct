"""
Service layer for anomaly detection business logic.
"""
import time
from typing import Optional
from src.anomaly_models.anomaly_model import AnomalyDetectionModel
from src.models.schemas import TrainData, DataPoint, TrainResponse, PredictResponse
from src.storage.model_store import ModelStore
from src.utils.metrics import MetricsTracker
from src.utils.logger import logger
from src.exceptions import ModelNotFoundError, ModelNotFittedError


class AnomalyDetectionService:
    """
    Service for running anomaly detection.
    """

    def __init__(self, model_store: ModelStore, metrics_tracker: MetricsTracker):
        """
        Initialize the service.

        Args:
            model_store: ModelStore instance for persistence
        """

        self.model_store = model_store
        self.metrics_tracker = metrics_tracker


    def train_model(self, series_id: str, train_data: TrainData) -> TrainResponse:
        """
        Train a new anomaly detection model for a series.

        Args:
            series_id: Identifier for the time series
            train_data: Training data

        Returns:
            TrainResponse with training details
        """

        # Start timing
        start_time = time.time()

        time_series = train_data.to_time_series()

        # Create and train model
        model = AnomalyDetectionModel()
        model.fit(time_series)

        # Save model with versioning
        version = self.model_store.save_model(series_id, model)

        # Record latency in milliseconds
        latency_ms = (time.time() - start_time) * 1000
        self.metrics_tracker.record_training_latency(latency_ms)

        return TrainResponse(
            series_id=series_id,
            version=version,
            points_used=len(train_data.values)
        )


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
        """
        # Start timing
        start_time = time.time()

        # Load model - this will raise ModelNotFoundError if not found
        try:
            model, used_version = self.model_store.load_model(series_id, version)
        except FileNotFoundError:
            logger.warning(f"Model not found for series_id='{series_id}', version='{version}'")
            raise ModelNotFoundError(series_id, version)

        is_anomaly = model.predict(data_point)

        # Record latency in milliseconds
        latency_ms = (time.time() - start_time) * 1000
        self.metrics_tracker.record_inference_latency(latency_ms)

        return PredictResponse(
            anomaly=is_anomaly,
            model_version=used_version
        )

    def get_trained_series_count(self) -> int:
        """
        Get the number of series with trained models.

        Returns:
            Count of unique series_ids with models
        """
        return len(self.model_store.list_all_series())
