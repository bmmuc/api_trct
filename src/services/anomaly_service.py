"""
Service layer for anomaly detection business logic.
"""
import time
from typing import Optional
from src.models.schemas import TrainData, DataPoint, TrainResponse, PredictResponse
from src.storage.base_storage import BaseModelStorage
from src.utils.base_metrics import BaseMetricsExporter
from src.utils.logger import logger
from src.exceptions import ModelNotFoundError
from src.services.base_training_service import BaseTrainingService


class AnomalyDetectionService:
    """Service using abstract storage, metrics, and training."""

    def __init__(self,
                 model_storage: BaseModelStorage,
                 metrics_exporter: BaseMetricsExporter,
                 training_service: BaseTrainingService):
        self.model_storage = model_storage
        self.metrics_exporter = metrics_exporter
        self.training_service = training_service

    def train_model(self, series_id: str, train_data: TrainData, metadata: dict = None) -> TrainResponse:
        """Delegates training to configured training service."""
        return self.training_service.train(series_id, train_data, metadata)

    def predict_anomaly(
        self,
        series_id: str,
        data_point: DataPoint,
        version: Optional[str] = None
    ) -> PredictResponse:
        """Makes prediction using abstract storage."""
        start_time = time.time()

        try:
            model, used_version = self.model_storage.load_model(series_id, version)
        except FileNotFoundError as exc:
            logger.warning(
                "Model not found for series_id='%s', version='%s'", series_id, version
            )
            raise ModelNotFoundError(series_id, version) from exc

        is_anomaly = model.predict(data_point)

        latency_ms = (time.time() - start_time) * 1000
        self.metrics_exporter.record_inference_latency(latency_ms)

        return PredictResponse(
            anomaly=is_anomaly,
            model_version=used_version
        )

    def get_trained_series_count(self) -> int:
        """Returns number of trained series."""
        return len(self.model_storage.list_all_series())
