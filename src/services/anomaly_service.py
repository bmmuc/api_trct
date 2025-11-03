"""
Service layer for anomaly detection business logic.
"""
import time
from typing import Optional
from src.anomaly_models.model_factory import ModelFactory
from src.models.schemas import TrainData, DataPoint, TrainResponse, PredictResponse
from src.storage.base_storage import BaseModelStorage
from src.utils.base_metrics import BaseMetricsExporter
from src.utils.logger import logger
from src.exceptions import ModelNotFoundError
from src.config import config


class AnomalyDetectionService:
    """Service using abstract storage and metrics."""

    def __init__(self, model_storage: BaseModelStorage,
                 metrics_exporter: BaseMetricsExporter):
        self.model_storage = model_storage
        self.metrics_exporter = metrics_exporter

    def train_model(self, series_id: str, train_data: TrainData) -> TrainResponse:
        """Trains model using factory to create configured type."""
        start_time = time.time()

        time_series = train_data.to_time_series()

        # Get model specific config
        model_config = getattr(config, config.model_type)
        model = ModelFactory.create(config.model_type, **model_config.model_dump())
        model.fit(time_series)

        version = self.model_storage.save_model(series_id, model)

        latency_ms = (time.time() - start_time) * 1000
        self.metrics_exporter.record_training_latency(latency_ms)

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
