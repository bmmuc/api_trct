"""
Local training service implementation.
"""
import time
from src.services.base_training_service import BaseTrainingService
from src.models.schemas import TrainData, TrainResponse
from src.anomaly_models.model_factory import ModelFactory
from src.storage.base_storage import BaseModelStorage
from src.utils.base_metrics import BaseMetricsExporter
from src.config import config


class LocalTrainingService(BaseTrainingService):
    """Training service that trains models locally."""

    def __init__(self, model_storage: BaseModelStorage,
                 metrics_exporter: BaseMetricsExporter):
        self.model_storage = model_storage
        self.metrics_exporter = metrics_exporter

    def train(self, series_id: str, train_data: TrainData, metadata: dict = None) -> TrainResponse:
        """Trains model locally using factory to create configured type."""
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
