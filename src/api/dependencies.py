"""
Dependency injection providers for FastAPI routes.
Uses factories to create configurable backends.
"""
import threading
from fastapi import Depends
from src.services.anomaly_service import AnomalyDetectionService
from src.services.visualization_service import VisualizationService
from src.services.base_training_service import BaseTrainingService
from src.services.local_training_service import LocalTrainingService
from src.services.external_training_service import ExternalTrainingService
from src.storage.base_storage import BaseModelStorage
from src.storage.storage_factory import StorageFactory
from src.utils.base_metrics import BaseMetricsExporter
from src.utils.metrics_factory import MetricsFactory
from src.config import config

_model_storage_instance: BaseModelStorage | None = None
_metrics_exporter_instance: BaseMetricsExporter | None = None
_training_service_instance: BaseTrainingService | None = None

_storage_lock = threading.Lock()
_metrics_lock = threading.Lock()
_training_lock = threading.Lock()


def get_model_storage() -> BaseModelStorage:
    """Creates singleton of configured storage."""
    global _model_storage_instance # pylint: disable=global-statement

    if _model_storage_instance is None:
        with _storage_lock:
            if _model_storage_instance is None:
                if config.storage_type == "filesystem":
                    _model_storage_instance = StorageFactory.create(
                        "filesystem",
                        storage_path=config.filesystem.storage_path
                    )
                elif config.storage_type == "s3":
                    _model_storage_instance = StorageFactory.create(
                        "s3",
                        bucket_name=config.s3.bucket,
                        prefix=config.s3.prefix
                    )

    return _model_storage_instance


def get_metrics_exporter() -> BaseMetricsExporter:
    """Creates singleton of configured metrics exporter."""
    global _metrics_exporter_instance # pylint: disable=global-statement

    if _metrics_exporter_instance is None:
        with _metrics_lock:
            if _metrics_exporter_instance is None:
                if config.metrics_type == "memory":
                    _metrics_exporter_instance = MetricsFactory.create(
                        "memory",
                        max_samples=config.memory.max_samples
                    )
                elif config.metrics_type == "prometheus":
                    _metrics_exporter_instance = MetricsFactory.create(
                        "prometheus",
                        namespace=config.prometheus.namespace
                    )

    return _metrics_exporter_instance


def get_training_service(
    model_storage: BaseModelStorage = Depends(get_model_storage),
    metrics_exporter: BaseMetricsExporter = Depends(get_metrics_exporter)
) -> BaseTrainingService:
    """Creates singleton of configured training service."""
    global _training_service_instance # pylint: disable=global-statement

    if _training_service_instance is None:
        with _training_lock:
            if _training_service_instance is None:
                if config.training_type == "local":
                    _training_service_instance = LocalTrainingService(
                        model_storage=model_storage,
                        metrics_exporter=metrics_exporter
                    )
                elif config.training_type == "external":
                    _training_service_instance = ExternalTrainingService(
                        api_url=config.external_training.api_url,
                        metrics_exporter=metrics_exporter,
                        api_key=config.external_training.api_key,
                        timeout=config.external_training.timeout
                    )

    return _training_service_instance


def get_anomaly_service(
    model_storage: BaseModelStorage = Depends(get_model_storage),
    metrics_exporter: BaseMetricsExporter = Depends(get_metrics_exporter),
    training_service: BaseTrainingService = Depends(get_training_service)
) -> AnomalyDetectionService:
    """Injects service with abstract dependencies."""
    return AnomalyDetectionService(model_storage, metrics_exporter, training_service)


def get_visualization_service(
    model_storage: BaseModelStorage = Depends(get_model_storage)
) -> VisualizationService:
    """Injects visualization service."""
    return VisualizationService(model_storage)
