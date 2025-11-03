"""
Dependency injection providers for FastAPI routes.
Uses factories to create configurable backends.
"""
import threading
from fastapi import Depends
from src.services.anomaly_service import AnomalyDetectionService
from src.services.visualization_service import VisualizationService
from src.storage.base_storage import BaseModelStorage
from src.storage.storage_factory import StorageFactory
from src.utils.base_metrics import BaseMetricsExporter
from src.utils.metrics_factory import MetricsFactory
from src.config import config

_model_storage_instance: BaseModelStorage | None = None
_metrics_exporter_instance: BaseMetricsExporter | None = None

_storage_lock = threading.Lock()
_metrics_lock = threading.Lock()


def get_model_storage() -> BaseModelStorage:
    """Creates singleton of configured storage."""
    global _model_storage_instance

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
    global _metrics_exporter_instance

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


def get_anomaly_service(
    model_storage: BaseModelStorage = Depends(get_model_storage),
    metrics_exporter: BaseMetricsExporter = Depends(get_metrics_exporter)
) -> AnomalyDetectionService:
    """Injects service with abstract dependencies."""
    return AnomalyDetectionService(model_storage, metrics_exporter)


def get_visualization_service(
    model_storage: BaseModelStorage = Depends(get_model_storage)
) -> VisualizationService:
    """Injects visualization service."""
    return VisualizationService(model_storage)
