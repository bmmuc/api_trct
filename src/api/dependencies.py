"""
Dependency injection providers for FastAPI routes.

This module defines dependency provider functions that FastAPI will use to inject
dependencies into route handlers via the Depends() mechanism.
"""
from functools import lru_cache
from fastapi import Depends
from src.services.anomaly_service import AnomalyDetectionService
from src.storage.model_store import ModelStore
from src.utils.metrics import MetricsTracker

_model_store_instance: ModelStore | None = None
_metrics_tracker_instance: MetricsTracker | None = None


def get_model_store() -> ModelStore:
    """
    Dependency provider for ModelStore.

    Returns:
        ModelStore: Singleton instance of ModelStore
    """
    global _model_store_instance
    if _model_store_instance is None:
        _model_store_instance = ModelStore()
    return _model_store_instance


def get_metrics_tracker() -> MetricsTracker:
    """
    Dependency provider for MetricsTracker.
    Returns:
        MetricsTracker: Singleton instance of MetricsTracker
    """
    global _metrics_tracker_instance
    if _metrics_tracker_instance is None:
        _metrics_tracker_instance = MetricsTracker()
    return _metrics_tracker_instance


def get_anomaly_service(
    model_store: ModelStore = Depends(get_model_store),
    metrics_tracker: MetricsTracker = Depends(get_metrics_tracker)
) -> AnomalyDetectionService:
    """
    Dependency provider for AnomalyDetectionService.

    Args:
        model_store: Injected by FastAPI via get_model_store()
        metrics_tracker: Injected by FastAPI via get_metrics_tracker()

    Returns:
        AnomalyDetectionService: New instance with injected dependencies
    """
    return AnomalyDetectionService(model_store, metrics_tracker)
