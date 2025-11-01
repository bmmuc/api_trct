"""API routes for anomaly detection service."""
from typing import Optional
from fastapi import APIRouter, Query, Depends
from src.models.schemas import TrainData, TrainResponse,\
    PredictData, PredictResponse, HealthCheckResponse
from src.services.anomaly_service import AnomalyDetectionService
from src.utils.metrics import MetricsTracker
from src.api.dependencies import get_anomaly_service, get_metrics_tracker

router = APIRouter()


@router.post("/fit/{series_id}", response_model=TrainResponse, tags=["Training"])
async def train_model(
    series_id: str,
    train_data: TrainData,
    anomaly_service: AnomalyDetectionService = Depends(get_anomaly_service)
) -> TrainResponse:
    """Train an anomaly detection model for a specific time series."""
    return anomaly_service.train_model(series_id, train_data)


@router.post("/predict/{series_id}", response_model=PredictResponse, tags=["Prediction"])
async def predict_anomaly(
    series_id: str,
    predict_data: PredictData,
    version: Optional[str] = Query(None),
    anomaly_service: AnomalyDetectionService = Depends(get_anomaly_service)
) -> PredictResponse:
    """Predict if a data point is anomalous for a specific time series."""
    return anomaly_service.predict_anomaly(series_id, predict_data.to_data_point(), version)


@router.get("/healthcheck", response_model=HealthCheckResponse, tags=["Health Check"])
async def healthcheck(
    anomaly_service: AnomalyDetectionService = Depends(get_anomaly_service),
    metrics_tracker: MetricsTracker = Depends(get_metrics_tracker)
) -> HealthCheckResponse:
    """Get system health and performance metrics."""
    # Get the number of trained series
    series_trained = anomaly_service.get_trained_series_count()

    # Get latency metrics
    inference_metrics = metrics_tracker.get_inference_metrics()
    training_metrics = metrics_tracker.get_training_metrics()

    return HealthCheckResponse(
        series_trained=series_trained,
        inference_latency_ms=inference_metrics,
        training_latency_ms=training_metrics
    )