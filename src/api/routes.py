"""API routes for anomaly detection service."""
from typing import Optional
from fastapi import APIRouter, Query
from models.schemas import TrainData, TrainResponse,\
    PredictData, PredictResponse, HealthCheckResponse
from services.anomaly_service import AnomalyDetectionService
from storage.model_store import ModelStore

router = APIRouter()

model_store = ModelStore()
anomaly_service = AnomalyDetectionService(model_store)


@router.post("/fit/{series_id}", response_model=TrainResponse, tags=["Training"])
async def train_model(series_id: str,
                      train_data: TrainData) -> TrainResponse:
    """Train an anomaly detection model for a specific time series."""
    return anomaly_service.train_model(series_id, train_data)


@router.post("/predict/{series_id}", response_model=PredictResponse, tags=["Prediction"])
async def predict_anomaly(series_id: str, predict_data: PredictData,
                          version: Optional[str] = Query(None)) -> PredictResponse:
    """Predict if a data point is anomalous for a specific time series."""
    return anomaly_service.predict_anomaly(series_id, predict_data.to_data_point(), version)


@router.get("/healthcheck", response_model=HealthCheckResponse, tags=["Health Check"])
async def healthcheck() -> HealthCheckResponse:
    """Get system health and performance metrics."""
    pass