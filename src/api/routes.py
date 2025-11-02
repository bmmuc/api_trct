"""API routes for anomaly detection service."""
from typing import Optional
from fastapi import APIRouter, Query, Depends, HTTPException
from fastapi.responses import Response
from src.models.schemas import (
    TrainData, TrainResponse, PredictData, PredictResponse,
    HealthCheckResponse, validate_series_id
)
from src.services.anomaly_service import AnomalyDetectionService
from src.services.visualization_service import VisualizationService
from src.utils.metrics import MetricsTracker
from src.utils.logger import logger
from src.api.dependencies import get_anomaly_service, get_metrics_tracker, get_visualization_service
from src.exceptions import (
    ModelNotFoundError, ValidationError, InvalidSeriesIdError,
    ModelNotFittedError, AnomalyDetectionError
)

router = APIRouter()


@router.post("/fit/{series_id}", response_model=TrainResponse, tags=["Training"])
async def train_model(
    series_id: str,
    train_data: TrainData,
    anomaly_service: AnomalyDetectionService = Depends(get_anomaly_service)
) -> TrainResponse:
    """Train an anomaly detection model for a specific time series."""
    try:
        # Validate series_id format
        validate_series_id(series_id)

        logger.info(
            f"Training request for series_id='{series_id}' with {len(train_data.values)} data points"
        )

        result = anomaly_service.train_model(series_id, train_data)

        logger.info(
            f"Training completed for series_id='{series_id}', version='{result.version}'"
        )

        return result

    except (ValidationError, InvalidSeriesIdError) as e:
        logger.warning(f"Validation error for series_id='{series_id}': {e.message}")
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except AnomalyDetectionError as e:
        logger.error(f"Error training model for series_id='{series_id}': {e.message}")
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.error(f"Unexpected error training model for series_id='{series_id}': {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during training")


@router.post("/predict/{series_id}", response_model=PredictResponse, tags=["Prediction"])
async def predict_anomaly(
    series_id: str,
    predict_data: PredictData,
    version: Optional[str] = Query(None),
    anomaly_service: AnomalyDetectionService = Depends(get_anomaly_service)
) -> PredictResponse:
    """Predict if a data point is anomalous for a specific time series."""
    try:
        # Validate series_id format
        validate_series_id(series_id)

        logger.debug(
            f"Prediction request for series_id='{series_id}', version='{version}', value={predict_data.value}"
        )

        result = anomaly_service.predict_anomaly(series_id, predict_data.to_data_point(), version)

        logger.debug(
            f"Prediction result for series_id='{series_id}': anomaly={result.anomaly}, "
            f"version='{result.model_version}'"
        )

        return result

    except ModelNotFoundError as e:
        logger.warning(f"Model not found: {e.message}")
        raise HTTPException(status_code=404, detail=e.message)
    except (ValidationError, InvalidSeriesIdError, ModelNotFittedError) as e:
        logger.warning(f"Validation error: {e.message}")
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except AnomalyDetectionError as e:
        logger.error(f"Error making prediction: {e.message}")
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.error(f"Unexpected error making prediction for series_id='{series_id}': {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during prediction")


@router.get("/healthcheck", response_model=HealthCheckResponse, tags=["Health Check"])
async def healthcheck(
    anomaly_service: AnomalyDetectionService = Depends(get_anomaly_service),
    metrics_tracker: MetricsTracker = Depends(get_metrics_tracker)
) -> HealthCheckResponse:
    """Get system health and performance metrics."""
    try:
        # Get the number of trained series
        series_trained = anomaly_service.get_trained_series_count()

        # Get latency metrics
        inference_metrics = metrics_tracker.get_inference_metrics()
        training_metrics = metrics_tracker.get_training_metrics()

        logger.debug(
            f"Health check: {series_trained} series trained, "
            f"avg inference latency: {inference_metrics.avg}ms"
        )

        return HealthCheckResponse(
            series_trained=series_trained,
            inference_latency_ms=inference_metrics,
            training_latency_ms=training_metrics
        )
    except Exception as e:
        logger.error(f"Error in healthcheck: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during health check")


@router.get("/plot/{series_id}", tags=["Visualization"])
async def plot_time_series(
    series_id: str,
    version: Optional[str] = Query(None, description="Model version to use for plotting"),
    format: str = Query("png", description="Image format (png, jpg, svg)"),
    visualization_service: VisualizationService = Depends(get_visualization_service)
) -> Response:
    """
    Generate a plot of the time series with anomaly detection boundaries.

    Returns a PNG image showing:
    - Mean line
    - Upper and lower bounds (mean ± 3σ)
    - Normal range shaded area
    - Model statistics
    """
    try:
        # Validate series_id format
        validate_series_id(series_id)

        # Validate format
        if format not in ['png', 'jpg', 'svg']:
            raise ValidationError(f"Unsupported format '{format}'. Use 'png', 'jpg', or 'svg'", field="format")

        logger.info(f"Generating plot for series_id='{series_id}', version='{version}', format='{format}'")

        # Generate plot
        image_bytes = visualization_service.plot_time_series(series_id, version, format)

        # Determine media type
        media_type_map = {
            'png': 'image/png',
            'jpg': 'image/jpeg',
            'svg': 'image/svg+xml'
        }

        return Response(
            content=image_bytes,
            media_type=media_type_map[format],
            headers={
                'Content-Disposition': f'inline; filename="timeseries_{series_id}_{version or "latest"}.{format}"'
            }
        )

    except ModelNotFoundError as e:
        logger.warning(f"Model not found for plot: {e.message}")
        raise HTTPException(status_code=404, detail=e.message)
    except (ValidationError, InvalidSeriesIdError) as e:
        logger.warning(f"Validation error in plot: {e.message}")
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except AnomalyDetectionError as e:
        logger.error(f"Error generating plot: {e.message}")
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.error(f"Unexpected error generating plot for series_id='{series_id}': {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during plot generation")