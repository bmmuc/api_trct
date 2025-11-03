"""API routes for anomaly detection service."""
from typing import Optional, Union
from fastapi import APIRouter, Query, Depends, HTTPException
from fastapi.responses import Response
from src.models.schemas import (
    TrainData, TrainDataExternal, TrainResponse, PredictData, PredictResponse,
    HealthCheckResponse, validate_series_id
)
from src.services.anomaly_service import AnomalyDetectionService
from src.services.visualization_service import VisualizationService
from src.utils.base_metrics import BaseMetricsExporter
from src.utils.logger import logger
from src.api.dependencies import (
    get_anomaly_service, get_metrics_exporter, get_visualization_service
)
from src.exceptions import (
    ModelNotFoundError, ValidationError, InvalidSeriesIdError,
    ModelNotFittedError, AnomalyDetectionError
)

router = APIRouter()


@router.post("/fit/{series_id}", response_model=TrainResponse, tags=["Training"])
async def train_model(
    series_id: str,
    train_data: Union[TrainData, TrainDataExternal],
    anomaly_service: AnomalyDetectionService = Depends(get_anomaly_service)
) -> TrainResponse:
    """
    Train an anomaly detection model for a specific time series.

    Args:
        series_id: Identifier for the time series
        train_data: Training data with timestamps, values, and optional metadata
        anomaly_service: Injected anomaly detection service
    """
    try:
        # Validate series_id format
        validate_series_id(series_id)

        logger.info(
            "Training request for series_id='%s' with %d data points",
            series_id, len(train_data.values)
        )
        if not hasattr(train_data, 'metadata'):
            metadata = None
        else:
            metadata = train_data.metadata

        result = anomaly_service.train_model(series_id, train_data, metadata)

        logger.info(
            "Training completed for series_id='%s', version='%s'",
            series_id, result.version
        )

        return result

    except (ValidationError, InvalidSeriesIdError) as e:
        logger.warning("Validation error for series_id='%s': %s", series_id, e.message)
        raise HTTPException(status_code=e.status_code, detail=e.message) from e
    except AnomalyDetectionError as e:
        logger.error("Error training model for series_id='%s': %s", series_id, e.message)
        raise HTTPException(status_code=e.status_code, detail=e.message) from e
    except Exception as e:
        logger.error(
            "Unexpected error training model for series_id='%s': %s",
            series_id, str(e), exc_info=True
        )
        raise HTTPException(
            status_code=500, detail="Internal server error during training"
        ) from e


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
            "Prediction request for series_id='%s', version='%s', value=%s",
            series_id, version, predict_data.value
        )

        result = anomaly_service.predict_anomaly(
            series_id, predict_data.to_data_point(), version
        )

        logger.debug(
            "Prediction result for series_id='%s': anomaly=%s, version='%s'",
            series_id, result.anomaly, result.model_version
        )

        return result

    except ModelNotFoundError as e:
        logger.warning("Model not found: %s", e.message)
        raise HTTPException(status_code=404, detail=e.message) from e
    except (ValidationError, InvalidSeriesIdError, ModelNotFittedError) as e:
        logger.warning("Validation error: %s", e.message)
        raise HTTPException(status_code=e.status_code, detail=e.message) from e
    except AnomalyDetectionError as e:
        logger.error("Error making prediction: %s", e.message)
        raise HTTPException(status_code=e.status_code, detail=e.message) from e
    except Exception as e:
        logger.error(
            "Unexpected error making prediction for series_id='%s': %s",
            series_id, str(e), exc_info=True
        )
        raise HTTPException(
            status_code=500, detail="Internal server error during prediction"
        ) from e


@router.get("/healthcheck", response_model=HealthCheckResponse, tags=["Health Check"])
async def healthcheck(
    anomaly_service: AnomalyDetectionService = Depends(get_anomaly_service),
    metrics_exporter: BaseMetricsExporter = Depends(get_metrics_exporter)
) -> HealthCheckResponse:
    """Get system health and performance metrics."""
    try:
        # Get the number of trained series
        series_trained = anomaly_service.get_trained_series_count()

        # Get latency metrics
        inference_metrics = metrics_exporter.get_inference_metrics()
        training_metrics = metrics_exporter.get_training_metrics()

        logger.debug(
            "Health check: %d series trained, avg inference latency: %sms",
            series_trained, inference_metrics.avg
        )

        return HealthCheckResponse(
            series_trained=series_trained,
            inference_latency_ms=inference_metrics,
            training_latency_ms=training_metrics
        )
    except Exception as e:
        logger.error("Error in healthcheck: %s", str(e), exc_info=True)
        raise HTTPException(
            status_code=500, detail="Internal server error during health check"
        ) from e


@router.get("/plot/{series_id}", tags=["Visualization"])
async def plot_time_series(
    series_id: str,
    version: Optional[str] = Query(None, description="Model version to use for plotting"),
    img_format: str = Query("png", description="Image format (png, jpg, svg)", alias="format"),
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
        if img_format not in ['png', 'jpg', 'svg']:
            raise ValidationError(
                f"Unsupported format '{img_format}'. Use 'png', 'jpg', or 'svg'",
                field="format"
            )

        logger.info(
            "Generating plot for series_id='%s', version='%s', format='%s'",
            series_id, version, img_format
        )

        # Generate plot
        image_bytes = visualization_service.plot_time_series(series_id, version, img_format)

        # Determine media type
        media_type_map = {
            'png': 'image/png',
            'jpg': 'image/jpeg',
            'svg': 'image/svg+xml'
        }

        filename = f'timeseries_{series_id}_{version or "latest"}.{img_format}'
        return Response(
            content=image_bytes,
            media_type=media_type_map[img_format],
            headers={'Content-Disposition': f'inline; filename="{filename}"'}
        )

    except ModelNotFoundError as e:
        logger.warning("Model not found for plot: %s", e.message)
        raise HTTPException(status_code=404, detail=e.message) from e
    except (ValidationError, InvalidSeriesIdError) as e:
        logger.warning("Validation error in plot: %s", e.message)
        raise HTTPException(status_code=e.status_code, detail=e.message) from e
    except AnomalyDetectionError as e:
        logger.error("Error generating plot: %s", e.message)
        raise HTTPException(status_code=e.status_code, detail=e.message) from e
    except Exception as e:
        logger.error(
            "Unexpected error generating plot for series_id='%s': %s",
            series_id, str(e), exc_info=True
        )
        raise HTTPException(
            status_code=500, detail="Internal server error during plot generation"
        ) from e
