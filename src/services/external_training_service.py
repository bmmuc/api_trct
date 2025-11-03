"""
External API training service implementation.
"""
import time
from typing import Optional
import requests
from src.services.base_training_service import BaseTrainingService
from src.models.schemas import TrainData, TrainResponse
from src.utils.base_metrics import BaseMetricsExporter
from src.utils.logger import logger
from src.exceptions import AnomalyDetectionError


class ExternalTrainingService(BaseTrainingService):  # pylint: disable=too-few-public-methods
    """Training service that delegates training to an external API."""

    def __init__(self,
                 api_url: str,
                 metrics_exporter: BaseMetricsExporter,
                 api_key: Optional[str] = None,
                 timeout: int = 30):
        """
        Initialize external training service.

        Args:
            api_url: Base URL of the external training API
            metrics_exporter: Metrics exporter instance
            api_key: Optional API key for authentication
            timeout: Request timeout in seconds
        """
        self.api_url = api_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        self.metrics_exporter = metrics_exporter

    def train(self, series_id: str, train_data: TrainData, metadata: dict = None) -> TrainResponse:
        """
        Sends training data to external API.

        Args:
            series_id: Identifier for the time series
            train_data: Training data
            metadata: Optional metadata to pass to the training API

        Returns:
            TrainResponse with training results from external API
        """
        start_time = time.time()

        # Prepare payload
        payload = {
            "series_id": series_id,
            "timestamps": train_data.timestamps,
            "values": train_data.values,
            "metadata": metadata or {}
        }

        # Prepare headers
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            logger.info(
                "Sending training request to external API for series_id='%s'",
                series_id
            )

            # Send request to external API
            # for the future, the request has to be async
            # and use webhooks or polling to get the result.
            response = requests.post(
                f"{self.api_url}/train",
                json=payload,
                headers=headers,
                timeout=self.timeout
            )
            response.raise_for_status()

            result = response.json()

            latency_ms = (time.time() - start_time) * 1000
            self.metrics_exporter.record_training_latency(latency_ms)

            logger.info(
                "External training completed for series_id='%s', version='%s'",
                series_id, result.get("version")
            )

            # Parse response into TrainResponse
            return TrainResponse(
                series_id=result.get("series_id", series_id),
                version=result.get("version", "external-unknown"),
                points_used=result.get("points_used", len(train_data.values))
            )

        except requests.exceptions.Timeout as e:
            logger.error(
                "Timeout calling external training API for series_id='%s'",
                series_id
            )
            raise AnomalyDetectionError(
                f"External training API timeout for series {series_id}",
                status_code=504
            ) from e

        except requests.exceptions.HTTPError as e:
            logger.error(
                "HTTP error from external training API for series_id='%s': %s",
                series_id, str(e)
            )
            raise AnomalyDetectionError(
                f"External training API error: {e.response.text}",
                status_code=e.response.status_code
            ) from e

        except requests.exceptions.RequestException as e:
            logger.error(
                "Request error calling external training API for series_id='%s': %s",
                series_id, str(e)
            )
            raise AnomalyDetectionError(
                f"Failed to connect to external training API: {str(e)}",
                status_code=502
            ) from e

        except Exception as e:
            logger.error(
                "Unexpected error calling external training API for series_id='%s': %s",
                series_id, str(e), exc_info=True
            )
            raise AnomalyDetectionError(
                f"Unexpected error during external training: {str(e)}",
                status_code=500
            ) from e
