"""
Custom exceptions for the Anomaly Detection API.

This module defines domain-specific exceptions that provide better
error handling and HTTP status code mapping.
"""


class AnomalyDetectionError(Exception):
    """Base exception for all anomaly detection errors."""

    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


class ModelNotFoundError(AnomalyDetectionError):
    """Raised when a requested model does not exist."""

    def __init__(self, series_id: str, version: str | None = None):
        self.series_id = series_id
        self.version = version
        if version:
            message = f"Model for series '{series_id}' version '{version}' not found"
        else:
            message = f"No model found for series '{series_id}'"
        super().__init__(message, status_code=404)


class ModelNotFittedError(AnomalyDetectionError):
    """Raised when trying to use a model that hasn't been trained."""

    def __init__(self, series_id: str | None = None):
        message = "Model must be fitted before making predictions"
        if series_id:
            message = f"Model for series '{series_id}' must be trained before predictions"
        super().__init__(message, status_code=400)


class ValidationError(AnomalyDetectionError):
    """Raised when input data fails validation."""

    def __init__(self, message: str, field: str | None = None):
        self.field = field
        if field:
            message = f"Validation error in '{field}': {message}"
        super().__init__(message, status_code=422)


class InvalidSeriesIdError(AnomalyDetectionError):
    """Raised when series_id has invalid format or characters."""

    def __init__(self, series_id: str, reason: str = "contains invalid characters"):
        self.series_id = series_id
        message = f"Invalid series_id '{series_id}': {reason}"
        super().__init__(message, status_code=400)
