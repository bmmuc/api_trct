"""
Pydantic models for Time Series data structures and API requests/responses.
"""
from typing import Sequence, List
import re
import math
import numpy as np
from pydantic import BaseModel, Field, field_validator
from src.exceptions import ValidationError, InvalidSeriesIdError


def validate_series_id(series_id: str) -> None:
    """
    Validate series_id format to prevent path injection attacks.

    Args:
        series_id: The series identifier to validate

    Raises:
        InvalidSeriesIdError: If series_id contains invalid characters
    """
    if not series_id:
        raise InvalidSeriesIdError(series_id, "series_id cannot be empty")

    if '..' in series_id or '/' in series_id or '\\' in series_id:
        raise InvalidSeriesIdError(
            series_id,
            "series_id cannot contain path traversal characters"
        )

    if not re.match(r'^[a-zA-Z0-9_\-\.]+$', series_id):
        raise InvalidSeriesIdError(
            series_id,
            "series_id can only contain alphanumeric characters, underscores, hyphens, and dots"
        )

    if len(series_id) > 100:
        raise InvalidSeriesIdError(
            series_id,
            f"series_id too long (max 100 characters, got {len(series_id)})"
        )


class DataPoint(BaseModel):
    """Represents a single data point in a time series."""
    timestamp: int = Field(
        ..., description="Unix timestamp of the time the data point was collected"
    )
    value: float = Field(
        ..., description="Value of the time series measured at time `timestamp`"
    )


class TimeSeries(BaseModel):
    """Represents a complete time series with multiple data points."""
    data: Sequence[DataPoint] = Field(
        ...,
        description=(
            "List of datapoints, ordered in time, "
            "of subsequent measurements of some quantity"
        )
    )


class TrainData(BaseModel):
    """Request model for training endpoint."""
    timestamps: List[int] = Field(
        ..., description="Timestamp values should be in the unix timestamp format"
    )
    values: List[float] = Field(
        ..., description="Values corresponding to each timestamp"
    )

    def model_post_init(self, __context):  # noqa: ARG002
        """Validate that timestamps and values have the same length."""
        if len(self.timestamps) != len(self.values):
            raise ValidationError(
                f"Timestamps ({len(self.timestamps)}) and values "
                f"({len(self.values)}) must have the same length"
            )

    @field_validator('timestamps')
    @classmethod
    def validate_timestamps(cls, v):
        """Validate timestamp array."""
        if len(v) == 0:
            raise ValidationError("Timestamps list cannot be empty", field="timestamps")

        if len(v) < 3:
            raise ValidationError(
                f"Minimum 3 data points required for training, got {len(v)}",
                field="timestamps"
            )

        # Check for timestamp ordering
        if not all(v[i] <= v[i+1] for i in range(len(v)-1)):
            raise ValidationError(
                "Timestamps must be in ascending order",
                field="timestamps"
            )

        return v

    @field_validator('values')
    @classmethod
    def validate_values(cls, v):
        """Validate values array."""
        if len(v) == 0:
            raise ValidationError("Values list cannot be empty", field="values")

        if len(v) < 3:
            raise ValidationError(
                f"Minimum 3 data points required for training, got {len(v)}",
                field="values"
            )

        # Check for NaN or Inf values
        for i, val in enumerate(v):
            if math.isnan(val):
                raise ValidationError(
                    f"NaN value detected at index {i}",
                    field="values"
                )
            if math.isinf(val):
                raise ValidationError(
                    f"Infinite value detected at index {i}",
                    field="values"
                )

        # Check for constant values (std = 0)
        if len(v) >= 3 and np.std(v) == 0:
            raise ValidationError(
                "Cannot train on constant values (standard deviation is 0)",
                field="values"
            )

        return v

    def to_time_series(self) -> TimeSeries:
        """Convert TrainData to TimeSeries object."""
        if len(self.timestamps) != len(self.values):
            raise ValidationError(
                f"Timestamps ({len(self.timestamps)}) and values "
                f"({len(self.values)}) must have the same length"
            )

        data_points = [
            DataPoint(timestamp=ts, value=val)
            for ts, val in zip(self.timestamps, self.values)
        ]
        return TimeSeries(data=data_points)


class TrainDataExternal(TrainData):
    metadata: dict = Field(
        default_factory=dict,
        description="Optional metadata to pass to the training service (useful for external APIs)"
    )


class TrainResponse(BaseModel):
    """Response model for training endpoint."""
    series_id: str = Field(..., description="Identifier for the trained series")
    version: str = Field(..., description="Version identifier for the trained model")
    points_used: int = Field(..., description="Number of data points used for training")


# Prediction API Models
class PredictData(BaseModel):
    """Request model for prediction endpoint."""
    timestamp: str = Field(..., description="Timestamp of the data point")
    value: float = Field(..., description="Value to check for anomaly")

    def to_data_point(self) -> DataPoint:
        """Convert PredictData to DataPoint object."""
        # Handle both string and int timestamps
        try:
            ts = int(self.timestamp)
        except ValueError as exc:
            raise ValueError(
                "Timestamp must be a valid integer or numeric string"
            ) from exc
        return DataPoint(timestamp=ts, value=self.value)


class PredictResponse(BaseModel):
    """Response model for prediction endpoint."""
    model_config = {"protected_namespaces": ()}

    anomaly: bool = Field(..., description="Whether the data point is anomalous")
    model_version: str = Field(..., description="Version of the model used for prediction")


# Health Check API Models
class Metrics(BaseModel):
    """Metrics for performance monitoring."""
    avg: float | None = None
    p95: float | None = None


class HealthCheckResponse(BaseModel):
    """Response model for health check endpoint."""
    series_trained: int = Field(..., description="Number of series with trained models")
    inference_latency_ms: Metrics = Field(..., description="Inference latency metrics")
    training_latency_ms: Metrics = Field(..., description="Training latency metrics")
