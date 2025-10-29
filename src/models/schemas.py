"""
Pydantic models for Time Series data structures and API requests/responses.
"""
from typing import Sequence, List
from pydantic import BaseModel, Field, field_validator


class DataPoint(BaseModel):
    """Represents a single data point in a time series."""
    timestamp: int = Field(..., description="Unix timestamp of the time the data point was collected")
    value: float = Field(..., description="Value of the time series measured at time `timestamp`")


class TimeSeries(BaseModel):
    """Represents a complete time series with multiple data points."""
    data: Sequence[DataPoint] = Field(
        ...,
        description="List of datapoints, ordered in time, of subsequent measurements of some quantity"
    )



class TrainData(BaseModel):
    """Request model for training endpoint."""
    timestamps: List[int] = Field(..., description="Timestamp values should be in the unix timestamp format")
    values: List[float] = Field(..., description="Values corresponding to each timestamp")

    @field_validator('timestamps', 'values')
    @classmethod
    def check_not_empty(cls, v):
        if len(v) == 0:
            raise ValueError("List cannot be empty")
        return v

    def to_time_series(self) -> TimeSeries:
        """Convert TrainData to TimeSeries object."""
        if len(self.timestamps) != len(self.values):
            raise ValueError("Timestamps and values must have the same length")

        data_points = [
            DataPoint(timestamp=ts, value=val)
            for ts, val in zip(self.timestamps, self.values)
        ]
        return TimeSeries(data=data_points)


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
        except ValueError:
            raise ValueError("Timestamp must be a valid integer or numeric string")
        return DataPoint(timestamp=ts, value=self.value)

class PredictResponse(BaseModel):
    """Response model for prediction endpoint."""
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
