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
    pass

class TrainResponse(BaseModel):
    """Response model for training endpoint."""
    pass

# Prediction API Models
class PredictData(BaseModel):
    """Request model for prediction endpoint."""
    pass

class PredictResponse(BaseModel):
    """Response model for prediction endpoint."""
    pass

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
