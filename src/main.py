"""
Time Series Anomaly Detection API
Main application entry point.
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError as PydanticValidationError
from src.api.routes import router
from src.api.exception_handlers import (
    anomaly_detection_error_handler,
    validation_error_handler,
    pydantic_validation_error_handler,
    general_exception_handler
)
from src.exceptions import AnomalyDetectionError, ValidationError
from src.utils.logger import logger


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001 pylint: disable=unused-argument, redefined-outer-name
    """
    Lifespan context manager for startup and shutdown events.

    Args:
        app: FastAPI application instance (required by FastAPI but not used here)
    """
    # Startup
    logger.info("Starting Time Series Anomaly Detection API")
    yield
    # Shutdown
    logger.info("Shutting down Time Series Anomaly Detection API")


# Initialize FastAPI app
app = FastAPI(
    title="Time Series Anomaly Detection API",
    description="API for training and inference of anomaly detection models on time series data",
    version="0.1.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register exception handlers
app.add_exception_handler(AnomalyDetectionError, anomaly_detection_error_handler)
app.add_exception_handler(ValidationError, validation_error_handler)
app.add_exception_handler(PydanticValidationError, pydantic_validation_error_handler)
app.add_exception_handler(Exception, general_exception_handler)


# Include routes
app.include_router(router)


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint returning API information."""
    return {
        "message": "Time Series Anomaly Detection API",
        "version": "0.1.0",
        "docs": "/docs",
        "openapi": "/openapi.json"
    }
