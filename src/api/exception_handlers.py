"""
Exception handlers for the FastAPI application.
"""
from fastapi import Request, status
from fastapi.responses import JSONResponse
from pydantic import ValidationError as PydanticValidationError
from src.exceptions import AnomalyDetectionError, ValidationError
from src.utils.logger import logger


async def anomaly_detection_error_handler(
    request: Request,  # noqa: ARG001 pylint: disable=unused-argument
    exc: AnomalyDetectionError
):
    """Handle custom anomaly detection errors."""
    logger.error("AnomalyDetectionError: %s (status: %d)", exc.message, exc.status_code)
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.message}
    )


async def validation_error_handler(
    request: Request,  # noqa: ARG001 pylint: disable=unused-argument
    exc: ValidationError
):
    """Handle validation errors."""
    logger.warning("ValidationError: %s", exc.message)
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.message, "field": exc.field}
    )


async def pydantic_validation_error_handler(
    request: Request,  # noqa: ARG001 pylint: disable=unused-argument
    exc: PydanticValidationError
):
    """Handle Pydantic validation errors."""
    logger.warning("Pydantic validation error: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": exc.errors()}
    )


async def general_exception_handler(
    request: Request,  # noqa: ARG001 pylint: disable=unused-argument
    exc: Exception
):
    """Catch-all handler for unexpected errors."""
    logger.error("Unhandled exception: %s", str(exc), exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"}
    )
