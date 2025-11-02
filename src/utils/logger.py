"""
Structured logging configuration for the Anomaly Detection API.

Provides a centralized logging setup with configurable levels and formatting.
"""
import logging
import sys
import os
from typing import Optional


def setup_logger(
    name: str = "anomaly_detection",
    level: Optional[str] = None
) -> logging.Logger:
    """
    Configure and return a logger with structured formatting.

    Args:
        name: Logger name
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
               If None, reads from LOG_LEVEL env var, defaults to INFO.

    Returns:
        Configured logger instance
    """
    # Get log level from environment or use provided level
    if level is None:
        level = os.getenv("LOG_LEVEL", "INFO").upper()

    # Create logger
    log_instance = logging.getLogger(name)
    log_instance.setLevel(getattr(logging, level, logging.INFO))

    # Avoid adding handlers multiple times
    if log_instance.handlers:
        return log_instance

    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, level, logging.INFO))

    # Create formatter with structured output
    formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)

    # Add handler to logger
    log_instance.addHandler(handler)

    return log_instance


# Create default logger instance
logger = setup_logger()
