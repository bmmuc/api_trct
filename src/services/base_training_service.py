"""
Abstract base class for training services.
"""
from abc import ABC, abstractmethod
from src.models.schemas import TrainData, TrainResponse


class BaseTrainingService(ABC):  # pylint: disable=too-few-public-methods
    """Base class for training services."""

    @abstractmethod
    def train(self, series_id: str, train_data: TrainData, metadata: dict = None) -> TrainResponse:
        """
        Train a model for the given series.

        Args:
            series_id: Identifier for the time series
            train_data: Training data
            metadata: Optional metadata to pass to the training service

        Returns:
            TrainResponse with training results
        """
