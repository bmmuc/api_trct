"""
Base interface for different storage backends.
"""
from abc import ABC, abstractmethod
from typing import Optional, List
from src.anomaly_models.base_model import BaseAnomalyModel


class BaseModelStorage(ABC):
    """Interface for model storage."""

    @abstractmethod
    def save_model(self, series_id: str, model: BaseAnomalyModel,
                   version: Optional[str] = None) -> str:
        """Persists the model and returns the version."""
        pass

    @abstractmethod
    def load_model(self, series_id: str,
                   version: Optional[str] = None) -> tuple[BaseAnomalyModel, str]:
        """Loads model, returns (model, version)."""
        pass

    @abstractmethod
    def get_latest_version(self, series_id: str) -> Optional[str]:
        """Gets the most recent version."""
        pass

    @abstractmethod
    def list_versions(self, series_id: str) -> List[str]:
        """Lists all versions of a series."""
        pass

    @abstractmethod
    def list_all_series(self) -> List[str]:
        """Lists all available series_id."""
        pass

    @abstractmethod
    def model_exists(self, series_id: str, version: Optional[str] = None) -> bool:
        """Checks if the model exists."""
        pass
