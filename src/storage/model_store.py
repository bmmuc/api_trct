"""
Model persistence and versioning layer.
"""
import json
import os
from typing import Dict, Optional, List
from pathlib import Path
from datetime import datetime
from src.models.anomaly_model import AnomalyDetectionModel


class ModelStore:
    """
    Handles persistence and versioning of trained models.
    Each series_id can have multiple versions.
    """

    def __init__(self, storage_path: str = "./model_storage"):
        """
        Initialize the model store.

        Args:
            storage_path: Directory path where models will be stored
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

    def _get_series_dir(self, series_id: str) -> Path:
        """Get directory path for a specific series."""
        series_dir = self.storage_path / series_id
        series_dir.mkdir(parents=True, exist_ok=True)
        return series_dir

    def _generate_version(self, series_id: str) -> str:
        """Generate a version identifier based on an incrementing number."""
        versions = self.list_versions(series_id)
        if not versions:
            return "v0"

        latest_version = versions[-1]
        version_number = int(latest_version.lstrip('v'))
        return f"v{version_number + 1}"

    def _get_model_path(self, series_id: str, version: str) -> Path:
        """Get file path for a specific model version."""
        return self._get_series_dir(series_id) / f"{version}.json"

    def save_model(self, series_id: str,
                   model: AnomalyDetectionModel,
                   version: Optional[str] = None) -> str:
        """
        Save a trained model with versioning.

        Args:
            series_id: Identifier for the time series
            model: Trained AnomalyDetectionModel instance
            version: Optional version string, auto-generated if not provided

        Returns:
            Version identifier of the saved model
        """
        if not model._is_fitted:
            raise ValueError("Cannot save an unfitted model")

        if version is None:
            version = self._generate_version(series_id)

        model_path = self._get_model_path(series_id, version)

        model_data = {
            "series_id": series_id,
            "version": version,
            "model_params": model.to_dict(),
            "saved_at": datetime.utcnow().isoformat()
        }

        with open(model_path, 'w') as f:
            json.dump(model_data, f, indent=2)

        return version

    def load_model(self, series_id: str,
                   version: Optional[str] = None) -> tuple[AnomalyDetectionModel, str]:
        """
        Load a trained model.

        Args:
            series_id: Identifier for the time series
            version: Optional version string, loads latest if not provided

        Returns:
            Tuple of (model, version)
        """
        if version is None:
            version = self.get_latest_version(series_id)
            if version is None:
                raise FileNotFoundError(f"No models found for series_id: {series_id}")
        model_path = self._get_model_path(series_id, version)

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {series_id}/{version}")

        with open(model_path, 'r') as f:
            model_data = json.load(f)
        model = AnomalyDetectionModel()
        model = model.from_dict(model_data["model_params"])

        return model, version

    def get_latest_version(self, series_id: str) -> Optional[str]:
        """
        Get the latest version for a series_id.

        Args:
            series_id: Identifier for the time series

        Returns:
            Latest version string or None if no versions exist
        """
        versions = self.list_versions(series_id)
        return versions[-1] if versions else None

    def list_versions(self, series_id: str) -> List[str]:
        """
        List all available versions for a series_id.

        Args:
            series_id: Identifier for the time series

        Returns:
            Sorted list of version strings
        """
        series_dir = self._get_series_dir(series_id)

        if not series_dir.exists():
            return []

        versions = []

        for file_path in series_dir.glob("*.json"):
            name = file_path.stem
            if not name:
                continue

            if name.startswith("v") and name[1:].isdigit():
                versions.append(name)

        return sorted(versions, key=lambda v: int(v.lstrip('v')))

    def list_all_series(self) -> List[str]:
        """
        List all series_ids that have trained models.

        Returns:
            List of series_id strings
        """
        if not self.storage_path.exists():
            return []

        series_dirs: List[str] = []
        for entry in self.storage_path.iterdir():
            if not entry.is_dir():
                continue
            if any(entry.glob("*.json")):
                series_dirs.append(entry.name)
        return series_dirs

    def model_exists(self, series_id: str, version: Optional[str] = None) -> bool:
        """
        Check if a model exists.

        Args:
            series_id: Identifier for the time series
            version: Optional version string, checks latest if not provided

        Returns:
            True if model exists, False otherwise
        """
        try:
            if version is None:
                version = self.get_latest_version(series_id)
                if version is None:
                    return False

            model_path = self._get_model_path(series_id, version)
            return model_path.exists()
        except Exception:
            return False
