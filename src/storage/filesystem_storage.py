"""
Model storage in local filesystem (original implementation).
"""
import json
import os
import tempfile
from typing import Optional, List
from pathlib import Path
from datetime import datetime
from filelock import FileLock
from src.storage.base_storage import BaseModelStorage
from src.anomaly_models.base_model import BaseAnomalyModel
from src.anomaly_models.model_factory import ModelFactory
from src.utils.logger import logger


class FilesystemModelStorage(BaseModelStorage):
    """Stores models in JSON on local disk."""

    def __init__(self, storage_path: str = "./model_storage"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.locks_path = self.storage_path / ".locks"
        self.locks_path.mkdir(parents=True, exist_ok=True)

    def _get_series_dir(self, series_id: str) -> Path:
        series_dir = self.storage_path / series_id
        series_dir.mkdir(parents=True, exist_ok=True)
        return series_dir

    def _get_lock_path(self, series_id: str) -> Path:
        return self.locks_path / f"{series_id}.lock"

    def _generate_version(self, series_id: str) -> str:
        versions = self._list_versions(series_id)
        if not versions:
            return "v0"

        latest_version = versions[-1]
        version_number = int(latest_version.lstrip('v'))
        return f"v{version_number + 1}"

    def _get_model_path(self, series_id: str, version: str) -> Path:
        return self._get_series_dir(series_id) / f"{version}.bin"

    def _get_metadata_path(self, series_id: str, version: str) -> Path:
        return self._get_series_dir(series_id) / f"{version}.meta.json"

    def _atomic_write_bytes(self, target_path: Path, data: bytes) -> None:
        temp_fd, temp_path = tempfile.mkstemp(
            dir=target_path.parent,
            suffix='.tmp',
            prefix='.model_'
        )

        try:
            with os.fdopen(temp_fd, 'wb') as f:
                f.write(data)
            os.replace(temp_path, target_path)
        except Exception:
            try:
                os.unlink(temp_path)
            except OSError:
                pass
            raise

    def _atomic_write_json(self, target_path: Path, data: dict) -> None:
        temp_fd, temp_path = tempfile.mkstemp(
            dir=target_path.parent,
            suffix='.tmp',
            prefix='.meta_'
        )

        try:
            with os.fdopen(temp_fd, 'w') as f:
                json.dump(data, f, indent=2)
            os.replace(temp_path, target_path)
        except Exception:
            try:
                os.unlink(temp_path)
            except OSError:
                pass
            raise

    def save_model(self, series_id: str, model: BaseAnomalyModel,
                   version: Optional[str] = None) -> str:
        if not model.is_fitted():
            raise ValueError("Cannot save an unfitted model")

        lock_path = self._get_lock_path(series_id)
        lock = FileLock(lock_path, timeout=10)

        with lock:
            if version is None:
                version = self._generate_version(series_id)

            model_path = self._get_model_path(series_id, version)
            metadata_path = self._get_metadata_path(series_id, version)

            # Save model as bytes
            model_bytes = model.save()
            self._atomic_write_bytes(model_path, model_bytes)

            # Save metadata as JSON
            metadata = {
                "series_id": series_id,
                "version": version,
                "model_type": model.get_model_type(),
                "saved_at": datetime.utcnow().isoformat()
            }
            self._atomic_write_json(metadata_path, metadata)

            logger.info(
                "Saved model: series_id='%s', version='%s', type='%s' to %s",
                series_id, version, model.get_model_type(), model_path
            )

        return version

    def load_model(self, series_id: str,
                   version: Optional[str] = None) -> tuple[BaseAnomalyModel, str]:
        lock_path = self._get_lock_path(series_id)
        lock = FileLock(lock_path, timeout=10)

        with lock:
            if version is None:
                versions = self._list_versions(series_id)
                version = versions[-1] if versions else None
                if version is None:
                    logger.warning("No models found for series_id: %s", series_id)
                    raise FileNotFoundError(
                        f"No models found for series_id: {series_id}"
                    )

            model_path = self._get_model_path(series_id, version)
            metadata_path = self._get_metadata_path(series_id, version)

            try:
                # Load metadata
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)

                # Load model bytes
                with open(model_path, 'rb') as f:
                    model_bytes = f.read()

                logger.debug(
                    "Loaded model: series_id='%s', version='%s', type='%s' from %s",
                    series_id, version, metadata["model_type"], model_path
                )
            except FileNotFoundError as exc:
                logger.warning("Model file not found: %s", model_path)
                raise FileNotFoundError(
                    f"Model not found: {series_id}/{version}"
                ) from exc
            except json.JSONDecodeError as e:
                logger.error("Corrupted metadata file: %s", metadata_path)
                raise ValueError(
                    f"Corrupted metadata: {series_id}/{version}"
                ) from e

            # Create empty model and load data
            model = ModelFactory.create(metadata["model_type"])
            model.load(model_bytes)

        return model, version

    def get_latest_version(self, series_id: str) -> Optional[str]:
        versions = self.list_versions(series_id)
        return versions[-1] if versions else None

    def _list_versions(self, series_id: str) -> List[str]:
        series_dir = self._get_series_dir(series_id)

        if not series_dir.exists():
            return []

        versions = []

        # Look for .bin files (models)
        for file_path in series_dir.glob("*.bin"):
            name = file_path.stem
            if not name:
                continue

            if name.startswith("v") and name[1:].isdigit():
                versions.append(name)

        return sorted(versions, key=lambda v: int(v.lstrip('v')))

    def list_versions(self, series_id: str) -> List[str]:
        lock_path = self._get_lock_path(series_id)
        lock = FileLock(lock_path, timeout=10)

        with lock:
            return self._list_versions(series_id)

    def list_all_series(self) -> List[str]:
        if not self.storage_path.exists():
            return []

        series_dirs: List[str] = []
        for entry in self.storage_path.iterdir():
            if entry.name == ".locks":
                continue

            if not entry.is_dir():
                continue

            if any(entry.glob("*.bin")):
                series_dirs.append(entry.name)

        return series_dirs

    def model_exists(self, series_id: str, version: Optional[str] = None) -> bool:
        try:
            lock_path = self._get_lock_path(series_id)
            lock = FileLock(lock_path, timeout=10)

            with lock:
                if version is None:
                    versions = self._list_versions(series_id)
                    version = versions[-1] if versions else None
                    if version is None:
                        return False

                model_path = self._get_model_path(series_id, version)
                return model_path.exists()
        except (OSError, IOError, ValueError):
            return False
