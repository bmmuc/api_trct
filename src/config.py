"""
Centralized configuration of system backends.
"""
from pydantic import BaseModel
from pydantic_settings import BaseSettings


class StatisticalModelConfig(BaseModel):
    """Config for statistical model."""
    threshold: float = 3.0


class SklearnModelConfig(BaseModel):
    """Config for sklearn models."""
    contamination: float = 0.1
    n_estimators: int = 100


class FilesystemStorageConfig(BaseModel):
    """Config for filesystem storage."""
    storage_path: str = "./model_storage"


class S3StorageConfig(BaseModel):
    """Config for S3 storage."""
    bucket: str = ""
    prefix: str = "models"


class MemoryMetricsConfig(BaseModel):
    """Config for in-memory metrics."""
    max_samples: int = 10000


class PrometheusMetricsConfig(BaseModel):
    """Config for Prometheus metrics."""
    namespace: str = "anomaly_detection"


class ExternalTrainingConfig(BaseModel):
    """Config for external training API."""
    api_url: str = ""
    api_key: str = ""
    timeout: int = 30


class AppConfig(BaseSettings):
    """Application settings."""

    # Backend type
    model_type: str = "statistical"
    storage_type: str = "filesystem"
    metrics_type: str = "memory"
    training_type: str = "local"  # "local" or "external"

    # Model configs
    statistical: StatisticalModelConfig = StatisticalModelConfig()
    sklearn: SklearnModelConfig = SklearnModelConfig()

    # Storage configs
    filesystem: FilesystemStorageConfig = FilesystemStorageConfig()
    s3: S3StorageConfig = S3StorageConfig()

    # Metrics configs
    memory: MemoryMetricsConfig = MemoryMetricsConfig()
    prometheus: PrometheusMetricsConfig = PrometheusMetricsConfig()

    # Training configs
    external_training: ExternalTrainingConfig = ExternalTrainingConfig()

    class Config:
        env_prefix = "APP_"
        case_sensitive = False
        env_nested_delimiter = "__"


# Global instance
config = AppConfig()
