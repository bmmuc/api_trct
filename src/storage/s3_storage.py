"""
Mock storage using S3 (AWS).
"""
from typing import Optional, List
from src.storage.base_storage import BaseModelStorage
from src.anomaly_models.base_model import BaseAnomalyModel
from src.utils.logger import logger


class S3ModelStorage(BaseModelStorage):
    """Mock of S3 storage."""

    def __init__(self, bucket_name: str, prefix: str = "models"):
        self.bucket_name = bucket_name
        self.prefix = prefix
        # Mock: self.s3_client = boto3.client('s3')

    def save_model(self, series_id: str, model: BaseAnomalyModel,
                   version: Optional[str] = None) -> str:
        """Simulates model upload to S3."""
        if not model.is_fitted():
            raise ValueError("Cannot save an unfitted model")

        if version is None:
            version = self._generate_version(series_id)

        # Mock: s3_key = f"{self.prefix}/{series_id}/{version}.json"
        # Mock: self.s3_client.put_object(
        #     Bucket=self.bucket_name,
        #     Key=s3_key,
        #     Body=json.dumps(model.to_dict())
        # )

        logger.info("Mock: uploaded model to S3: %s/%s", series_id, version)
        return version

    def load_model(self, series_id: str,
                   version: Optional[str] = None) -> tuple[BaseAnomalyModel, str]:
        """Simulates model download from S3."""
        # Mock: if version is None:
        #     version = self._get_latest_version_from_s3(series_id)
        #
        # Mock: s3_key = f"{self.prefix}/{series_id}/{version}.json"
        # Mock: response = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)
        # Mock: model_data = json.loads(response['Body'].read())
        # Mock: model = ModelFactory.from_dict(model_data)

        logger.info("Mock: downloaded model from S3: %s/%s", series_id, version)
        raise NotImplementedError("Mock S3 storage")

    def get_latest_version(self, series_id: str) -> Optional[str]:
        """Simulates fetching latest version from S3."""
        # Mock: list_objects_v2 with prefix and get the last one
        return None

    def list_versions(self, series_id: str) -> List[str]:
        """Simulates listing versions in S3."""
        # Mock: list_objects_v2 and parse keys
        return []

    def list_all_series(self) -> List[str]:
        """Simulates listing series in S3."""
        # Mock: list prefixes
        return []

    def model_exists(self, series_id: str, version: Optional[str] = None) -> bool:
        """Simulates existence check in S3."""
        # Mock: head_object
        return False

    def _generate_version(self, series_id: str) -> str:  # pylint: disable=unused-argument
        """Simulates version generation."""
        # Mock: list existing versions and increment
        return "v0"
