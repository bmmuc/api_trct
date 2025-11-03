"""
Factory to create different storage backends.
"""
from src.storage.base_storage import BaseModelStorage
from src.storage.filesystem_storage import FilesystemModelStorage
from src.storage.s3_storage import S3ModelStorage


class StorageFactory:
    """Creates storage based on configuration."""

    @classmethod
    def create(cls, storage_type: str, **kwargs) -> BaseModelStorage:
        """Instantiates storage by type."""
        if storage_type == "filesystem":
            return FilesystemModelStorage(**kwargs)
        elif storage_type == "s3":
            return S3ModelStorage(**kwargs)
        else:
            raise ValueError(f"Storage type '{storage_type}' not supported")
