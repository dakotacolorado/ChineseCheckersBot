import boto3
from abc import ABC, abstractmethod
from typing import Generic, TypeVar, List, Type
from pathlib import PurePosixPath
from .ICatalog import ICatalog
from .IMetadata import IMetadata
from .IDataMetadata import IDataMetadata

M = TypeVar('M', bound=IMetadata)
DM = TypeVar('DM', bound=IDataMetadata)

class S3DataCatalog(ABC, ICatalog, Generic[M, DM]):
    def __init__(self, s3_client=None, s3_prefix: str = ''):
        self.s3 = s3_client if s3_client else boto3.client('s3')
        self.s3_prefix = s3_prefix.rstrip('/')

    @abstractmethod
    def _bucket_name(self) -> str:
        """Returns the S3 bucket name."""
        pass

    @abstractmethod
    def _object_type(self) -> str:
        """Returns the object type (prefix) for the data catalog."""
        pass

    @property
    @abstractmethod
    def metadata_cls(self) -> Type[M]:
        pass

    @property
    @abstractmethod
    def data_metadata_cls(self) -> Type[DM]:
        pass

    @property
    @abstractmethod
    def base_filename(self) -> str:
        pass

    def _get_s3_key_prefix(self, metadata: M) -> str:
        path = metadata.to_path()
        object_type = self._object_type()
        full_path = f"{self.s3_prefix}/{object_type}/{str(path.as_posix())}/"
        return full_path.lstrip('/')

    @abstractmethod
    def add_record(self, data_metadata: DM) -> None:
        """Add a record to the S3 catalog."""
        pass

    @abstractmethod
    def load_dataset(self, metadata: M) -> List[DM]:
        """Load a dataset from the S3 catalog based on metadata."""
        pass

    def list_datasets(self) -> List[M]:
        prefix = f"{self.s3_prefix}/{self._object_type()}/" if self.s3_prefix else f"{self._object_type()}/"
        paginator = self.s3.get_paginator('list_objects_v2')
        result = paginator.paginate(Bucket=self._bucket_name(), Prefix=prefix)
        metadata_set = {}
        for page in result:
            for obj in page.get('Contents', []):
                key = obj['Key']
                metadata = self._metadata_from_key(key)
                if metadata:
                    metadata_set[metadata.to_path()] = metadata  # Use path as key to ensure uniqueness
        return list(metadata_set.values())

    def delete_dataset(self, metadata: M) -> None:
        base_key = self._get_s3_key_prefix(metadata)

        paginator = self.s3.get_paginator('list_objects_v2')
        response_iterator = paginator.paginate(
            Bucket=self._bucket_name(),
            Prefix=base_key
        )

        keys_to_delete = []
        for page in response_iterator:
            for obj in page.get('Contents', []):
                keys_to_delete.append({'Key': obj['Key']})

        # Delete objects in batches
        for i in range(0, len(keys_to_delete), 1000):
            batch = keys_to_delete[i:i+1000]
            self.s3.delete_objects(
                Bucket=self._bucket_name(),
                Delete={'Objects': batch}
            )

    def _metadata_from_key(self, key: str) -> M:
        prefix = f"{self.s3_prefix}/{self._object_type()}/" if self.s3_prefix else f"{self._object_type()}/"
        if not key.startswith(prefix):
            return None  # Key does not match the expected prefix
        key = key[len(prefix):]
        key_parts = PurePosixPath(key).parts
        metadata_fields = {}
        for part in key_parts:
            if '=' in part:
                k, v = part.split('=', 1)
                metadata_fields[k] = v
            else:
                break  # Stop at the first part that doesn't match 'key=value' format
        try:
            cls = self.metadata_cls
            field_types = cls.__annotations__
            for field, field_type in field_types.items():
                if field in metadata_fields:
                    # Handle types appropriately
                    if field_type == int:
                        metadata_fields[field] = int(metadata_fields[field])
                    elif field_type == float:
                        metadata_fields[field] = float(metadata_fields[field])
                    else:
                        metadata_fields[field] = metadata_fields[field]
            return cls(**metadata_fields)
        except Exception:
            return None
