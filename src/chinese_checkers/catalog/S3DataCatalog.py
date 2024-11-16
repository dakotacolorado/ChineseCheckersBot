import boto3
from abc import ABC, abstractmethod
from typing import Generic, TypeVar, List, Type
from pathlib import PurePosixPath
import random
import awswrangler as wr
import pandas as pd
from .ICatalog import ICatalog
from .IMetadata import IMetadata
from .IDataMetadata import IDataMetadata

M = TypeVar('M', bound=IMetadata)
DM = TypeVar('DM', bound=IDataMetadata)


class S3DataCatalog(ABC, ICatalog, Generic[M, DM]):
    def __init__(self, s3_client=None, s3_prefix: str = '', batch_size: int = 100):
        self.s3 = s3_client if s3_client else boto3.client('s3')
        self.s3_prefix = s3_prefix.rstrip('/')
        self.batch_size = batch_size
        self.batches = {}

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

    def add_record(self, data_metadata: DM) -> None:
        metadata_key = data_metadata.metadata

        if metadata_key not in self.batches:
            self.batches[metadata_key] = []

        self.batches[metadata_key].append(data_metadata.data)

        if len(self.batches[metadata_key]) >= self.batch_size:
            self._flush_for_key(metadata_key)

    def add_record_list(self, data_metadata_list: List[DM]) -> None:
        """Appends records in batches to the dataset identified by the provided metadata keys."""
        for data_metadata in data_metadata_list:
            self.add_record(data_metadata)
        self.flush()

    def _flush_for_key(self, metadata_key) -> None:
        if not self.batches[metadata_key]:
            return

        # Convert batch data to a DataFrame
        dataframes = [data.to_dataframe() for data in self.batches[metadata_key]]
        combined_df = pd.concat(dataframes, ignore_index=True)

        # Prepare S3 path
        base_key = self._get_s3_key_prefix(metadata_key)
        s3_path = f"s3://{self._bucket_name()}/{base_key}"

        # Write DataFrame to S3 using AWS Data Wrangler
        wr.s3.to_parquet(
            df=combined_df,
            path=s3_path,
            dataset=True,
            mode='append',
            boto3_session=self.s3_session
        )

        self.batches[metadata_key] = []

    def flush(self) -> None:
        """Flushes all batches to S3."""
        for metadata_key in list(self.batches.keys()):
            self._flush_for_key(metadata_key)

    def load_dataset(self, metadata: M) -> List[DM]:
        base_key = self._get_s3_key_prefix(metadata)
        s3_path = f"s3://{self._bucket_name()}/{base_key}"

        # Read Parquet files from S3
        df = wr.s3.read_parquet(
            path=s3_path,
            dataset=True,
            boto3_session=self.s3_session
        )

        # Reconstruct data objects
        records = []
        for _, row in df.iterrows():
            data = self.data_metadata_cls.from_dataframe(row)
            record = self.data_metadata_cls(data=data, metadata=metadata)
            records.append(record)

        return records

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

    def load_random_batch(self, metadata: M, file_sample_count: int) -> List[DM]:
        """
        Loads a batch of records from a specified number of randomly selected Parquet files.

        Args:
            metadata (M): Metadata to filter the files.
            file_sample_count (int): Number of files to randomly select and load.

        Returns:
            List[DM]: List of records loaded from the selected files.
        """
        if not hasattr(self, "file_keys") or not self.file_keys:
            self.file_keys = self._fetch_file_keys(metadata)
            random.shuffle(self.file_keys)

        selected_files = random.sample(self.file_keys, min(file_sample_count, len(self.file_keys)))

        records = []
        for file in selected_files:
            df = wr.s3.read_parquet(file, boto3_session=self.s3_session)

            file_records = [
                self.data_metadata_cls.from_data_metadata(  # Use static method to create an instance
                    data=self.data_metadata_cls.data_cls().from_dataframe(row),  # Convert row to data
                    metadata=metadata
                )
                for _, row in df.iterrows()
            ]
            records.extend(file_records)

        return records

    def _fetch_file_keys(self, metadata: M) -> List[str]:
        """
        Fetches a list of all file keys (paths) matching the metadata in the S3 bucket.

        Args:
            metadata (M): Metadata to filter the files.

        Returns:
            List[str]: List of file keys.
        """
        base_key = self._get_s3_key_prefix(metadata)
        bucket = self._bucket_name()
        prefix = f"s3://{bucket}/{base_key}"

        file_keys = wr.s3.list_objects(path=prefix, boto3_session=self.s3_session)

        if not file_keys:
            raise ValueError(f"No files found under prefix: {prefix}")

        return file_keys

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

    @property
    def s3_session(self):
        import boto3
        if self.s3 is not None:
            credentials = self.s3._request_signer._credentials
            return boto3.Session(
                aws_access_key_id=credentials.access_key,
                aws_secret_access_key=credentials.secret_key,
                aws_session_token=credentials.token,
                region_name=self.s3._client_config.region_name
            )
        else:
            return boto3.Session()

    def __del__(self):
        self.flush()
