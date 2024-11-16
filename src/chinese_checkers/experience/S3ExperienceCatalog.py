import random

import awswrangler as wr
import pandas as pd
from typing import Type, List

from ..catalog import S3DataCatalog
from .ExperienceData import ExperienceData
from .ExperienceMetadata import ExperienceMetadata
from .Experience import Experience


class S3ExperienceCatalog(S3DataCatalog[ExperienceMetadata, Experience]):
    def __init__(self, s3_client=None, s3_prefix: str = '', batch_size: int = 10000):
        super().__init__(s3_client, s3_prefix)
        self.batch_size = batch_size
        self.batches = {}  # Dictionary to store lists of batch data for each metadata key

    def _bucket_name(self) -> str:
        return 'dakotajp-chinese-checkers-game-experiences'

    def _object_type(self) -> str:
        return 'experience'

    @property
    def metadata_cls(self) -> Type[ExperienceMetadata]:
        return ExperienceMetadata

    @property
    def data_metadata_cls(self) -> Type[Experience]:
        return Experience

    @property
    def base_filename(self) -> str:
        return 'ExperienceData.parquet'

    def add_record(self, data_metadata: Experience) -> None:
        metadata_key = data_metadata.metadata

        if metadata_key not in self.batches:
            self.batches[metadata_key] = []

        self.batches[metadata_key].append(data_metadata.data)

        if len(self.batches[metadata_key]) >= self.batch_size:
            self._flush_for_key(metadata_key)

    def add_record_list(self, data_metadata_list: List[Experience]) -> None:
        """Appends records in batches to the dataset identified by the provided metadata keys."""
        for data_metadata in data_metadata_list:
            self.add_record(data_metadata)
        # Optionally flush after adding all records to ensure any remaining data is saved
        self.flush()

    def _flush_for_key(self, metadata_key) -> None:
        """Private method to flush batch data for a specific metadata key."""
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

    def load_dataset(self, metadata: ExperienceMetadata) -> List[Experience]:
        base_key = self._get_s3_key_prefix(metadata)
        s3_path = f"s3://{self._bucket_name()}/{base_key}"

        # Read Parquet files from S3
        df = wr.s3.read_parquet(
            path=s3_path,
            dataset=True,
            boto3_session=self.s3_session
        )

        # Reconstruct ExperienceData instances
        experiences = []
        for _, row in df.iterrows():
            data = ExperienceData.from_dataframe(row)
            experience = Experience(data=data, metadata=metadata)
            experiences.append(experience)

        return experiences

    def load_random_batch(self, metadata: ExperienceMetadata, file_sample_count: int) -> List[Experience]:
        """
        Loads a batch of experiences from a specified number of randomly selected Parquet files.

        Args:
            metadata (ExperienceMetadata): Metadata to filter the files.
            file_sample_count (int): Number of files to randomly select and load.

        Returns:
            List[Experience]: List of experiences loaded from the selected files.
        """
        # Fetch and shuffle file keys if not already done
        if not hasattr(self, "file_keys") or not self.file_keys:
            self.file_keys = self._fetch_file_keys(metadata)
            random.shuffle(self.file_keys)

        # Select a random subset of files
        selected_files = random.sample(self.file_keys, min(file_sample_count, len(self.file_keys)))

        # Load data from the selected files
        experiences = []
        for file in selected_files:
            # Load Parquet data
            df = wr.s3.read_parquet(file, boto3_session=self.s3_session)

            # Convert each row to an Experience object
            file_experiences = [
                Experience(
                    data=ExperienceData.from_dataframe(row),
                    metadata=self.metadata_cls.from_simulation_metadata(metadata, metadata.generator_name, metadata.current_player)
                )
                for _, row in df.iterrows()
            ]
            experiences.extend(file_experiences)

        return experiences

    def _fetch_file_keys(self, metadata: ExperienceMetadata) -> List[str]:
        """
        Fetches a list of all file keys (paths) matching the metadata in the S3 bucket.
        """
        base_key = self._get_s3_key_prefix(metadata)
        bucket = self._bucket_name()
        prefix = f"s3://{bucket}/{base_key}"

        # Use awswrangler to list objects with minimal overhead
        file_keys = wr.s3.list_objects(path=prefix, boto3_session=self.s3_session)

        if not file_keys:
            raise ValueError(f"No files found under prefix: {prefix}")

        return file_keys

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
        # Ensure any remaining data is flushed when the object is destroyed
        self.flush()
