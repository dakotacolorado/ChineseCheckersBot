import awswrangler as wr
import pandas as pd
from typing import Type, List
from ..catalog import S3DataCatalog
from .SimulationData import SimulationData
from .SimulationMetadata import SimulationMetadata
from .GameSimulation import GameSimulation

class S3SimulationCatalog(S3DataCatalog[SimulationMetadata, GameSimulation]):
    def __init__(self, s3_client=None, s3_prefix: str = '', batch_size: int = 100):
        super().__init__(s3_client, s3_prefix)
        self.batch_size = batch_size
        self.batch_data = []  # Initialize batch data list
        self.current_metadata = None  # Initialize current metadata

    def _bucket_name(self) -> str:
        return 'dakotajp-chinese-checkers-game-simulations'

    def _object_type(self) -> str:
        return 'simulation'

    @property
    def metadata_cls(self) -> Type[SimulationMetadata]:
        return SimulationMetadata

    @property
    def data_metadata_cls(self) -> Type[GameSimulation]:
        return GameSimulation

    @property
    def base_filename(self) -> str:
        return 'SimulationData.parquet'

    def add_record(self, data_metadata: GameSimulation) -> None:
        if self.current_metadata is None:
            self.current_metadata = data_metadata.metadata
        elif self.current_metadata != data_metadata.metadata:
            # Flush the batch if metadata changes
            self.flush()
            self.current_metadata = data_metadata.metadata

        self.batch_data.append(data_metadata.data)

        if len(self.batch_data) >= self.batch_size:
            self.flush()

    def add_record_list(self, data_metadata_list: List[GameSimulation]) -> None:
        """Appends records in batches to the dataset identified by the provided metadata keys."""
        for data_metadata in data_metadata_list:
            self.add_record(data_metadata)
        # Optionally flush after adding all records to ensure any remaining data is saved
        self.flush()

    def flush(self):
        if not self.batch_data:
            return

        # Convert batch data to a DataFrame
        dataframes = [data.to_dataframe() for data in self.batch_data]
        combined_df = pd.concat(dataframes, ignore_index=True)

        # Prepare S3 path
        base_key = self._get_s3_key_prefix(self.current_metadata)
        s3_path = f"s3://{self._bucket_name()}/{base_key}"

        # Write DataFrame to S3 using AWS Data Wrangler
        wr.s3.to_parquet(
            df=combined_df,
            path=s3_path,
            dataset=True,   # Use dataset=True to organize data in a dataset format
            mode='append',  # Append to existing data
            boto3_session=self._s3_session  # Use the existing boto3 session if needed
        )

        # Clear the batch
        self.batch_data = []

    def load_dataset(self, metadata: SimulationMetadata) -> List[GameSimulation]:
        base_key = self._get_s3_key_prefix(metadata)
        s3_path = f"s3://{self._bucket_name()}/{base_key}"

        # Read Parquet files from S3
        df = wr.s3.read_parquet(
            path=s3_path,
            dataset=True,  # Read as a dataset
            boto3_session=self._s3_session  # Use the existing boto3 session if needed
        )

        # Reconstruct SimulationData instances
        simulations = []
        for _, row in df.iterrows():
            data = SimulationData.from_dataframe(row)
            simulation = GameSimulation(data=data, metadata=metadata)
            simulations.append(simulation)

        return simulations

    @property
    def _s3_session(self):
        # Create a boto3 session using the s3 client if necessary
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
