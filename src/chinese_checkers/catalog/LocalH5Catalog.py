from pathlib import Path
import logging
from typing import List, Type, TypeVar, Generic
from abc import ABC, abstractmethod
import h5py
from tqdm import tqdm  # Import tqdm for progress bar
from .IMetadata import IMetadata
from .IData import IData
from .IDataMetadata import IDataMetadata

logger = logging.getLogger(__name__)

D = TypeVar('D', bound=IData)
M = TypeVar('M', bound=IMetadata)
DM = TypeVar('DM', bound=IDataMetadata[D, M])

class LocalH5Catalog(ABC, Generic[M, D, DM]):
    def __init__(self, catalog_directory: str = "D:/chinese_checkers_games"):
        self.catalog_path = Path(catalog_directory)
        self.catalog_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized catalog at {self.catalog_path}")

    @property
    @abstractmethod
    def filename(self) -> str:
        """Abstract property that must be defined by each subclass to specify the filename."""
        ...

    @property
    @abstractmethod
    def metadata_cls(self) -> Type[M]:
        """The concrete metadata class to be used in the catalog."""
        ...

    @property
    @abstractmethod
    def data_cls(self) -> Type[D]:
        """The concrete data class to be used in the catalog."""
        ...

    @property
    @abstractmethod
    def data_metadata_cls(self) -> Type[DM]:
        """The concrete data-metadata class to be used in the catalog."""
        ...

    def _construct_path(self, metadata: M) -> Path:
        """Constructs the path for the dataset based on metadata."""
        return self.catalog_path.joinpath(metadata.to_path(), self.filename)

    def add_record(self, data_metadata: DM) -> None:
        """Appends a single record to the dataset identified by the provided metadata key."""
        self._add_record_internal(data_metadata)

    def add_record_list(self, data_metadata_list: List[DM], batch_size: int = 1000) -> None:
        """Appends records in batches to the dataset identified by the provided metadata keys."""
        total_batches = (len(data_metadata_list) + batch_size - 1) // batch_size  # Calculate total number of batches

        for i in tqdm(range(0, len(data_metadata_list), batch_size), total=total_batches, desc="Adding records in batches", disable=True):
            batch = data_metadata_list[i:i + batch_size]
            logger.info(f"Processing batch {i // batch_size + 1} of {total_batches}")
            for data_metadata in batch:
                self._add_record_internal(data_metadata)

    def _add_record_internal(self, data_metadata: DM) -> None:
        """Internal helper to append a record to the dataset, creating files and directories as needed."""
        file_path = self._construct_path(data_metadata.metadata)
        storable_data = data_metadata.data.to_storable()

        # Ensure the directory and file exist before adding the record
        file_path.parent.mkdir(parents=True, exist_ok=True)
        if not file_path.exists():
            with h5py.File(file_path, 'w') as _:
                pass
            logger.info(f"Created new dataset for metadata {data_metadata.metadata}")

        # Append the record to the file
        with h5py.File(file_path, 'a') as h5file:
            group = h5file.create_group(f'data_{len(h5file.keys())}')
            for key, value in storable_data.items():
                group.create_dataset(key, data=value)

        logger.info(f"Added record to dataset with metadata {data_metadata.metadata}")

    def load_dataset(self, metadata: M) -> List[DM]:
        """Loads all records associated with the specified metadata key."""
        dataset_path = self._construct_path(metadata)

        if not dataset_path.is_file():
            logger.error(f"Dataset file {dataset_path} does not exist.")
            return []

        records = []
        with h5py.File(dataset_path, 'r') as h5file:
            for key in h5file.keys():
                group = h5file[key]
                storable_data = {k: group[k][:] for k in group.keys()}
                records.append(self.data_metadata_cls.from_data_metadata(self.data_cls.from_storable(storable_data), metadata))

        return records

    def list_datasets(self) -> List[M]:
        """Returns a list of metadata objects currently available in the catalog."""
        dataset_paths = list(self.catalog_path.rglob(self.filename))
        return [self._extract_metadata_from_path(path.parent, self.metadata_cls) for path in dataset_paths]

    @classmethod
    def _extract_metadata_from_path(cls, directory_path: Path, metadata_cls: Type[M]) -> M:
        """Extracts metadata from a directory path using the metadata class's structure with type conversion."""
        try:
            parts = {part.split('=')[0]: part.split('=')[1] for part in directory_path.parts if '=' in part}
            metadata_fields = metadata_cls.__annotations__

            # Ensure type consistency based on metadata_cls field types
            for key, value in parts.items():
                if key in metadata_fields:
                    # Convert to int if required by the field annotation
                    field_type = metadata_fields[key]
                    if field_type == int:
                        parts[key] = int(value)

            return metadata_cls(**parts)
        except (IndexError, ValueError, TypeError) as e:
            logger.error(f"Failed to extract metadata from path {directory_path}: {e}")
            raise

    def delete_dataset(self, metadata: M) -> None:
        """Deletes the dataset associated with the specified metadata key."""
        dataset_path = self._construct_path(metadata)

        # Debugging: Log the constructed path and expected filename
        logger.debug(f"Attempting to delete dataset at path: {dataset_path}")
        logger.debug(f"Expected filename: {self.filename}")

        if not dataset_path.is_file():
            logger.warning(f"Dataset file {dataset_path} does not exist. Nothing to delete.")
            return

        try:
            # Remove the file
            dataset_path.unlink()
            logger.info(f"Deleted dataset file at {dataset_path}")

            # Remove the directory if it is empty
            parent_dir = dataset_path.parent
            if not any(parent_dir.iterdir()):  # Check if the directory is empty
                parent_dir.rmdir()
                logger.info(f"Deleted empty directory at {parent_dir}")

        except Exception as e:
            logger.error(f"Failed to delete dataset at {dataset_path}: {e}")
            raise

