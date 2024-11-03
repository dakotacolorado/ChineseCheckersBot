from pathlib import Path
import logging
from typing import List, Type, TypeVar, Generic, Iterator
from abc import ABC, abstractmethod
import h5py
from .IMetadata import IMetadata
from .IData import IData

logger = logging.getLogger(__name__)

M = TypeVar('M', bound=IMetadata)
D = TypeVar('D', bound=IData)

class LocalH5Catalog(ABC, Generic[M, D]):
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

    def _construct_path(self, metadata: M) -> Path:
        """Constructs the path for the dataset based on metadata."""
        return self.catalog_path.joinpath(metadata.to_path(), self.filename)

    def create_dataset(self, metadata: M) -> None:
        """Creates a new dataset identified by the provided metadata if it does not exist."""
        file_path = self._construct_path(metadata)
        if not file_path.exists():
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with h5py.File(file_path, 'w') as _:
                pass  # Creates an empty HDF5 file
            logger.info(f"Created new dataset for metadata {metadata}")
        else:
            logger.info(f"Dataset already exists for metadata {metadata}")

    def add_record(self, metadata: M, record: D) -> None:
        """Appends a record to the dataset identified by the provided metadata key."""
        file_path = self._construct_path(metadata)
        storable_data = record.to_storable()

        with h5py.File(file_path, 'a') as h5file:
            group = h5file.create_group(f'data_{len(h5file.keys())}')
            for key, value in storable_data.items():
                group.create_dataset(key, data=value)

        logger.info(f"Added record to dataset with metadata {metadata}")

    def load_dataset(self, metadata: M) -> List[D]:
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
                records.append(self.data_cls.from_storable(storable_data))

        return records

    def list_datasets(self) -> List[M]:
        """Returns a list of metadata objects currently available in the catalog."""
        dataset_paths = list(self.catalog_path.rglob(self.filename))
        return [self._extract_metadata_from_path(path.parent, self.metadata_cls) for path in dataset_paths]

    @staticmethod
    def _extract_metadata_from_path(directory_path: Path, metadata_cls: Type[M]) -> M:
        """Extracts metadata from a directory path using the metadata class's structure with type conversion."""
        try:
            parts = {part.split('=')[0]: part.split('=')[1] for part in directory_path.parts if '=' in part}
            metadata_fields = metadata_cls.__annotations__

            # Convert each part to the appropriate type based on metadata_cls field types
            for key, value in parts.items():
                if key in metadata_fields and metadata_fields[key] == int:
                    parts[key] = int(value)

            return metadata_cls(**parts)
        except (IndexError, ValueError, TypeError) as e:
            logger.error(f"Failed to extract metadata from path {directory_path}: {e}")
            raise
