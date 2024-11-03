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

    def _construct_path(self, metadata: M) -> Path:
        """Constructs the path for the dataset based on metadata."""
        return self.catalog_path.joinpath(metadata.to_path(), self.filename)

    def save_data(self, metadata: M, data: D) -> None:
        """Saves data associated with the provided metadata, appending to an existing dataset or creating a new one."""
        file_path = self._construct_path(metadata)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        storable_data = data.to_storable()

        with h5py.File(file_path, 'a') as h5file:
            group = h5file.create_group(f'data_{len(h5file.keys())}')
            for key, value in storable_data.items():
                group.create_dataset(key, data=value)

        logger.info(f"Saved data at {file_path}")

    def load_data(self, metadata: M, data_cls: Type[D]) -> Iterator[D]:
        """Loads data associated with the specified metadata."""
        dataset_path = self._construct_path(metadata)

        if not dataset_path.is_file():
            logger.error(f"Dataset file {dataset_path} does not exist.")
            return

        with h5py.File(dataset_path, 'r') as h5file:
            for key in h5file.keys():
                group = h5file[key]
                storable_data = {k: group[k][:] for k in group.keys()}
                yield data_cls.from_storable(storable_data)

    def list_available_metadata(self, metadata_cls: Type[M]) -> List[M]:
        """Lists all metadata objects available in the catalog."""
        dataset_paths = list(self.catalog_path.rglob(self.filename))
        return [self._extract_metadata_from_path(path.parent, metadata_cls) for path in dataset_paths]

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
