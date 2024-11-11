from pathlib import Path
import logging
from typing import List, Type, TypeVar, Generic
from abc import ABC, abstractmethod
import h5py
from tqdm import tqdm  # Import tqdm for progress bar
from .IMetadata import IMetadata
from .IData import IData
from .IDataMetadata import IDataMetadata

D = TypeVar('D', bound=IData)
M = TypeVar('M', bound=IMetadata)
DM = TypeVar('DM', bound=IDataMetadata[D, M])


class LocalH5Catalog(ABC, Generic[M, D, DM]):
    logger = logging.getLogger(__name__)

    def __init__(self, catalog_directory: str = "D:/chinese_checkers_games"):
        self.logger.debug(f"Initializing LocalH5Catalog with catalog_directory: {catalog_directory}")
        try:
            self.catalog_path = Path(catalog_directory)
            self.logger.debug(f"Setting catalog path to {self.catalog_path}")

            if not self.catalog_path.exists():
                self.logger.debug(f"Directory {self.catalog_path} does not exist; attempting to create it.")
            self.catalog_path.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Successfully initialized catalog at {self.catalog_path}")
        except Exception as e:
            self.logger.error(f"Failed to initialize catalog directory at {catalog_directory}: {e}")
            raise

    @property
    @abstractmethod
    def filename(self) -> str:
        ...

    @property
    @abstractmethod
    def metadata_cls(self) -> Type[M]:
        ...

    @property
    @abstractmethod
    def data_cls(self) -> Type[D]:
        ...

    @property
    @abstractmethod
    def data_metadata_cls(self) -> Type[DM]:
        ...

    def _construct_path(self, metadata: M) -> Path:
        path = self.catalog_path.joinpath(metadata.to_path(), self.filename)
        self.logger.debug(f"Constructed path for metadata {metadata}: {path}")
        return path

    def add_record(self, data_metadata: DM) -> None:
        self.logger.debug(f"Adding a single record with metadata: {data_metadata.metadata}")
        self._add_record_internal(data_metadata)

    def add_record_list(self, data_metadata_list: List[DM], batch_size: int = 1000) -> None:
        total_batches = (len(data_metadata_list) + batch_size - 1) // batch_size
        self.logger.info(
            f"Adding {len(data_metadata_list)} records in batches of {batch_size} (total {total_batches} batches)")

        for i in tqdm(range(0, len(data_metadata_list), batch_size), total=total_batches,
                      desc="Adding records in batches", disable=True):
            batch = data_metadata_list[i:i + batch_size]
            self.logger.info(f"Processing batch {i // batch_size + 1} of {total_batches}")
            for data_metadata in batch:
                self._add_record_internal(data_metadata)

    def _add_record_internal(self, data_metadata: DM) -> None:
        file_path = self._construct_path(data_metadata.metadata)
        storable_data = data_metadata.data.to_storable()

        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            if not file_path.exists():
                with h5py.File(file_path, 'w') as _:
                    pass
                self.logger.info(f"Created new dataset for metadata {data_metadata.metadata}")

            with h5py.File(file_path, 'a') as h5file:
                group = h5file.create_group(f'data_{len(h5file.keys())}')
                for key, value in storable_data.items():
                    group.create_dataset(key, data=value)

            self.logger.info(f"Added record to dataset with metadata {data_metadata.metadata}")
        except Exception as e:
            self.logger.error(f"Failed to add record for metadata {data_metadata.metadata} at {file_path}: {e}")
            raise

    def load_dataset(self, metadata: M) -> List[DM]:
        dataset_path = self._construct_path(metadata)
        self.logger.debug(f"Loading dataset for metadata: {metadata} from path: {dataset_path}")

        if not dataset_path.is_file():
            self.logger.error(f"Dataset file {dataset_path} does not exist.")
            return []

        records = []
        try:
            with h5py.File(dataset_path, 'r') as h5file:
                for key in h5file.keys():
                    group = h5file[key]
                    storable_data = {k: group[k][:] for k in group.keys()}
                    records.append(
                        self.data_metadata_cls.from_data_metadata(self.data_cls.from_storable(storable_data), metadata))
            self.logger.info(f"Loaded {len(records)} records for metadata {metadata}")
        except Exception as e:
            self.logger.error(f"Failed to load dataset for metadata {metadata} at {dataset_path}: {e}")
            raise

        return records

    def list_datasets(self) -> List[M]:
        self.logger.debug("Listing available datasets in catalog")
        dataset_paths = list(self.catalog_path.rglob(self.filename))
        metadata_list = [self._extract_metadata_from_path(path.parent, self.metadata_cls) for path in dataset_paths]
        self.logger.info(f"Found {len(metadata_list)} datasets available in the catalog")
        return metadata_list

    @classmethod
    def _extract_metadata_from_path(cls, directory_path: Path, metadata_cls: Type[M]) -> M:
        try:
            parts = {part.split('=')[0]: part.split('=')[1] for part in directory_path.parts if '=' in part}
            metadata_fields = metadata_cls.__annotations__

            for key, value in parts.items():
                if key in metadata_fields:
                    field_type = metadata_fields[key]
                    if field_type == int:
                        parts[key] = int(value)

            return metadata_cls(**parts)
        except (IndexError, ValueError, TypeError) as e:
            cls.logger.error(f"Failed to extract metadata from path {directory_path}: {e}")
            raise

    def delete_dataset(self, metadata: M) -> None:
        dataset_path = self._construct_path(metadata)
        self.logger.debug(f"Attempting to delete dataset at path: {dataset_path}")

        if not dataset_path.is_file():
            self.logger.warning(f"Dataset file {dataset_path} does not exist. Nothing to delete.")
            return

        try:
            dataset_path.unlink()
            self.logger.info(f"Deleted dataset file at {dataset_path}")

            parent_dir = dataset_path.parent
            if not any(parent_dir.iterdir()):
                parent_dir.rmdir()
                self.logger.info(f"Deleted empty directory at {parent_dir}")
        except Exception as e:
            self.logger.error(f"Failed to delete dataset at {dataset_path}: {e}")
            raise
