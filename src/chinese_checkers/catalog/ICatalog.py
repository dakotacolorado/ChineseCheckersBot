from typing import List, Protocol

from .IData import IData
from .IDataMetadata import IDataMetadata
from .IMetadata import IMetadata


class ICatalog(Protocol):
    def add_record(self, data_metadata: IDataMetadata) -> None:
        """Appends a record to the dataset identified by the provided metadata key."""
        ...

    def add_record_list(self, data_metadata_list: List[IDataMetadata]) -> None:
        """Appends records in batches to the dataset identified by the provided metadata keys."""
        ...

    def load_dataset(self, metadata: IMetadata) -> List[IDataMetadata]:
        """Retrieves all records associated with the specified metadata key."""
        ...

    def list_datasets(self) -> List[IMetadata]:
        """Returns a list of metadata keys currently available in the catalog."""
        ...

    def delete_dataset(self, metadata: IMetadata) -> None:
        """Deletes the dataset associated with the specified metadata key."""
        ...
