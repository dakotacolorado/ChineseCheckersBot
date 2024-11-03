from typing import List, Protocol

from .IMetadata import IMetadata


class ICatalog(Protocol):
    def create_dataset(self, metadata: "IMetadata") -> None:
        """Creates a new dataset identified by the provided metadata.
        If a dataset with this metadata already exists, it remains unchanged."""
        ...

    def add_record(self, metadata: "IMetadata", record: any) -> None:
        """Appends a record to the dataset identified by the provided metadata key."""
        ...

    def load_dataset(self, metadata: "IMetadata") -> List:
        """Retrieves all records associated with the specified metadata key."""
        ...

    def list_datasets(self) -> List["IMetadata"]:
        """Returns a list of metadata keys currently available in the catalog."""
        ...
