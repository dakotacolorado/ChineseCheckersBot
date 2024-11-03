from abc import ABC, abstractmethod
from pathlib import Path


class IMetadata(ABC):
    @abstractmethod
    def to_path(self) -> Path:
        """Generates a unique file path based on metadata properties for organizing storage."""
        ...

    @abstractmethod
    def __hash__(self) -> int:
        """Enables metadata instances to serve as unique keys in a dictionary."""
        ...

    @abstractmethod
    def __eq__(self, other: object) -> bool:
        """Supports equality checks for metadata instances, allowing for reliable comparisons."""
        ...
