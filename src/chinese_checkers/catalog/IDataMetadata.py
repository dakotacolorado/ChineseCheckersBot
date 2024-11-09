from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from .IMetadata import IMetadata
from .IData import IData

T = TypeVar('T', bound=IData)  # Generic type for data
S = TypeVar('S', bound=IMetadata)  # Generic type for metadata


class IDataMetadata(ABC, Generic[T, S]):
    def __init__(self, data: T, metadata: S):
        self.data = data
        self.metadata = metadata

    @staticmethod
    @abstractmethod
    def from_data_metadata(data: T, metadata: S) -> 'IDataMetadata[T, S]':
        """
        Abstract factory method to create an instance from data and metadata.

        Args:
            data (T): The data object.
            metadata (S): The metadata object.

        Returns:
            IDataMetadata[T, S]: A new instance of the implementing class.
        """
        pass
