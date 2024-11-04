from abc import abstractmethod, ABC
from typing import Protocol, Dict, Any

class IData(ABC):

    @abstractmethod
    def to_storable(self) -> Dict[str, Any]:
        """Converts the object to a dictionary format suitable for storage."""
        ...

    @staticmethod
    @abstractmethod
    def from_storable(data: Dict[str, Any]) -> "IData":
        """Creates an instance of the implementing class from stored data."""
        ...
