from abc import ABC, abstractmethod
from typing import TypeVar, Dict, Callable, Generic

# Define a generic type variable for the factory output type
T = TypeVar('T')

class IFactory(ABC, Generic[T]):
    """
    Generic factory interface for creating instances of a specified type based on a string identifier.
    """

    @property
    @abstractmethod
    def _encoders(self) -> Dict[str, Callable[[], T]]:
        """
        A dictionary mapping string identifiers to encoder classes.

        Returns:
            Dict[str, Callable[[], T]]: The available encoders.
        """
        pass

    def create(self, name: str) -> T:
        """
        Creates an instance of the specified type based on a string identifier.

        Args:
            name (str): The name identifier of the type to create.

        Returns:
            T: An instance of the requested type.

        Raises:
            ValueError: If the specified encoder name is not available.
        """
        encoder_class = self._encoders.get(name)
        if not encoder_class:
            raise ValueError(f"Encoder strategy '{name}' is not available.")
        return encoder_class()
