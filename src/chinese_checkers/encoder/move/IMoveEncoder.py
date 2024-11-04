from abc import ABC
from typing import TypeVar, Generic

from ..IEncoder import IEncoder
from ...game.Move import Move

# output type T, output shape S
T = TypeVar('T')
S = TypeVar('S')

class IMoveEncoder(IEncoder[Move, T, S], ABC, Generic[T, S]):
    """
    Interface for encoders that encode moves in Chinese Checkers, converting them
    into start and end coordinates as a tensor.
    """

    @property
    def input_type(self) -> type:
        """
        Returns the expected type of the input object for encoding.
        """
        return Move

