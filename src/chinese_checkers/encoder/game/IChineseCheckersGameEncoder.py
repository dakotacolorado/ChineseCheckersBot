from abc import ABC
from typing import TypeVar, Generic

from ..IEncoder import IEncoder
from ...game import ChineseCheckersGame

T = TypeVar('T')
S = TypeVar('S')

class IChineseCheckersGameEncoder(IEncoder[ChineseCheckersGame, T, S], ABC, Generic[T, S]):
    """
    Base interface for all Chinese Checkers game encoders.
    """

    @property
    def input_type(self) -> type:
        """
        Returns the expected type of the input object for encoding.
        """
        return ChineseCheckersGame

