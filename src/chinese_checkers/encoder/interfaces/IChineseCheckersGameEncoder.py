from abc import ABC, abstractmethod
from typing import Tuple
import torch

from chinese_checkers.encoder.interfaces.IEncoder import IEncoder
from chinese_checkers.game.ChineseCheckersGame import ChineseCheckersGame


class IChineseCheckersGameEncoder(IEncoder[ChineseCheckersGame], ABC):
    """
    Base interface for all Chinese Checkers game encoders.
    """

    @property
    def input_type(self) -> type:
        """
        Returns the expected type of the input object for encoding.
        """
        return ChineseCheckersGame

    @property
    @abstractmethod
    def shape(self) -> Tuple[int, ...]:
        """
        Returns the shape of the output tensor from the encoding.
        """
        pass

    @abstractmethod
    def encode(self, game: ChineseCheckersGame) -> torch.Tensor:
        """
        Encodes a `ChineseCheckersGame` state into a torch.Tensor.

        Args:
            game (ChineseCheckersGame): The game state to encode.

        Returns:
            torch.Tensor: The encoded game state.
        """
        pass
