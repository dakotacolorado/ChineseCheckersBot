from abc import ABC, abstractmethod
from typing import Tuple
import torch
from ..IEncoder import IEncoder
from ...game.Move import Move

class IMoveEncoder(IEncoder[Move], ABC):
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

    @property
    @abstractmethod
    def shape(self) -> Tuple[int, ...]:
        """
        Returns the shape of the output tensor from the encoding.

        Returns:
            Tuple[int, ...]: Shape of the encoded torch.Tensor.
        """
        pass

    @abstractmethod
    def encode(self, move: Move) -> torch.Tensor:
        """
        Encodes a single move as a torch.Tensor.

        Args:
            move (Move): The move to encode.

        Returns:
            torch.Tensor: Encoded representation of the move.
        """
        pass

    def create_dummy_action(self) -> torch.Tensor:
        """
        Creates a dummy action encoding to use as padding when fewer moves are available.

        Returns:
            torch.Tensor: A tensor filled with zeros, shaped according to the output shape.
        """
        return torch.zeros(self.shape, dtype=torch.int32)
