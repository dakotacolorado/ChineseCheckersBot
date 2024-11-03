from abc import ABC, abstractmethod
from typing import Tuple
import torch
from ..IEncoder import IEncoder
from ...experience import Experience


class IExperienceEncoder(IEncoder[Experience], ABC):
    """
    Interface for encoders that encode an experience tuple into a format suitable for a DQN model.
    """

    @property
    def input_type(self) -> type:
        """
        Returns the expected type of the input object for encoding.
        """
        return Experience

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
    def encode(self, experience: Experience) -> Tuple[torch.Tensor, torch.Tensor, float, torch.Tensor, bool]:
        """
        Encodes an `Experience` instance into DQN-compatible format.

        Args:
            experience (Experience): The experience to encode.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, float, torch.Tensor, bool]: Encoded experience tuple with:
                - Encoded current state (torch.Tensor)
                - Encoded action (torch.Tensor)
                - Reward (float)
                - Encoded next state (torch.Tensor)
                - Done flag (bool)
        """
        pass