from abc import ABC, abstractmethod
from typing import Tuple, TypeVar
import torch
from .IEncoder import IEncoder

# Define T for input type, S is explicitly torch.Tensor for ITensorEncoder
T = TypeVar('T')

class ITensorEncoder(IEncoder[T, torch.Tensor, Tuple[int, ...]], ABC):
    """
    Interface for encoders that output a single tensor with a defined shape. Implementations
    should specify the tensor's shape and provide encoding logic to convert input data
    into a torch.Tensor of this shape.
    """

    @property
    @abstractmethod
    def shape(self) -> Tuple[int, ...]:
        """
        Defines the shape of the encoded torch.Tensor.

        Returns:
            Tuple[int, ...]: The shape of the output tensor, where each integer represents
                             the size of a specific dimension.
        """
        pass
