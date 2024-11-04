from abc import ABC, abstractmethod
from typing import Tuple, TypeVar
import torch
from .IEncoder import IEncoder

# Define T for input type, S is explicitly Tuple[torch.Tensor, ...] for ITensorTupleEncoder
T = TypeVar('T')

class ITensorTupleEncoder(IEncoder[T, Tuple[torch.Tensor, ...], Tuple[Tuple[int, ...], ...]], ABC):
    """
    Interface for encoders that output a tuple of tensors, each with a defined shape. Implementations
    should specify the shapes of each tensor in the output tuple and provide encoding logic to convert
    input data into this tuple of torch.Tensors.
    """

    @property
    @abstractmethod
    def shape(self) -> Tuple[Tuple[int, ...], ...]:
        """
        Defines the shape of each tensor in the output tuple.

        Returns:
            Tuple[Tuple[int, ...], ...]: A tuple where each element is a tuple representing
                                         the shape of a tensor in the output. Each inner tuple
                                         specifies the size of each dimension for a tensor.
        """
        pass
