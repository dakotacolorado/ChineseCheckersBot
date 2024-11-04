from abc import ABC
from typing import Type

from typing_extensions import Generic, TypeVar

from ..IEncoder import IEncoder
from ...simulation import GameSimulation

T = TypeVar('T')
S = TypeVar('S')

class ISimulationEncoder(IEncoder[GameSimulation, T, S], ABC, Generic[T, S]):
    """
    Interface for reward encoders that calculate a reward based on the state of a GameSimulation.
    """

    @property
    def input_type(self) -> Type:
        """
        Returns the expected type of the input object for encoding, which is GameSimulation.
        """
        return GameSimulation
