from abc import ABC
from typing import Type

from .. import IEncoder
from ...simulation import GameSimulation

class ISimulationRewardEncoder(IEncoder[GameSimulation], ABC):
    """
    Interface for reward encoders that calculate a reward based on the state of a GameSimulation.
    """

    @property
    def input_type(self) -> Type:
        """
        Returns the expected type of the input object for encoding, which is GameSimulation.
        """
        return GameSimulation
