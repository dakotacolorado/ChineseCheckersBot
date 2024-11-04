from typing import Dict, Callable
from .ISimulationEncoder import ISimulationEncoder
from .DistanceToWinRewardSimulationEncoder import DistanceToWinRewardSimulationEncoder
from .. import IFactory


class SimulationRewardEncoderFactory(IFactory):

    @property
    def _encoders(self) -> Dict[str, Callable[[], ISimulationEncoder]]:
        return {
            "distance_to_win_reward_strategy": lambda: DistanceToWinRewardSimulationEncoder(),
            # Additional reward strategies can be added here in the future
        }