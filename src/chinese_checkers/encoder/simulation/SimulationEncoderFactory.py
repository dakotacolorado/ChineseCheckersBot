from typing import Dict, Callable
from .ISimulationRewardEncoder import ISimulationRewardEncoder
from .DistanceToWinRewardStrategyEncoder import DistanceToWinRewardStrategyEncoder
from .. import IFactory


class SimulationRewardEncoderFactory(IFactory):

    @property
    def _encoders(self) -> Dict[str, Callable[[], ISimulationRewardEncoder]]:
        return {
            "distance_to_win_reward_strategy": lambda: DistanceToWinRewardStrategyEncoder(),
            # Additional reward strategies can be added here in the future
        }