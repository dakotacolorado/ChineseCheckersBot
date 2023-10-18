from dataclasses import dataclass

from torch import tensor


@dataclass
class SimulationDataSet:
    player_count: int
    board_size: int
    game_length: int
    name: str
    version: str
    description: str
    data: tensor
    labels: tensor
