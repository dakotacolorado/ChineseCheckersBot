from dataclasses import dataclass
from typing import List

from ..game import Player

@dataclass
class DirectoryAttributes:
    player_count: int
    board_size: int
    max_game_length: int
    name: str
    version: str
    winning_player_id: str

@dataclass
class GameData:
    player_position_history: List[List[Player]]

@dataclass
class GameSimulationData:
    directory: DirectoryAttributes
    data: GameData
