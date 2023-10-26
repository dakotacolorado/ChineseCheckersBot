from dataclasses import dataclass
from typing import List

from ..game import Player

@dataclass
class DirectoryAttributes:
    player_count: int
    board_size: int
    max_game_length: int
    winning_player: str
    name: str
    version: str

@dataclass
class GameData:
    player_position_history: List[List[Player]]

    @staticmethod
    def to_storable(self):
        """
        I want to store the player locations and player IDs.

        player_ids: List[str]
        player_target_positions: List[List[(int, int)]]
        player_historical_positions: List[List[List[(int, int)]]]
        """
        pass

    @staticmethod
    def from_storable(self) -> List[List[Player]]:
        pass

@dataclass
class GameSimulationData:
    directory: DirectoryAttributes
    data: GameData
