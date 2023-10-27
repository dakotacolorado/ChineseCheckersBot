from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from ..game import Player

@dataclass
class GameMetadata:
    """Metadata about the game simulation."""
    player_count: int
    board_size: int
    max_game_length: int
    winning_player: str
    name: str
    version: str

@dataclass
class GamePositions:
    """Data related to player positions in the game."""
    player_ids: List[str]
    player_target_positions: List[List[Tuple[int, int]]]
    player_historical_positions: List[List[List[Tuple[int, int]]]]

    @staticmethod
    def to_storable(game_positions: "GamePositions") -> dict:
        """Converts the object to a format suitable for h5py storage."""
        return {
            "player_ids": np.array(game_positions.player_ids, dtype="S"),  # Convert to byte strings for h5py
            "player_target_positions": np.array(game_positions.player_target_positions),
            "player_historical_positions": np.array(game_positions.player_historical_positions),
        }

    @staticmethod
    def from_storable(data: dict) -> "GamePositions":
        """Creates a GamePositions object from stored h5py data."""
        player_ids = [pid.decode() for pid in data["player_ids"]]

        player_target_positions = [tuple(position) for position in data["player_target_positions"].tolist()]
        player_historical_positions = [[tuple(position) for position in pos_list] for pos_list in
                                       data["player_historical_positions"].tolist()]

        return GamePositions(
            player_ids=player_ids,
            player_target_positions=player_target_positions,
            player_historical_positions=player_historical_positions
        )

@dataclass
class GameSimulationData:
    metadata: GameMetadata
    positions: GamePositions
