from dataclasses import dataclass
from typing import List

import numpy as np

from ..game.Move import Move
from ..game.Position import Position


@dataclass(frozen=True)
class SimulationData:
    """Data related to player positions in the game."""
    player_ids: List[str]
    player_start_positions: List[List[Position]]
    player_target_positions: List[List[Position]]
    historical_moves: List[Move]

    def to_storable(self: "SimulationData") -> dict:
        """Converts the object to a format suitable for h5py storage."""
        return {
            "player_ids": np.array(self.player_ids, dtype="S"),  # Convert to byte strings for h5py
            "player_start_positions": np.array(
                [[pos.to_tuple() for pos in sublist] for sublist in self.player_start_positions]
            ),
            "player_target_positions": np.array(
                [[pos.to_tuple() for pos in sublist] for sublist in self.player_target_positions]
            ),
            "historical_moves": np.array([move.to_tuple() for move in self.historical_moves])
        }

    @staticmethod
    def from_storable(data: dict) -> "SimulationData":
        """Creates a GamePositions object from stored h5py data."""
        player_ids = [pid.decode() for pid in data["player_ids"]]
        player_start_positions = [
            [Position.from_tuple(tuple(pos)) for pos in sublist] for sublist in data["player_start_positions"].tolist()
        ]
        player_target_positions = [
            [Position.from_tuple(tuple(pos)) for pos in sublist] for sublist in data["player_target_positions"].tolist()
        ]
        historical_moves = [Move.from_tuple(tuple(move)) for move in data["historical_moves"].tolist()]

        return SimulationData(
            player_ids=player_ids,
            player_start_positions=player_start_positions,
            player_target_positions=player_target_positions,
            historical_moves=historical_moves
        )
