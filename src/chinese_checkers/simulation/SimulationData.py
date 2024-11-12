from dataclasses import dataclass
from typing import List, Type, Optional
import numpy as np

from ..catalog import IData
from ..game.Move import Move
from ..game.Position import Position


@dataclass(frozen=True)
class SimulationData(IData):
    """Data related to player positions in the game."""

    player_ids: List[str]
    player_start_positions: List[List[Position]]
    player_target_positions: List[List[Position]]
    historical_moves: List[Move]

    def to_storable(self) -> dict:
        """
        Converts the object to a format suitable for h5py storage.

        Notes:
            The player_ids are converted to byte strings to be compatible with h5py.
        """
        return {
            "player_ids": self._player_ids_to_bytes(self.player_ids),
            "player_start_positions": self._nested_list_to_array(self.player_start_positions),
            "player_target_positions": self._nested_list_to_array(self.player_target_positions),
            "historical_moves": np.array([move.to_tuple() for move in self.historical_moves])
        }

    @staticmethod
    def _player_ids_to_bytes(player_ids) -> np.ndarray:
        return np.array(player_ids, dtype="S")

    @staticmethod
    def _nested_list_to_array(data: List[List[Position]]) -> np.ndarray:
        return np.array([[pos.to_tuple() for pos in sublist] for sublist in data])

    @staticmethod
    def from_storable(data: dict) -> 'SimulationData':
        """
        Creates a GamePositions object from stored h5py data.

        Ensures that the required keys exist in the data dictionary to safeguard against potential errors.
        """
        if not all(key in data for key in
                   ["player_ids", "player_start_positions", "player_target_positions", "historical_moves"]):
            raise ValueError("The provided data dictionary is missing required keys.")

        player_ids = list(map(lambda x: x.decode(), data["player_ids"]))
        player_start_positions = SimulationData._convert_to_positions(data["player_start_positions"])
        player_target_positions = SimulationData._convert_to_positions(data["player_target_positions"])
        historical_moves = list(map(lambda move: Move.from_tuple(tuple(move)), data["historical_moves"]))

        return SimulationData(
            player_ids=player_ids,
            player_start_positions=player_start_positions,
            player_target_positions=player_target_positions,
            historical_moves=historical_moves
        )

    @staticmethod
    def _convert_to_positions(data: np.ndarray) -> List[List[Position]]:
        return [[Position.from_tuple(tuple(pos)) for pos in sublist] for sublist in data]
