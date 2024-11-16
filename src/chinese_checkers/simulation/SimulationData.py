import json
import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Any
import numpy as np
import pandas as pd

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
    uuid: str = field(default_factory=lambda: str(uuid.uuid4()))  # Auto-generate UUID if not provided

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
            "historical_moves": np.array([move.to_tuple() for move in self.historical_moves]),
            "uuid": self.uuid,
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
        Creates a SimulationData object from stored data.

        Ensures that the required keys exist in the data dictionary to safeguard against potential errors.
        """
        if not all(key in data for key in
                   ["player_ids", "player_start_positions", "player_target_positions", "historical_moves", "uuid"]):
            raise ValueError("The provided data dictionary is missing required keys.")

        player_ids = list(map(lambda x: x.decode(), data["player_ids"]))
        player_start_positions = SimulationData._convert_to_positions(data["player_start_positions"])
        player_target_positions = SimulationData._convert_to_positions(data["player_target_positions"])
        historical_moves = list(map(lambda move: Move.from_tuple(tuple(move)), data["historical_moves"]))

        return SimulationData(
            player_ids=player_ids,
            player_start_positions=player_start_positions,
            player_target_positions=player_target_positions,
            historical_moves=historical_moves,
            uuid=data.get("uuid", str(uuid.uuid4()))  # Use a new UUID if missing
        )

    @staticmethod
    def _convert_to_positions(data: np.ndarray) -> List[List[Position]]:
        return [[Position.from_tuple(tuple(pos)) for pos in sublist] for sublist in data]

    def to_dataframe(self) -> pd.DataFrame:
        """
        Converts the object into a Pandas DataFrame.
        """
        data = {
            'player_ids': json.dumps(self.player_ids),
            'player_start_positions': json.dumps(self._positions_to_dict_list(self.player_start_positions)),
            'player_target_positions': json.dumps(self._positions_to_dict_list(self.player_target_positions)),
            'historical_moves': json.dumps(self._moves_to_dict_list(self.historical_moves)),
            'uuid': self.uuid,
        }
        return pd.DataFrame([data])

    @staticmethod
    def from_dataframe(row: pd.Series) -> 'SimulationData':
        """
        Reconstructs SimulationData from a Pandas DataFrame row.
        """
        player_ids = json.loads(row['player_ids'])
        player_start_positions = SimulationData._dict_list_to_positions(json.loads(row['player_start_positions']))
        player_target_positions = SimulationData._dict_list_to_positions(json.loads(row['player_target_positions']))
        historical_moves = SimulationData._dict_list_to_moves(json.loads(row['historical_moves']))
        return SimulationData(
            player_ids=player_ids,
            player_start_positions=player_start_positions,
            player_target_positions=player_target_positions,
            historical_moves=historical_moves,
            uuid=row.get('uuid', str(uuid.uuid4()))  # Use a new UUID if missing
        )

    @staticmethod
    def _positions_to_dict_list(positions: List[List[Position]]) -> List[List[Dict[str, int]]]:
        """Converts positions to a list of dictionaries with native Python ints."""
        return [
            [{'i': int(pos.i), 'j': int(pos.j)} for pos in pos_list]
            for pos_list in positions
        ]

    @staticmethod
    def _dict_list_to_positions(data: List[List[Dict[str, int]]]) -> List[List[Position]]:
        """Converts a list of dictionaries back to positions."""
        return [
            [Position(i=pos_dict['i'], j=pos_dict['j']) for pos_dict in pos_list]
            for pos_list in data
        ]

    @staticmethod
    def _moves_to_dict_list(moves: List[Move]) -> List[Dict[str, Any]]:
        """Converts moves to a list of dictionaries with native Python types."""
        return [
            {
                'i': int(move.i),
                'j': int(move.j),
                'position': {'i': int(move.position.i), 'j': int(move.position.j)}
            }
            for move in moves
        ]

    @staticmethod
    def _dict_list_to_moves(data: List[Dict[str, Any]]) -> List[Move]:
        """Converts a list of dictionaries back to moves."""
        return [
            Move(
                i=move_dict['i'],
                j=move_dict['j'],
                position=Position(i=move_dict['position']['i'], j=move_dict['position']['j'])
            )
            for move_dict in data
        ]
