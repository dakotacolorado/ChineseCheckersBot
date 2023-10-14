from typing import List

from .move import Move
from .position import Position


class Player:

    def __init__(
            self,
            positions: List[Position],
            target_positions: List[Position],
            player_id: str
    ):
        self.positions = positions
        self.target_positions = target_positions
        self.player_id = player_id

    def apply_move(self, move: Move):
        return Player(
            [
                move.apply() if position == move.position else position
                for position in self.positions
            ],
            self.target_positions,
            self.player_id
        )

    def __eq__(self, other: 'Player') -> bool:
        return self.player_id == other.player_id

    def has_player_reached_target(self) -> bool:
        return set(self.positions) == set(self.target_positions)
