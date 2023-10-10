from typing import List

from .move import Move
from .position import Position


class Player:

    def __init__(
            self,
            positions: List[Position],
            target_positions: List[Position]
    ):
        self.positions = positions
        self.target_positions = target_positions

    def apply_move(self, move: Move):
        return Player([
            move.apply() if position == move.position else position
            for position in self.positions
        ])
