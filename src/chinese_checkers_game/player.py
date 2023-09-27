from typing import List

from src.chinese_checkers_game.position import Position


class Player:

    def __init__(self, positions: List[Position]):
        self.positions = positions

    def applyMove(self, move):
        return Player([move.apply() if position == move.position else position for position in self.positions])
