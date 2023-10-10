from src.chinese_checkers.geometry.vector import Vector
from .position import Position


class Move(Vector):

    def __init__(self, i: int, j: int, position: Position):
        self.i = i
        self.j = j
        self.position = position

    def apply(self) -> Position:
        return Position(self.i + self.position.i, self.j + self.position.j)

