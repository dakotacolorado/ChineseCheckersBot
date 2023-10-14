from src.chinese_checkers.geometry.Vector import Vector
from .Position import Position


class Move(Vector):

    def __init__(self, i: int, j: int, position: Position):
        self.i = i
        self.j = j
        self.position = position

    def apply(self) -> Position:
        return Position(self.i + self.position.i, self.j + self.position.j)

    def __str__(self):
        return f"Move({self.i}, {self.j}), Position{self.position}"

