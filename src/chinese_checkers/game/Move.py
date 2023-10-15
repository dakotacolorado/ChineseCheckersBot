from src.chinese_checkers.geometry.Vector import Vector
from .Position import Position


class Move(Vector):

    def __init__(self, i: int, j: int, position: Position):
        self.i = i
        self.j = j
        self.position = position

    def apply(self) -> Position:
        return Position(self.i + self.position.i, self.j + self.position.j)

    def __eq__(self, other: 'Move') -> bool:
        return super().__eq__(other) and self.position == other.position

    def __hash__(self):
        return hash((super().__hash__(), self.position.__hash__()))

    def __repr__(self):
        return f"Move({self.i}, {self.j}), Position{self.position}"
