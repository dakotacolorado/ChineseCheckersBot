from typing import Tuple, Dict, Any

from ..geometry.Vector import Vector
from .Position import Position


class Move(Vector):

    def __init__(self, i: int, j: int, position: Position):
        self.i = i
        self.j = j
        self.position = position

    def __eq__(self, other: 'Move') -> bool:
        return super().__eq__(other) and self.position == other.position

    def __hash__(self):
        return hash((super().__hash__(), self.position.__hash__()))

    def __repr__(self):
        return f"Move({self.i}, {self.j}), Position{self.position}"

    def apply(self) -> Position:
        return Position(self.i + self.position.i, self.j + self.position.j)

    def to_tuple(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        return (self.i, self.j), self.position.to_tuple()

    @staticmethod
    def from_tuple(tpl: Tuple[Tuple[int, int], Tuple[int, int]]) -> "Move":
        return Move(tpl[0][0], tpl[0][1], Position.from_tuple(tpl[1]))

    def to_dict(self) -> Dict[str, Any]:
        return {
            'i': self.i,
            'j': self.j,
            'position': self.position.to_dict()
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Move":
        return Move(data['i'], data['j'], Position.from_dict(data['position']))