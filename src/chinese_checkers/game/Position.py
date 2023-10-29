from typing import Tuple

from ..geometry.Vector import Vector


class Position(Vector):
    def __init__(self, i: int, j: int):
        self.i = i
        self.j = j

    def to_tuple(self) -> Tuple[int, int]:
        return self.i, self.j

    @staticmethod
    def from_tuple(tpl: Tuple[int, int]) -> "Position":
        return Position(tpl[0], tpl[1])
