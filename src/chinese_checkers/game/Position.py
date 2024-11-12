from typing import Tuple, Dict

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

    def to_dict(self) -> Dict[str, int]:
        return {'i': self.i, 'j': self.j}

    @staticmethod
    def from_dict(data: Dict[str, int]) -> "Position":
        return Position(data['i'], data['j'])
