from typing import Tuple, Dict
import pyarrow as pa

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
        return {'i': int(self.i), 'j': int(self.j)}

    @staticmethod
    def from_dict(data: Dict[str, int]) -> "Position":
        return Position(data['i'], data['j'])

    def to_struct(self) -> Dict[str, int]:
        return {'i': self.i, 'j': self.j}

    @staticmethod
    def struct_type() -> pa.DataType:
        return pa.struct([
            ('i', pa.int64()),
            ('j', pa.int64())
        ])

    @staticmethod
    def from_struct(struct: pa.StructScalar) -> 'Position':
        return Position(i=struct['i'].as_py(), j=struct['j'].as_py())
