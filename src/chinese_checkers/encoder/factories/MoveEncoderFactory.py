from typing import Dict, Callable

from .IFactory import IFactory
from ..implementations.BasicMoveEncoder import BasicMoveEncoder
from ..interfaces.IMoveEncoder import IMoveEncoder


class MoveEncoderFactory(IFactory[IMoveEncoder]):

    @property
    def _encoders(self) -> Dict[str, Callable[[], IMoveEncoder]]:
        return {
            "basic_move_encoder": lambda: BasicMoveEncoder(),
        }
