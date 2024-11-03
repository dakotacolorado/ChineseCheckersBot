from typing import Dict, Callable

from ..IFactory import IFactory
from .BasicMoveEncoder import BasicMoveEncoder
from .IMoveEncoder import IMoveEncoder


class MoveEncoderFactory(IFactory[IMoveEncoder]):

    @property
    def _encoders(self) -> Dict[str, Callable[[], IMoveEncoder]]:
        return {
            "basic_move_encoder": lambda: BasicMoveEncoder(),
        }
