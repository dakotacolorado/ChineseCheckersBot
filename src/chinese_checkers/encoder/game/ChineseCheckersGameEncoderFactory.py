from typing import Dict, Callable

from .GridPositionTargetEncoder import GridPositionTargetEncoder
from .IChineseCheckersGameEncoder import IChineseCheckersGameEncoder
from .SpatialBoardMetricsEncoder import SpatialBoardMetricsEncoder
from .SpatialBoardStateEncoder import SpatialBoardStateEncoder
from ..IFactory import IFactory
from ..move.MoveEncoderFactory import MoveEncoderFactory


class ChineseCheckersGameEncoderFactory(IFactory[IChineseCheckersGameEncoder]):

    def __init__(
            self,
            move_encoder_factory: MoveEncoderFactory,
            board_size: int,
    ):
        self.move_encoder_factory = move_encoder_factory
        self.board_size = board_size

    @property
    def _encoders(self) -> Dict[str, Callable[[], IChineseCheckersGameEncoder]]:
        return {
            "grid_position_250": lambda: GridPositionTargetEncoder(
                move_encoder=self.move_encoder_factory.create("basic_move_encoder"),
                max_moves=250,
                board_size=self.board_size
            ),
            "grid_position_100": lambda: GridPositionTargetEncoder(
                move_encoder=self.move_encoder_factory.create("basic_move_encoder"),
                max_moves=100,
                board_size=self.board_size
            ),
            "spatial_board_state": lambda: SpatialBoardStateEncoder(
                board_size=self.board_size
            ),
            "spatial_board_metrics": lambda: SpatialBoardMetricsEncoder(
                board_size=self.board_size
            ),
        }
