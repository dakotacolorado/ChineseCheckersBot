from typing import Callable

from .ChineseCheckersGameEncoderFactory import ChineseCheckersGameEncoderFactory
from .IFactory import IFactory
from .MoveEncoderFactory import MoveEncoderFactory
from ..implementations import ExperienceEncoder
from ..interfaces import IExperienceEncoder


class ExperienceEncoderFactory(IFactory[IExperienceEncoder]):

    def __init__(
            self,
            game_encoder_factory: ChineseCheckersGameEncoderFactory,
            move_encoder_factory: MoveEncoderFactory,
    ):
        self.game_encoder_factory = game_encoder_factory
        self.move_encoder_factory = move_encoder_factory

    @property
    def _encoders(self) -> dict[str, Callable[[], ExperienceEncoder]]:
        return {
            "v0.0.1": lambda: ExperienceEncoder(
                game_encoder=self.game_encoder_factory.create("grid_position_250"),
                move_encoder=self.move_encoder_factory.create("basic_move_encoder")
            ),
        }