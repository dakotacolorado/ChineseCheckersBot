from typing import Callable

from .EncoderMetadata import EncoderMetadata
from .factories import MoveEncoderFactory, ChineseCheckersGameEncoderFactory, ExperienceEncoderFactory
from .factories.IFactory import IFactory
from .interfaces import IEncoder


class EncoderFactory(IFactory[IEncoder]):

    def __init__(self, encoder_metadata : EncoderMetadata):
        self.encoder_metadata = encoder_metadata
        self.move_encoder_factory = MoveEncoderFactory()
        self.chinese_checkers_game_encoder_factory = ChineseCheckersGameEncoderFactory(
            move_encoder_factory=self.move_encoder_factory,
            board_size=encoder_metadata.board_size,
        )
        self.experience_encoder_factory = ExperienceEncoderFactory(
            game_encoder_factory=self.chinese_checkers_game_encoder_factory,
            move_encoder_factory=self.move_encoder_factory,
        )

    @property
    def _encoders(self) -> dict[str, Callable[[], IEncoder]]:
        return {
            "v0.0.1": lambda: self.experience_encoder_factory.create("v0.0.1"),
        }


