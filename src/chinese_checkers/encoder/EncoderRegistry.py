from typing import Callable, Dict
from .EncoderMetadata import EncoderMetadata
from .factories import MoveEncoderFactory, ChineseCheckersGameEncoderFactory, ExperienceEncoderFactory
from .interfaces import IEncoder


class EncoderRegistry:

    @staticmethod
    def get_encoder(encoder_metadata: EncoderMetadata) -> IEncoder:
        """
        Retrieves an encoder instance based on the provided encoder metadata.

        Args:
            encoder_metadata (EncoderMetadata): The metadata that specifies the encoder configuration.

        Returns:
            IEncoder: An instance of the encoder corresponding to the metadata.

        Raises:
            ValueError: If no matching encoder is found in the registry.
        """

        # Map of encoders using metadata-based names
        encoders: Dict[str, Callable[[], IEncoder]] = {
            "experience-encoder-v0.0.1": lambda : EncoderRegistry._create_experience_encoder(board_size=encoder_metadata.board_size),
            # Add more encoder configurations here as needed
        }

        encoder_factory = encoders.get(encoder_metadata.encoder_name)

        if not encoder_factory:
            raise ValueError(f"Encoder with metadata '{encoder_metadata}' is not available.")

        return encoder_factory()

    @staticmethod
    def _create_experience_encoder(board_size: int) -> IEncoder:
        move_encoder_factory = MoveEncoderFactory()
        game_encoder_factory = ChineseCheckersGameEncoderFactory(
            move_encoder_factory=move_encoder_factory,
            board_size=board_size
        )
        return ExperienceEncoderFactory(
            game_encoder_factory=game_encoder_factory,
            move_encoder_factory=move_encoder_factory
        ).create("v0.0.1")
