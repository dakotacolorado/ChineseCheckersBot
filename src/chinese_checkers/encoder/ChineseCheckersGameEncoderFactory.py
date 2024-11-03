from .IChineseCheckersGameEncoder import IChineseCheckersGameEncoder
from .GridPositionTargetEncoder import GridPositionTargetEncoder

class ChineseCheckersGameEncoderFactory:
    """
    Factory for creating instances of IChineseCheckersGameEncoder based on a string identifier.
    """

    _encoders = {
        "grid_position_target": GridPositionTargetEncoder,
        # Add other encoders here if needed, e.g. "another_encoder": AnotherEncoder
    }

    @staticmethod
    def create(encoder_name: str, max_moves: int) -> IChineseCheckersGameEncoder:
        """
        Creates an instance of an encoder based on the specified name.

        Args:
            encoder_name (str): The name of the encoder to create.
            max_moves (int): Maximum number of turns to encode in the state representation.

        Returns:
            IChineseCheckersGameEncoder: An instance of the requested encoder.

        Raises:
            ValueError: If the specified encoder name is not available.
        """
        encoder_class = ChineseCheckersGameEncoderFactory._encoders.get(encoder_name)
        if not encoder_class:
            raise ValueError(f"Encoder strategy '{encoder_name}' is not available.")
        return encoder_class(max_moves=max_moves)
