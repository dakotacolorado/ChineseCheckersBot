from abc import ABC, abstractmethod
from ..game.ChineseCheckersGame import ChineseCheckersGame

class IChineseCheckersGameEncoder(ABC):
    @abstractmethod
    def encode(self, game: ChineseCheckersGame) -> any:
        """
        Encodes a single game state into a format suitable for ML models.

        Args:
            game (ChineseCheckersGame): The game state to encode.

        Returns:
            Encoded data in the desired format.
        """
        pass
