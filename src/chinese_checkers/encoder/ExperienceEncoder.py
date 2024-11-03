from typing import Tuple, Any
import numpy as np
from numpy import ndarray

from .ChineseCheckersGameEncoderFactory import ChineseCheckersGameEncoderFactory
from .MoveEncoder import MoveEncoder
from ..experience import Experience

class ExperienceEncoder:
    """
    Encodes an experience tuple into a format suitable for a DQN model.

    Uses `GridPositionTargetEncoder` for encoding game states and `MoveEncoder` for encoding moves.
    """

    def __init__(self, game_encoder: str = "grid_position_target"):
        self.game_encoder = ChineseCheckersGameEncoderFactory.create(game_encoder)
        self.move_encoder = MoveEncoder()

    def encode(self, experience: Experience) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]:
        """
        Encodes an `Experience` instance into DQN-compatible format.

        Args:
            experience (Experience): The experience to encode.

        Returns:
            Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]: Encoded experience tuple with:
                - Encoded current state (np.ndarray)
                - Encoded action (np.ndarray)
                - Reward (float)
                - Encoded next state (np.ndarray)
                - Done flag (bool)
        """
        # Encode current and next states as single arrays
        encoded_state = self.game_encoder.encode(experience.state)
        encoded_next_state = self.game_encoder.encode(experience.next_state)

        # Encode action
        encoded_action = self.move_encoder.encode(experience.action)

        # Package into a tuple
        return encoded_state, encoded_action, experience.reward, encoded_next_state, experience.done
