from typing import Tuple

import torch

from .IExperienceEncoder import IExperienceEncoder
from ..ITensorEncoder import ITensorEncoder
from ..game import IChineseCheckersGameEncoder
from ..move import IMoveEncoder
from ...experience import Experience


class ExperienceEncoder(IExperienceEncoder, ITensorEncoder[Experience, torch.Tensor]):
    """
    Encodes an experience tuple into a format suitable for a DQN model.

    Uses `IChineseCheckersGameEncoder` for encoding game states and `IMoveEncoder` for encoding moves.
    """

    def __init__(
            self,
            game_encoder: IChineseCheckersGameEncoder,
            move_encoder: IMoveEncoder
    ):
        self.game_encoder = game_encoder
        self.move_encoder = move_encoder

    @property
    def shape(self) -> Tuple[int, ...]:
        """
        Returns the expected shape of the encoded experience output.

        This combines the shapes of:
        - The game encoder's state encoding
        - The move encoder's action encoding
        """
        return (
            *self.game_encoder.shape,  # Shape of encoded game state
            *self.move_encoder.shape  # Shape of encoded action
        )

    def encode(self, experience: Experience) -> Tuple[torch.Tensor, torch.Tensor, float, torch.Tensor, bool]:
        """
        Encodes an `Experience` instance into DQN-compatible format.

        Args:
            experience (Experience): The experience to encode.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, float, torch.Tensor, bool]: Encoded experience tuple with:
                - Encoded current state (torch.Tensor)
                - Encoded action (torch.Tensor)
                - Reward (float)
                - Encoded next state (torch.Tensor)
                - Done flag (bool)
        """
        # Encode the current and next game states as tensors
        encoded_state = self.game_encoder.encode(experience.state)
        encoded_next_state = self.game_encoder.encode(experience.next_state)

        # Encode the action as a tensor
        encoded_action = self.move_encoder.encode(experience.action)

        # Return the encoded experience tuple
        return encoded_state, encoded_action, experience.reward, encoded_next_state, experience.done
