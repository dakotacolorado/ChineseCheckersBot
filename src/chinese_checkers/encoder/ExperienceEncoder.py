import concurrent
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple, List

import numpy as np
from tqdm import tqdm

from .ChineseCheckersGameEncoderFactory import ChineseCheckersGameEncoderFactory
from .MoveEncoder import MoveEncoder
from ..experience import Experience


class ExperienceEncoder:
    """
    Encodes an experience tuple into a format suitable for a DQN model.

    Uses `GridPositionTargetEncoder` for encoding game states and `MoveEncoder` for encoding moves.
    """

    def __init__(self, game_encoder: str = "grid_position_target", max_moves: int = 100):
        self.game_encoder = ChineseCheckersGameEncoderFactory.create(encoder_name=game_encoder, max_moves=max_moves)
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

    def batch_encode(self, experiences: List, batch_size: int) -> List:
        """
        Encodes experiences in batches and parallelizes the process.

        Args:
            experiences (List): The list of experiences to encode.
            batch_size (int): The number of experiences to process in a single batch.

        Returns:
            List: A list of encoded experiences.
        """
        encoded_experiences = []

        # Split experiences into batches
        batches = [experiences[i:i + batch_size] for i in range(0, len(experiences), batch_size)]

        with ThreadPoolExecutor() as executor:
            # Run encoding in parallel and track progress with tqdm
            futures = {executor.submit(self._encode_batch, batch): batch for batch in batches}
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(batches), desc="Encoding batches",
                               unit="batch"):
                try:
                    encoded_experiences.extend(future.result())
                except Exception as ex:
                    print(f"Error encountered while encoding a batch: {ex}")

        return encoded_experiences

    def _encode_batch(self, batch: List) -> List:
        """
        Encodes a single batch of experiences.

        Args:
            batch (List): A batch of experiences to encode.

        Returns:
            List: A list of encoded experiences.
        """
        return [self.encode(experience) for experience in batch]