import logging
from typing import Tuple
import numpy as np
import torch
from ...simulation import GameSimulation
from .ISimulationRewardEncoder import ISimulationRewardEncoder


class DistanceToWinRewardStrategyEncoder(ISimulationRewardEncoder):
    """
    Reward strategy encoder that calculates a reward based on the speed of winning or losing the game,
    returning a tensor of rewards for each turn up to max_game_length.
    """

    logger = logging.getLogger(__name__)

    @property
    def shape(self) -> Tuple[int, ...]:
        """
        Returns the shape of the output reward tensor, which is (max_game_length, 1).
        """
        return (None, 1)

    def encode(self, simulation: GameSimulation) -> torch.Tensor:
        """
        Calculates a reward based on whether the current player is winning or losing, with
        a higher reward given for winning quickly. Returns a reward tensor with a shape of (max_game_length, 1),
        where each entry represents the reward at a specific turn.

        Args:
            simulation (GameSimulation): The simulation state to evaluate for reward calculation.

        Returns:
            torch.Tensor: A tensor of shape (max_game_length, 1) representing the reward for each turn.
        """
        winning_player = simulation.metadata.winning_player
        current_player_id = simulation.data.player_ids[0]  # Assuming player_ids[0] represents the current player

        max_game_length = simulation.metadata.max_game_length
        turn_count = len(simulation.data.historical_moves)

        self.logger.debug(f"Encoding rewards for GameSimulation with max_game_length={max_game_length}")
        self.logger.debug(f"Winning player: {winning_player}, Current player ID: {current_player_id}")
        self.logger.debug(f"Total turns allowed: {max_game_length}, Current turn count: {turn_count}")

        reward_matrix = np.zeros((max_game_length, 1), dtype=np.float32)

        for turn_index in range(turn_count):
            if winning_player == current_player_id:
                reward = (turn_index + 1) / max_game_length
                self.logger.debug(f"Turn {turn_index}: Positive reward {reward} (win detected)")
            else:
                reward = -((turn_index + 1) / max_game_length)
                self.logger.debug(f"Turn {turn_index}: Negative reward {reward} (loss detected)")

            reward_matrix[turn_index, 0] = reward

        reward_tensor = torch.tensor(reward_matrix, dtype=torch.float32)
        self.logger.debug(f"Final reward tensor: {reward_tensor}")

        return reward_tensor
