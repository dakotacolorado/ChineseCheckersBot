import logging
from typing import Tuple

import numpy as np
import torch

from chinese_checkers.simulation import GameSimulation

class DistanceToWinRewardSimulationEncoder(ISimulationEncoder):
    """
    Reward strategy encode that calculates a reward based on the speed of winning or losing the game,
    returning a tensor of rewards for each turn up to max_game_length, with an additional dimension representing
    the count of overlapping positions with the target positions.
    """

    @property
    def shape(self) -> Tuple[int, ...]:
        return (2,)

    logger = logging.getLogger(__name__)

    @property
    def cols(self) -> int:
        """Returns the number of columns in the encode output, which is 2 (reward and overlap count)."""
        return 2

    def encode(self, simulation: GameSimulation) -> torch.Tensor:
        """
        Encodes the reward and position overlap count for each turn in the game simulation.

        Args:
            simulation (GameSimulation): The simulation state to evaluate for reward calculation.

        Returns:
            torch.Tensor: A tensor of shape (max_game_length, 2) representing the reward and position overlap count.
        """
        max_game_length = simulation.metadata.max_game_length
        turn_count = len(simulation.data.historical_moves)

        self.logger.debug(f"Encoding rewards for GameSimulation with max_game_length={max_game_length}")

        reward_matrix = np.zeros((max_game_length, self.cols), dtype=np.float32)

        for turn_index in range(turn_count):
            reward = self._calculate_turn_reward(simulation, turn_index)
            overlap_count = self._calculate_position_overlap(simulation)

            reward_matrix[turn_index, 0] = reward
            reward_matrix[turn_index, 1] = overlap_count

        reward_tensor = torch.tensor(reward_matrix, dtype=torch.float32)
        self.logger.debug(f"Final reward tensor: {reward_tensor}")

        return reward_tensor

    def _calculate_turn_reward(self, simulation: GameSimulation, turn_index: int) -> float:
        """
        Calculates the reward for a specific turn based on the game's winning state.

        Args:
            simulation (GameSimulation): The game simulation to evaluate.
            turn_index (int): The index of the current turn.

        Returns:
            float: The calculated reward for the current turn.
        """
        winning_player = simulation.metadata.winning_player
        current_player_id = simulation.data.player_ids[0]

        if winning_player == current_player_id:
            reward = (turn_index + 1) / simulation.metadata.max_game_length
            self.logger.debug(f"Turn {turn_index}: Positive reward {reward} (win detected)")
        else:
            reward = -((turn_index + 1) / simulation.metadata.max_game_length)
            self.logger.debug(f"Turn {turn_index}: Negative reward {reward} (loss detected)")

        return reward

    def _calculate_position_overlap(self, simulation: GameSimulation) -> int:
        """
        Calculates the overlap count between the current and target positions for the player.

        Args:
            simulation (GameSimulation): The game simulation containing position data.

        Returns:
            int: The number of positions that overlap with the target positions.
        """
        current_positions = simulation.data.player_start_positions[0]
        target_positions = simulation.data.player_target_positions[0]
        overlap_count = sum(1 for pos in current_positions if pos in target_positions)
        self.logger.debug(f"Overlap count: {overlap_count}")

        return overlap_count
