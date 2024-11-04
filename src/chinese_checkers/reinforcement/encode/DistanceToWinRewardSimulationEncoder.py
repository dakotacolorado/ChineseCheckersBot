import logging
from typing import List
import torch
from chinese_checkers.simulation import GameSimulation

class DistanceToWinRewardSimulationEncoder:
    logger = logging.getLogger(__name__)

    """
    Reward strategy encoder that calculates a reward based on the speed of winning or losing the game,
    returning a list of tensors, each of size 2, representing the reward and overlap count for each turn.
    """

    def encode(self, simulation: GameSimulation) -> List[torch.Tensor]:
        """
        Encodes the reward and position overlap count for each turn in the game simulation.

        Args:
            simulation (GameSimulation): The simulation state to evaluate for reward calculation.

        Returns:
            List[torch.Tensor]: A list of tensors, each of size 2, representing the reward and position overlap count.
        """
        max_game_length = simulation.metadata.max_game_length
        turn_count = len(simulation.data.historical_moves)

        self.logger.debug(f"Encoding rewards for GameSimulation with max_game_length={max_game_length}")

        reward_list = []

        for turn_index in range(turn_count):
            reward = self._calculate_turn_reward(simulation, turn_index)
            overlap_count = self._calculate_position_overlap(simulation)

            reward_tensor = torch.tensor([reward, overlap_count], dtype=torch.float32)
            reward_list.append(reward_tensor)

        self.logger.debug(f"Final reward list: {reward_list}")

        return reward_list

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
