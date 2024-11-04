import logging
import math
from typing import List
import torch

from chinese_checkers.game import ChineseCheckersGame
from chinese_checkers.geometry.Vector import Vector
from chinese_checkers.simulation import GameSimulation


class DistanceToWinRewardSimulationEncoder:
    logger = logging.getLogger(__name__)

    """
    Reward strategy encoder that calculates a reward based on the distance to the target centroid,
    returning a list of tensors, each of size 1, representing the combined Q-objective for each turn.
    """

    def encode(self, simulation: GameSimulation) -> List[torch.Tensor]:
        max_game_length = simulation.metadata.max_game_length
        game_sequence: List[ChineseCheckersGame] = simulation._to_game_sequence()

        self.logger.debug(f"Encoding rewards for GameSimulation with max_game_length={max_game_length}")

        q_objective_list = []
        for game in game_sequence:
            q_objective = self._calculate_inverse_distance_to_target(game, simulation.metadata.board_size)
            q_objective_tensor = torch.tensor([q_objective], dtype=torch.float32)
            q_objective_list.append(q_objective_tensor)

        self.logger.debug(f"Final Q-objective list: {q_objective_list}")
        return q_objective_list

    def _calculate_inverse_distance_to_target(self, game: ChineseCheckersGame, board_size: int) -> float:
        current_player = game.get_current_player()
        target_positions = current_player.target_positions
        current_positions = current_player.positions

        current_centroid = self._calculate_centroid(current_positions)
        target_centroid = self._calculate_centroid(target_positions)

        distance = current_centroid.distance(target_centroid) / board_size
        self.logger.debug(f"Distance to target centroid: {distance}")

        if distance == 0:
            return 60

        return (1 / distance)

    def _calculate_centroid(self, positions: List[Vector]) -> Vector:
        if not positions:
            return Vector(0.0, 0.0)

        avg_i = sum(pos.i for pos in positions) / len(positions)
        avg_j = sum(pos.j for pos in positions) / len(positions)
        centroid = Vector(avg_i, avg_j)
        self.logger.debug(f"Calculated centroid: {centroid}")
        return centroid

    # Future reward mechanisms (commented out)
    # def _calculate_turn_reward(self, simulation: GameSimulation, turn_index: int) -> float:
    #     winning_player = simulation.metadata.winning_player
    #     current_player_id = simulation.data.player_ids[0]
    #
    #     if winning_player == current_player_id:
    #         reward = (turn_index + 1) / simulation.metadata.max_game_length
    #         self.logger.debug(f"Turn {turn_index}: Positive reward {reward} (win detected)")
    #     else:
    #         reward = -((turn_index + 1) / simulation.metadata.max_game_length)
    #         self.logger.debug(f"Turn {turn_index}: Negative reward {reward} (loss detected)")
    #
    #     return reward

    # def _calculate_position_overlap(self, simulation: GameSimulation) -> int:
    #     current_positions = simulation.data.player_start_positions
    #     target_positions = simulation.data.player_target_positions
    #
    #     overlap_count = sum(1 for pos in current_positions if pos in target_positions)
    #     self.logger.debug(f"Overlap count: {overlap_count}")
    #
    #     return overlap_count
