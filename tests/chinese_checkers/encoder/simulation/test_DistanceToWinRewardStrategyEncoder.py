import unittest
import numpy as np
import logging
from unittest.mock import MagicMock
from src.chinese_checkers.encoder.simulation.DistanceToWinRewardStrategyEncoder import DistanceToWinRewardStrategyEncoder
from src.chinese_checkers.simulation import GameSimulation, SimulationMetadata, SimulationData
from src.chinese_checkers.game import Position, Move


class TestDistanceToWinRewardStrategyEncoder(unittest.TestCase):

    # @classmethod
    # def setUpClass(cls):
    #     logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

    def setUp(self):
        self.max_game_length = 10
        self.winning_player = "player_1"
        self.current_player = "player_1"
        self.player_count = 2

    def create_simulation(self, turn_count, winning=True):
        player_ids = ["player_1", "player_2"]
        player_start_positions = [[Position(0, 0)], [Position(1, 1)]]
        player_target_positions = [[Position(5, 5)], [Position(4, 4)]]
        historical_moves = [MagicMock(spec=Move) for _ in range(turn_count)]

        metadata = SimulationMetadata(
            player_count=self.player_count,
            board_size=5,
            max_game_length=self.max_game_length,
            winning_player=self.winning_player if winning else "player_2",
            name="test_simulation",
            version="1.0"
        )
        data = SimulationData(
            player_ids=player_ids,
            player_start_positions=player_start_positions,
            player_target_positions=player_target_positions,
            historical_moves=historical_moves
        )
        return GameSimulation(metadata=metadata, data=data)

    def test_encode_reward_for_winning_player(self):
        encoder = DistanceToWinRewardStrategyEncoder()
        simulation = self.create_simulation(turn_count=5, winning=True)

        reward_tensor = encoder.encode(simulation)
        expected_rewards = [(i + 1) / self.max_game_length for i in range(5)]
        expected_matrix = np.zeros((self.max_game_length, 1), dtype=np.float32)
        expected_matrix[:5, 0] = expected_rewards

        self.assertEqual(reward_tensor.shape, (self.max_game_length, 1))
        np.testing.assert_almost_equal(reward_tensor.numpy(), expected_matrix, decimal=5)

    def test_encode_reward_for_losing_player(self):
        encoder = DistanceToWinRewardStrategyEncoder()
        simulation = self.create_simulation(turn_count=5, winning=False)

        reward_tensor = encoder.encode(simulation)
        expected_rewards = [-(i + 1) / self.max_game_length for i in range(5)]
        expected_matrix = np.zeros((self.max_game_length, 1), dtype=np.float32)
        expected_matrix[:5, 0] = expected_rewards

        self.assertEqual(reward_tensor.shape, (self.max_game_length, 1))
        np.testing.assert_almost_equal(reward_tensor.numpy(), expected_matrix, decimal=5)

    def test_reward_matrix_padding_for_early_game_end(self):
        encoder = DistanceToWinRewardStrategyEncoder()
        simulation = self.create_simulation(turn_count=3, winning=True)

        reward_tensor = encoder.encode(simulation)
        expected_rewards = [(i + 1) / self.max_game_length for i in range(3)]
        expected_matrix = np.zeros((self.max_game_length, 1), dtype=np.float32)
        expected_matrix[:3, 0] = expected_rewards

        self.assertEqual(reward_tensor.shape, (self.max_game_length, 1))
        np.testing.assert_almost_equal(reward_tensor.numpy(), expected_matrix, decimal=5)
