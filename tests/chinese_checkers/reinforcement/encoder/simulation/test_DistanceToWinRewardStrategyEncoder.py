import unittest
import numpy as np
from unittest.mock import MagicMock
from chinese_checkers.reinforcement.encode.DistanceToWinRewardSimulationEncoder import DistanceToWinRewardSimulationEncoder
from src.chinese_checkers.simulation import GameSimulation, SimulationMetadata, SimulationData
from src.chinese_checkers.game import Position, Move


class TestDistanceToWinRewardStrategyEncoder(unittest.TestCase):

    def setUp(self):
        self.max_game_length = 10
        self.winning_player = "player_1"
        self.current_player = "player_1"
        self.player_count = 2

    def create_simulation(self, turn_count, winning=True, overlaps=0):
        player_ids = ["player_1", "player_2"]
        player_start_positions = [
            [Position(0, 0)] * overlaps + [Position(1, 1)] * (1 - overlaps),  # Control overlap
            [Position(1, 1)]
        ]
        player_target_positions = [[Position(0, 0)], [Position(4, 4)]]
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

    def test_encode_reward_for_winning_player_with_overlap(self):
        encoder = DistanceToWinRewardSimulationEncoder()
        simulation = self.create_simulation(turn_count=5, winning=True, overlaps=1)

        reward_tensor = encoder.encode(simulation)
        expected_rewards = [(i + 1) / self.max_game_length for i in range(5)]
        expected_matrix = np.zeros((self.max_game_length, 2), dtype=np.float32)
        expected_matrix[:5, 0] = expected_rewards
        expected_matrix[:5, 1] = 1  # Expecting overlap count of 1 for each turn

        self.assertEqual(reward_tensor.shape, (self.max_game_length, 2))
        np.testing.assert_almost_equal(reward_tensor.numpy(), expected_matrix, decimal=5)

    def test_encode_reward_for_losing_player_with_overlap(self):
        encoder = DistanceToWinRewardSimulationEncoder()
        simulation = self.create_simulation(turn_count=5, winning=False, overlaps=1)

        reward_tensor = encoder.encode(simulation)
        expected_rewards = [-(i + 1) / self.max_game_length for i in range(5)]
        expected_matrix = np.zeros((self.max_game_length, 2), dtype=np.float32)
        expected_matrix[:5, 0] = expected_rewards
        expected_matrix[:5, 1] = 1  # Expecting overlap count of 1 for each turn

        self.assertEqual(reward_tensor.shape, (self.max_game_length, 2))
        np.testing.assert_almost_equal(reward_tensor.numpy(), expected_matrix, decimal=5)

    def test_reward_matrix_padding_for_early_game_end_with_no_overlap(self):
        encoder = DistanceToWinRewardSimulationEncoder()
        simulation = self.create_simulation(turn_count=3, winning=True, overlaps=0)

        reward_tensor = encoder.encode(simulation)
        expected_rewards = [(i + 1) / self.max_game_length for i in range(3)]
        expected_matrix = np.zeros((self.max_game_length, 2), dtype=np.float32)
        expected_matrix[:3, 0] = expected_rewards
        expected_matrix[:3, 1] = 0  # Expecting overlap count of 0 for each turn

        self.assertEqual(reward_tensor.shape, (self.max_game_length, 2))
        np.testing.assert_almost_equal(reward_tensor.numpy(), expected_matrix, decimal=5)

    def test_encode_with_variable_overlap_count(self):
        encoder = DistanceToWinRewardSimulationEncoder()
        simulation = self.create_simulation(turn_count=4, winning=True, overlaps=2)

        reward_tensor = encoder.encode(simulation)
        expected_rewards = [(i + 1) / self.max_game_length for i in range(4)]
        expected_matrix = np.zeros((self.max_game_length, 2), dtype=np.float32)
        expected_matrix[:4, 0] = expected_rewards
        expected_matrix[:4, 1] = 2  # Expecting overlap count of 2 for each turn

        self.assertEqual(reward_tensor.shape, (self.max_game_length, 2))
        np.testing.assert_almost_equal(reward_tensor.numpy(), expected_matrix, decimal=5)

