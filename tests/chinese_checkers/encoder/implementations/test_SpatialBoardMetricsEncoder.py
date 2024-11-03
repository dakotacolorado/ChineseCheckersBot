import unittest
from unittest.mock import MagicMock

from src.chinese_checkers.encoder.implementations import SpatialBoardMetricsEncoder
from src.chinese_checkers.game import ChineseCheckersGame, Player, Position


class TestSpatialBoardMetricsEncoder(unittest.TestCase):
    def setUp(self):
        self.board_size = 4
        self.encoder = SpatialBoardMetricsEncoder(board_size=self.board_size)

    def test_calculate_centroid_distance(self):
        player = MagicMock(spec=Player)
        player.positions = [Position(0, 0), Position(2, 2)]
        player.target_positions = [Position(4, 4), Position(6, 6)]

        result = self.encoder._calculate_centroid_distance(player, self.board_size)
        expected_distance = 1.4142135623730951  # corrected normalized value
        self.assertAlmostEqual(result, expected_distance, places=5,
                               msg="Centroid distance between current and target positions should be correctly normalized by board size.")

    def test_calculate_move_sizes(self):
        game = MagicMock(spec=ChineseCheckersGame)
        move1 = MagicMock()
        move1.apply.return_value = Position(3, 3)
        move1.position = Position(1, 1)
        move2 = MagicMock()
        move2.apply.return_value = Position(4, 4)
        move2.position = Position(2, 2)
        game.get_next_moves.return_value = [move1, move2]

        avg_move_size, max_move_size = self.encoder._calculate_move_sizes(game, self.board_size)
        expected_avg = (2.8284271247461903 + 2.8284271247461903) / 2 / self.board_size
        expected_max = 2.8284271247461903 / self.board_size

        self.assertAlmostEqual(avg_move_size, expected_avg, places=5,
                               msg="Average move size should be correctly calculated and normalized.")
        self.assertAlmostEqual(max_move_size, expected_max, places=5,
                               msg="Max move size should be the largest move distance and correctly normalized.")

    def test_calculate_farthest_position(self):
        player = MagicMock(spec=Player)
        player.positions = [Position(0, 0), Position(3, 4), Position(1, 1)]
        player.target_positions = [Position(5, 5)]

        result = self.encoder._calculate_farthest_position(player, self.board_size)
        expected_distance = 1.7677669529663689  # corrected normalized value
        self.assertAlmostEqual(result, expected_distance, places=5,
                               msg="Farthest position distance from target centroid should be correctly normalized.")

    def test_calculate_shortest_position(self):
        player = MagicMock(spec=Player)
        player.positions = [Position(0, 0), Position(3, 4), Position(2, 2)]
        player.target_positions = [Position(5, 5)]

        result = self.encoder._calculate_shortest_position(player, self.board_size)
        expected_distance = 0.5590169943749475  # corrected normalized value
        self.assertAlmostEqual(result, expected_distance, places=5,
                               msg="Shortest position distance from target centroid should be correctly normalized.")

    def test_calculate_completed_positions(self):
        player = MagicMock(spec=Player)
        player.positions = [Position(0, 0), Position(1, 1), Position(2, 2)]
        player.target_positions = [Position(1, 1), Position(2, 2), Position(3, 3)]

        result = self.encoder._calculate_completed_positions(player)
        expected_count = 2
        self.assertEqual(result, expected_count,
                         msg="Completed positions count should match the intersection of current and target positions.")

    def test_encode(self):
        game = MagicMock(spec=ChineseCheckersGame)
        current_player = MagicMock(spec=Player)

        # Adjust target_positions to match some of current_player's positions
        current_player.positions = [Position(0, 0), Position(2, 2)]
        current_player.target_positions = [Position(0, 0), Position(2, 2)]  # Overlaps for testing

        game.get_current_player.return_value = current_player

        move1 = MagicMock()
        move1.apply.return_value = Position(3, 3)
        move1.position = Position(1, 1)
        move2 = MagicMock()
        move2.apply.return_value = Position(4, 4)
        move2.position = Position(2, 2)
        game.get_next_moves.return_value = [move1, move2]

        encoded_tensor = self.encoder.encode(game)
        self.assertEqual(encoded_tensor.shape[0], 6,
                         msg="Encoded tensor should have a shape of (6,), with each metric correctly calculated.")

        # Updated expected value to 0.0 due to centroids coinciding
        self.assertAlmostEqual(encoded_tensor[0].item(), 0.0, places=5,
                               msg="Centroid distance should match expected normalized value.")

        self.assertAlmostEqual(encoded_tensor[1].item(),
                               (2.8284271247461903 + 2.8284271247461903) / 2 / self.board_size, places=5,
                               msg="Average move size should match expected normalized value.")
        self.assertAlmostEqual(encoded_tensor[2].item(), 2.8284271247461903 / self.board_size, places=5,
                               msg="Max move size should match expected normalized value.")
        self.assertAlmostEqual(encoded_tensor[3].item(), 0.3535533905932738, places=5,
                               msg="Farthest position distance should match expected normalized value.")

        self.assertAlmostEqual(encoded_tensor[4].item(), 0.3535533905932738, places=5,
                               msg="Shortest position distance should match expected normalized value.")

        # Update expectation to 2 since there are 2 overlapping positions
        self.assertEqual(encoded_tensor[5].item(), 2,
                         msg="Completed positions count should match the expected intersection count.")
