import unittest
import numpy as np
from src.chinese_checkers.geometry.Hexagram import Hexagram
from src.chinese_checkers.game.ChineseCheckersGame import ChineseCheckersGame
from src.chinese_checkers.game.Player import Player
from src.chinese_checkers.game.Position import Position
from src.chinese_checkers.encoder.GridPositionTargetEncoder import GridPositionTargetEncoder


class TestGridPositionTargetEncoder(unittest.TestCase):

    def setUp(self):
        # Initialize the encoder
        self.encoder = GridPositionTargetEncoder()

        # Set up a hexagram with radius 4 for testing board dimension calculation
        self.hexagram = Hexagram(radius=4)

        # Create a sample ChineseCheckersGame instance with mock players and positions
        self.players = [
            Player([Position(0, 0)], [Position(3, 3)], "1"),
            Player([Position(1, 1)], [Position(-3, -3)], "2")
        ]
        self.game = ChineseCheckersGame(players=self.players, board=self.hexagram)

    def test_calculate_board_dim(self):
        # Expected dimension based on hexagram points with radius 4
        expected_dim = max(
            max(p.i for p in self.hexagram.hexagram_points) - min(p.i for p in self.hexagram.hexagram_points),
            max(p.j for p in self.hexagram.hexagram_points) - min(p.j for p in self.hexagram.hexagram_points)
        ) + 1  # +1 to account for zero indexing

        calculated_dim = self.encoder._calculate_board_dim(self.hexagram)

        self.assertEqual(
            calculated_dim, expected_dim,
            f"Expected board dimension to be {expected_dim}, but got {calculated_dim}. "
            "Ensure that _calculate_board_dim accounts for the full range of hexagram points."
        )

    def test_encode_game_state(self):
        # Expected dimension based on hexagram points
        board_dim = self.encoder._calculate_board_dim(self.hexagram)

        # Encode the game state
        encoded_state = self.encoder.encode(self.game)

        # Calculate the expected length of the encoded state
        expected_length = board_dim * board_dim * 2 + 6  # two matrices (current player + opponent) and one-hot vector

        # Check the length of the encoded output
        self.assertEqual(
            encoded_state.shape[0], expected_length,
            f"Expected encoded state length to be {expected_length}, but got {encoded_state.shape[0]}."
        )

        # Verify encoded current player positions
        for position in self.players[0].positions:
            x, y = position.i + board_dim // 2, position.j + board_dim // 2
            index = x * board_dim + y
            self.assertEqual(
                encoded_state[index], 1,
                f"Expected position index {index} in current player matrix to be 1, but found {encoded_state[index]}."
            )

        # Verify encoded opponent positions
        opponent_start = board_dim * board_dim  # Start of opponent matrix in flattened array
        for position in self.players[1].positions:
            x, y = position.i + board_dim // 2, position.j + board_dim // 2
            index = opponent_start + x * board_dim + y
            self.assertEqual(
                encoded_state[index], 1,
                f"Expected position index {index} in opponent matrix to be 1, but found {encoded_state[index]}."
            )

        # Verify the one-hot encoding for current player ID
        player_id_one_hot_start = board_dim * board_dim * 2  # Start of one-hot vector in flattened array
        for i in range(6):
            expected_value = 1 if i == int(self.players[0].player_id) else 0
            self.assertEqual(
                encoded_state[player_id_one_hot_start + i], expected_value,
                f"Expected one-hot encoding at index {player_id_one_hot_start + i} to be {expected_value}, "
                f"but found {encoded_state[player_id_one_hot_start + i]}."
            )
