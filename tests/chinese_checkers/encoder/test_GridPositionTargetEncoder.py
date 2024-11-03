import unittest
from unittest.mock import patch
import numpy as np
from src.chinese_checkers.geometry.Hexagram import Hexagram
from src.chinese_checkers.game.ChineseCheckersGame import ChineseCheckersGame
from src.chinese_checkers.game.Player import Player
from src.chinese_checkers.game.Position import Position
from src.chinese_checkers.game.Move import Move
from src.chinese_checkers.encoder.GridPositionTargetEncoder import GridPositionTargetEncoder


class TestGridPositionTargetEncoder(unittest.TestCase):

    def setUp(self):
        # Initialize the encoder with a maximum of 3 moves to encode
        self.encoder = GridPositionTargetEncoder(max_moves=3)

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
        self.assertEqual(calculated_dim, expected_dim, "Board dimension calculation mismatch.")

    def test_encode_current_player(self):
        # Expected dimension based on hexagram points
        board_dim = self.encoder._calculate_board_dim(self.hexagram)
        current_player_matrix = self.encoder._encode_current_player(self.game, board_dim)

        # Verify encoded positions of the current player
        for position in self.players[0].positions:
            x, y = position.i + board_dim // 2, position.j + board_dim // 2
            self.assertEqual(current_player_matrix[x, y], 1, "Current player encoding mismatch.")

    def test_encode_opponents(self):
        # Expected dimension based on hexagram points
        board_dim = self.encoder._calculate_board_dim(self.hexagram)
        opponent_matrix = self.encoder._encode_opponents(self.game, board_dim)

        # Verify encoded positions of the opponents
        for position in self.players[1].positions:
            x, y = position.i + board_dim // 2, position.j + board_dim // 2
            self.assertEqual(opponent_matrix[x, y], 1, "Opponent encoding mismatch.")

    def test_encode_player_id(self):
        # One-hot encoding test for current player ID
        player_id_one_hot = self.encoder._encode_player_id(self.game)
        expected_id = int(self.players[0].player_id)

        for i in range(6):
            expected_value = 1 if i == expected_id else 0
            self.assertEqual(player_id_one_hot[i], expected_value, f"Player ID encoding mismatch at index {i}.")

    @patch.object(ChineseCheckersGame, 'get_next_moves')
    def test_encode_moves(self, mock_get_next_moves):
        # Define moves to return from mock
        moves = [
            Move(0, 0, Position(1, 1)),
            Move(1, 1, Position(2, 2))
        ]
        mock_get_next_moves.return_value = moves  # Mock get_next_moves to return predefined moves

        encoded_moves = self.encoder._encode_moves(self.game)

        # Verify encoded moves and padding
        self.assertEqual(len(encoded_moves), 3, "Move encoding length mismatch.")
        for i, move in enumerate(moves):
            encoded_move = self.encoder.move_encoder.encode(move)
            np.testing.assert_array_equal(
                encoded_moves[i], encoded_move, f"Move encoding mismatch for move {i}."
            )

        # Check padding with dummy actions
        dummy_action = self.encoder.move_encoder.create_dummy_action()
        for i in range(len(moves), 3):
            np.testing.assert_array_equal(
                encoded_moves[i], dummy_action, f"Dummy action mismatch at index {i}."
            )

    def test_encode_full_state(self):
        # Expected dimension based on hexagram points
        board_dim = self.encoder._calculate_board_dim(self.hexagram)

        # Encode the full game state
        encoded_state = self.encoder.encode(self.game)

        # Calculate expected encoded state length
        expected_length = board_dim * board_dim * 2 + 6 + 3 * 4  # board matrices, one-hot ID, encoded moves
        self.assertEqual(encoded_state.shape[0], expected_length, "Encoded state length mismatch.")
