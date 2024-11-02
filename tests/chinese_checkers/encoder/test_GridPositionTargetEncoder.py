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
        board_state, current_player_index = self.encoder.encode(self.game)

        # Check that the encoded board state has the correct dimensions
        self.assertEqual(
            board_state.shape, (board_dim, board_dim),
            f"Expected encoded board state shape to be {(board_dim, board_dim)}, "
            f"but got {board_state.shape}. Ensure the board dimension is correctly applied in the encoding."
        )

        # Check that the current player indicator matches
        expected_current_player = 1
        self.assertEqual(
            current_player_index, expected_current_player,
            f"Expected current player indicator to be {expected_current_player}, but got {current_player_index}. "
            "Ensure that the encoder correctly identifies the current player."
        )

        # Check encoded positions of players on the grid
        for player_index, player in enumerate(self.game.players, start=1):
            for position in player.positions:
                x, y = position.i + board_dim // 2, position.j + board_dim // 2
                self.assertEqual(
                    board_state[x, y], player_index,
                    f"Expected position ({x}, {y}) to contain player index {player_index}, but found {board_state[x, y]}. "
                    "Verify that player positions are correctly encoded in the grid."
                )

        # Check encoded positions of target locations
        for player_index, player in enumerate(self.game.players, start=1):
            for target_position in player.target_positions:
                x, y = target_position.i + board_dim // 2, target_position.j + board_dim // 2
                self.assertEqual(
                    board_state[x, y], -player_index,
                    f"Expected target position ({x}, {y}) to contain target index {-player_index}, but found {board_state[x, y]}. "
                    "Verify that target positions are correctly encoded with negative player indices in the grid."
                )
