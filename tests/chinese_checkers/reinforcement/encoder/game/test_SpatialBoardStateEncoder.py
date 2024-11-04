import unittest
from unittest.mock import MagicMock

from src.chinese_checkers.game import ChineseCheckersGame, Position, Player
from chinese_checkers.reinforcement.encode.SpatialBoardStateEncoder import SpatialBoardStateEncoder


class TestSpatialBoardStateEncoder(unittest.TestCase):
    def setUp(self):
        self.board_size = 4
        self.encoder = SpatialBoardStateEncoder(board_size=self.board_size)

    def test_encode_two_players(self):
        game = MagicMock(spec=ChineseCheckersGame)

        current_player_positions = [Position(1, 1), Position(2, 2)]
        current_player_target_positions = [Position(3, 3), Position(3, 4)]
        other_player_positions = [Position(4, 4), Position(-1, -1)]

        current_player = MagicMock(spec=Player)
        current_player.positions = current_player_positions
        current_player.target_positions = current_player_target_positions
        game.get_current_player.return_value = current_player

        other_player = MagicMock(spec=Player)
        other_player.positions = other_player_positions
        game.get_other_players.return_value = [other_player]

        encoded_tensor = self.encoder.encode(game)

        self.assertEqual(
            encoded_tensor.shape,
            (3, self.encoder.board_dim, self.encoder.board_dim),
            "The encoded tensor shape should be (3, board_dim, board_dim) with three channels for player data."
        )

        self.assertEqual(
            encoded_tensor[0, 5, 5], 1,
            "Current player's piece at position (1, 1) should be encoded at (5, 5) in channel 0."
        )
        self.assertEqual(
            encoded_tensor[0, 6, 6], 1,
            "Current player's piece at position (2, 2) should be encoded at (6, 6) in channel 0."
        )

        self.assertEqual(
            encoded_tensor[1, 7, 7], 1,
            "Current player's target position at (3, 3) should be encoded at (7, 7) in channel 1."
        )
        self.assertEqual(
            encoded_tensor[1, 7, 8], 1,
            "Current player's target position at (3, 4) should be encoded at (7, 8) in channel 1."
        )

        self.assertEqual(
            encoded_tensor[2, 8, 8], 1,
            "Other player's piece at position (4, 4) should be encoded at (8, 8) in channel 2."
        )
        self.assertEqual(
            encoded_tensor[2, 3, 3], 1,
            "Other player's piece at position (-1, -1) should be encoded at (3, 3) in channel 2."
        )

    def test_encode_three_players(self):
        game = MagicMock(spec=ChineseCheckersGame)

        current_player_positions = [Position(0, 0), Position(1, 1)]
        current_player_target_positions = [Position(3, 3)]
        other_player_1_positions = [Position(2, -1)]
        other_player_2_positions = [Position(-2, 2)]

        current_player = MagicMock(spec=Player)
        current_player.positions = current_player_positions
        current_player.target_positions = current_player_target_positions
        game.get_current_player.return_value = current_player

        other_player_1 = MagicMock(spec=Player)
        other_player_1.positions = other_player_1_positions
        other_player_2 = MagicMock(spec=Player)
        other_player_2.positions = other_player_2_positions
        game.get_other_players.return_value = [other_player_1, other_player_2]

        encoded_tensor = self.encoder.encode(game)

        self.assertEqual(
            encoded_tensor.shape,
            (3, self.encoder.board_dim, self.encoder.board_dim),
            "The encoded tensor shape should be (3, board_dim, board_dim) with three channels for player data."
        )

        self.assertEqual(
            encoded_tensor[0, 4, 4], 1,
            "Current player's piece at position (0, 0) should be encoded at (4, 4) in channel 0."
        )
        self.assertEqual(
            encoded_tensor[0, 5, 5], 1,
            "Current player's piece at position (1, 1) should be encoded at (5, 5) in channel 0."
        )

        self.assertEqual(
            encoded_tensor[1, 7, 7], 1,
            "Current player's target position at (3, 3) should be encoded at (7, 7) in channel 1."
        )

        self.assertEqual(
            encoded_tensor[2, 6, 3], 1,
            "Other player's piece at position (2, -1) should be encoded at (6, 3) in channel 2."
        )
        self.assertEqual(
            encoded_tensor[2, 2, 6], 1,
            "Other player's piece at position (-2, 2) should be encoded at (2, 6) in channel 2."
        )
