import unittest
import numpy as np
from unittest.mock import MagicMock
from src.chinese_checkers.encoder.MoveEncoder import MoveEncoder
from src.chinese_checkers.game.Move import Move
from src.chinese_checkers.game.Position import Position


class TestMoveEncoder(unittest.TestCase):

    def test_encode_move(self):
        # Mock the apply method to return the expected end position
        move = Move(1, 1, Position(2, 2))
        move.apply = MagicMock(return_value=Position(2, 2))

        # Encode the move and check the output
        encoded_move = MoveEncoder.encode(move)
        expected_encoding = np.array([1, 1, 2, 2])

        np.testing.assert_array_equal(
            encoded_move, expected_encoding,
            f"Expected encoded move to be {expected_encoding}, but got {encoded_move}"
        )

    def test_encode_move_negative_coordinates(self):
        # Mock the apply method to return the expected end position with negative coordinates
        move = Move(-1, -2, Position(-3, -4))
        move.apply = MagicMock(return_value=Position(-3, -4))

        # Encode the move and check the output
        encoded_move = MoveEncoder.encode(move)
        expected_encoding = np.array([-1, -2, -3, -4])

        np.testing.assert_array_equal(
            encoded_move, expected_encoding,
            f"Expected encoded move with negative coordinates to be {expected_encoding}, but got {encoded_move}"
        )

    def test_create_dummy_action(self):
        # Test the dummy action encoding
        dummy_action = MoveEncoder.create_dummy_action()
        expected_dummy = np.array([0, 0, 0, 0])

        np.testing.assert_array_equal(
            dummy_action, expected_dummy,
            f"Expected dummy action to be {expected_dummy}, but got {dummy_action}"
        )


