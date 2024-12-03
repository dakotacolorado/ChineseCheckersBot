import unittest

from src.chinese_checkers.game import Move
from src.chinese_checkers.game import ChineseCheckersGame
from src.chinese_checkers.model import BootstrapModel
from src.chinese_checkers.geometry.Vector import Vector


class TestBootstrapModel(unittest.TestCase):

    def setUp(self):
        # Initialize the game and BootstrapModel for testing
        self.game = ChineseCheckersGame.start_game()
        self.game.start_game()
        self.model = BootstrapModel()

    def test_chose_next_move(self):
        # Verify that `_chose_next_move` returns a valid move
        move = self.model._choose_next_move(self.game)

        # Assert that a move is returned and that it's a Move instance
        self.assertIsNotNone(move, "The model did not return a move.")
        self.assertIsInstance(move, Move, "The returned move is not an instance of Move.")

        # Verify that the move's position is a Vector
        self.assertIsInstance(move.position, Vector, "The move's position is not a Vector instance.")

    def test_chose_next_move_no_viable_move(self):
        # Temporarily modify game state to simulate no viable moves
        self.game.get_next_moves = lambda: []

        # Check that an exception is raised when there are no moves
        with self.assertRaises(Exception) as context:
            self.model._choose_next_move(self.game)
        self.assertEqual(str(context.exception), "No viable move found", "Unexpected exception message.")


if __name__ == '__main__':
    unittest.main()
