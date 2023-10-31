from src.chinese_checkers.simulation.GameSimulation import GameSimulation


import unittest
from unittest.mock import MagicMock

class TestGameSimulation(unittest.TestCase):

    def setUp(self):
        # Mock models to be used in tests.
        self.model_1 = MagicMock()
        self.model_1.make_move.return_value = (MagicMock(), MagicMock())
        self.model_2 = MagicMock()
        self.model_2.make_move.return_value = (MagicMock(), MagicMock())
        self.models = [self.model_1, self.model_2]

    def test_validate_input_valid(self):
        # This should run without any exceptions
        GameSimulation._validate_input(
            self.models, "TestName", "1.0", 500, 4, 1
        )

    def test_validate_input_invalid_version(self):
        with self.assertRaises(ValueError):
            GameSimulation._validate_input(
                self.models, "TestName", "invalid", 500, 4, 1
            )

    def test_validate_input_invalid_max_turns(self):
        with self.assertRaises(ValueError):
            GameSimulation._validate_input(
                self.models, "TestName", "1.0", 1500, 4, 1
            )

    def test_validate_input_invalid_name(self):
        with self.assertRaises(ValueError):
            GameSimulation._validate_input(
                self.models, "Test Name with Spaces", "1.0", 500, 4, 1
            )

    def test_validate_input_invalid_print_period(self):
        with self.assertRaises(ValueError):
            GameSimulation._validate_input(
                self.models, "TestName", "1.0", 500, 4, -1
            )

    def test_validate_input_invalid_board_size(self):
        with self.assertRaises(ValueError):
            GameSimulation._validate_input(
                self.models, "TestName", "1.0", 500, 0, 1
            )

    def test_run_game_simulation_finishes_within_turns(self):
        mock_game = MagicMock()
        mock_game.is_game_won.return_value = False
        mock_game.turn = 501
        with self.assertRaises(Exception) as context:
            GameSimulation._run_game_simulation(mock_game, self.models, 500, 1)
        self.assertTrue("specified 500 turns" in str(context.exception))
