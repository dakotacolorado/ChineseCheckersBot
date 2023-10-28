from unittest import TestCase, mock

from matplotlib import animation, pyplot as plt

from src.chinese_checkers.game.ChineseCheckersGame import ChineseCheckersGame
from src.chinese_checkers.simulation.GameSimulationAnimation import GameSimulationAnimation
from src.chinese_checkers.simulation.GameSimulationData import GameSimulationData


class TestGameSimulationAnimation(TestCase):

    def setUp(self):
        # Mock some basic data for GameSimulationData
        self.mock_game_simulation_data = mock.MagicMock(spec=GameSimulationData)
        self.mock_game_sequence = [mock.MagicMock(spec=ChineseCheckersGame), mock.MagicMock(spec=ChineseCheckersGame)]

        # Mock the method to_game_sequence to return mock_game_sequence
        self.mock_game_simulation_data.to_game_sequence.return_value = self.mock_game_sequence

        self.mock_game_simulation_data.positions = mock.MagicMock()
        self.mock_game_simulation_data.positions.player_ids = ["player1", "player2"]
        self.mock_game_simulation_data.positions.player_start_positions = ["start1", "start2"]
        self.mock_game_simulation_data.positions.player_target_positions = ["target1", "target2"]
        self.mock_game_simulation_data.positions.historical_moves = [("move1", "move2")]

        # Set up 'metadata' and its sub-attributes for the mock
        self.mock_game_simulation_data.metadata = mock.MagicMock()
        self.mock_game_simulation_data.metadata.board_size = 4

    def test_from_simulation_data(self):
        animation_instance = GameSimulationAnimation.from_simulation_data(self.mock_game_simulation_data)

        # Test if the animation object has been created
        self.assertTrue(hasattr(animation_instance, "anim"))
        self.assertTrue(isinstance(animation_instance.game_sequence, list))
        self.assertEqual(len(animation_instance.game_sequence), 2)

    def test_display(self):
        animation_instance = GameSimulationAnimation(self.mock_game_sequence)

        with mock.patch.object(plt, "show", return_value=None):
            # Check if it runs without issues
            animation_instance.display()

