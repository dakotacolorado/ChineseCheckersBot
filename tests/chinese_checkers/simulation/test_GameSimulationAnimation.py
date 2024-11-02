from unittest import TestCase, mock
from matplotlib.animation import FuncAnimation
from IPython.display import Image
from src.chinese_checkers.game.ChineseCheckersGame import ChineseCheckersGame
from src.chinese_checkers.simulation.GameSimulationAnimation import GameSimulationAnimation
from src.chinese_checkers.simulation.GameSimulation import GameSimulation
import os


class TestGameSimulationAnimation(TestCase):

    def setUp(self):
        # Mock some basic data for GameSimulation
        self.mock_game_simulation_data = mock.MagicMock(spec=GameSimulation)
        self.mock_game_sequence = [mock.MagicMock(spec=ChineseCheckersGame), mock.MagicMock(spec=ChineseCheckersGame)]

        # Mock the method to_game_sequence to return mock_game_sequence
        self.mock_game_simulation_data.to_game_sequence.return_value = self.mock_game_sequence

        # Set up 'metadata' and its sub-attributes for the mock
        self.mock_game_simulation_data.metadata = mock.MagicMock()
        self.mock_game_simulation_data.metadata.board_size = 4

    def test_init_creates_animation(self):
        # Test that an animation object is created during initialization
        animation_instance = GameSimulationAnimation(self.mock_game_sequence)
        self.assertIsInstance(animation_instance.anim, FuncAnimation,
                              "Expected 'anim' to be an instance of FuncAnimation.")
        self.assertEqual(len(animation_instance.game_sequence), 2,
                         "Expected game sequence length to match input sequence length.")

    def test_display_creates_and_displays_gif(self):
        # Test that display method creates a temporary GIF and returns an Image display object
        animation_instance = GameSimulationAnimation(self.mock_game_sequence)

        # Mock the save method to prevent actual file creation
        with mock.patch.object(animation_instance.anim, "save") as mock_save:
            result = animation_instance.display()
            mock_save.assert_called_once_with(animation_instance.temp_gif_path, writer=mock.ANY)
            self.assertIsInstance(result, Image, "Expected display method to return an IPython.display.Image object.")
            self.assertTrue(os.path.exists(animation_instance.temp_gif_path), "Temporary GIF file should exist.")
            # Clean up temp file if it was created during the test
            if os.path.exists(animation_instance.temp_gif_path):
                os.remove(animation_instance.temp_gif_path)

    def test_update_plot_functionality(self):
        animation_instance = GameSimulationAnimation(self.mock_game_sequence)

        with mock.patch.object(self.mock_game_sequence[0], 'print') as mock_print:
            animation_instance._update_plot(0)
            mock_print.assert_called_with(axes=animation_instance.ax, show_plot=False)
            self.assertGreaterEqual(mock_print.call_count, 1, "Expected 'print' to be called at least once.")

    def test_save_to_file(self):
        # Test that save_to_file calls the right methods and logs the message
        animation_instance = GameSimulationAnimation(self.mock_game_sequence)

        with mock.patch.object(animation_instance.anim, "save") as mock_save, \
                mock.patch("logging.info") as mock_logging_info:
            animation_instance.save_to_file("test_animation.mp4")
            mock_save.assert_called_once_with("test_animation.mp4", writer=mock.ANY)
            mock_logging_info.assert_called_once_with("Saved animation to test_animation.mp4")

    def test_from_simulation_data_creates_instance(self):
        # Test that from_simulation_data creates a GameSimulationAnimation instance
        instance = GameSimulationAnimation.from_simulation_data(self.mock_game_simulation_data)
        self.assertIsInstance(instance, GameSimulationAnimation,
                              "Expected GameSimulationAnimation instance from simulation data.")

    def test_no_frames_in_game_sequence(self):
        # Test initialization with an empty game sequence
        with self.assertRaises(ValueError, msg="Expected ValueError when game sequence is empty."):
            GameSimulationAnimation([])
