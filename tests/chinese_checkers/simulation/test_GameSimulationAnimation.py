from unittest import TestCase, mock
from matplotlib.animation import FuncAnimation
from IPython.display import Image
from src.chinese_checkers.game.ChineseCheckersGame import ChineseCheckersGame
from src.chinese_checkers.simulation.GameSimulationAnimation import GameSimulationAnimation
from src.chinese_checkers.simulation.GameSimulation import GameSimulation
import os
import tempfile


class TestGameSimulationAnimation(TestCase):

    def setUp(self):
        # Mock basic data for GameSimulation
        self.mock_game_simulation_data = mock.MagicMock(spec=GameSimulation)
        self.mock_game_sequence = [mock.MagicMock(spec=ChineseCheckersGame), mock.MagicMock(spec=ChineseCheckersGame)]

        # Mock the method _to_game_sequence to return mock_game_sequence
        self.mock_game_simulation_data._to_game_sequence.return_value = self.mock_game_sequence

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

    @mock.patch("tempfile.gettempdir", return_value=tempfile.gettempdir())
    def test_display_creates_and_displays_gif(self, mock_tempdir):
        # Test that display method creates a temporary GIF and returns an Image display object
        animation_instance = GameSimulationAnimation(self.mock_game_sequence)

        # Save the animation directly to ensure file creation
        animation_instance.anim.save(animation_instance.temp_gif_path, writer='pillow')

        # Now attempt to display it
        result = animation_instance.display()
        self.assertIsInstance(result, Image, "Expected display method to return an IPython.display.Image object.")
        self.assertTrue(os.path.exists(animation_instance.temp_gif_path), "Temporary GIF file should exist.")

        # Clean up temp file if it was created during the test
        if os.path.exists(animation_instance.temp_gif_path):
            os.remove(animation_instance.temp_gif_path)

    def test_update_plot_functionality(self):
        # Ensure that _update_plot method can be called without exceptions
        animation_instance = GameSimulationAnimation(self.mock_game_sequence)

        try:
            animation_instance._update_plot(0)
        except Exception as e:
            self.fail(f"_update_plot raised an exception unexpectedly: {e}")

    @mock.patch("logging.info")
    def test_save_to_file(self, mock_logging_info):
        # Test that save_to_file calls the right methods and logs the message
        animation_instance = GameSimulationAnimation(self.mock_game_sequence)

        with mock.patch.object(animation_instance.anim, "save") as mock_save:
            animation_instance.save_to_file("test_animation.mp4")
            mock_save.assert_called_once_with("test_animation.mp4", writer=mock.ANY)
            mock_logging_info.assert_called_once_with("Saved animation to test_animation.mp4")

    def test_no_frames_in_game_sequence(self):
        # Test initialization with an empty game sequence
        with self.assertRaises(ValueError, msg="Expected ValueError when game sequence is empty."):
            GameSimulationAnimation([])
