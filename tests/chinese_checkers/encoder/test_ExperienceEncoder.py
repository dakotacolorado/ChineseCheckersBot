import unittest
import numpy as np
from src.chinese_checkers.encoder.ChineseCheckersGameEncoderFactory import ChineseCheckersGameEncoderFactory
from src.chinese_checkers.encoder.MoveEncoder import MoveEncoder
from src.chinese_checkers.encoder.ExperienceEncoder import ExperienceEncoder
from src.chinese_checkers.experience import Experience
from src.chinese_checkers.game.ChineseCheckersGame import ChineseCheckersGame
from src.chinese_checkers.game.Move import Move


class TestExperienceEncoder(unittest.TestCase):
    def setUp(self):
        """
        Set up a mock game, experience, and encoder for testing.
        """
        # Set up mock game state and moves
        self.mock_game_state = ChineseCheckersGame.start_game()  # Assuming an initialized game state
        self.mock_next_game_state = ChineseCheckersGame.start_game()  # Assuming next state after some action
        self.mock_move = Move(1, 0, None)  # Placeholder Move

        # Set up a sample experience
        self.experience = Experience(
            state=self.mock_game_state,
            action=self.mock_move,
            reward=1.0,
            next_state=self.mock_next_game_state,
            done=False
        )

    def test_initialization_with_valid_encoder(self):
        """
        Test initializing ExperienceEncoder with a valid game encoder.
        """
        encoder = ExperienceEncoder("grid_position_target")
        self.assertIsInstance(encoder.game_encoder, ChineseCheckersGameEncoderFactory._encoders["grid_position_target"])
        self.assertIsInstance(encoder.move_encoder, MoveEncoder)

    def test_encode_experience(self):
        """
        Test encoding an experience and verify the structure of the output.
        """
        encoder = ExperienceEncoder("grid_position_target")

        # Mock the output of the game encoder and move encoder for consistent testing
        encoder.game_encoder.encode = lambda game: np.zeros(10)  # Mock encoded state array
        encoder.move_encoder.encode = lambda move: np.array([0, 1])  # Mock encoded action array

        # Encode the experience
        encoded_experience = encoder.encode(self.experience)

        # Check the structure and types of encoded experience
        self.assertIsInstance(encoded_experience, tuple)
        self.assertEqual(len(encoded_experience), 5)

        # Validate the encoded current state, action, reward, next state, and done flag
        self.assertIsInstance(encoded_experience[0], np.ndarray)  # Encoded current state
        self.assertEqual(encoded_experience[0].shape, (10,))  # Mock shape of encoded state

        self.assertIsInstance(encoded_experience[1], np.ndarray)  # Encoded action
        self.assertEqual(encoded_experience[1].shape, (2,))  # Mock shape of encoded action

        self.assertIsInstance(encoded_experience[2], float)  # Reward
        self.assertEqual(encoded_experience[2], 1.0)  # Check reward value

        self.assertIsInstance(encoded_experience[3], np.ndarray)  # Encoded next state
        self.assertEqual(encoded_experience[3].shape, (10,))  # Mock shape of encoded next state

        self.assertIsInstance(encoded_experience[4], bool)  # Done flag
        self.assertFalse(encoded_experience[4])  # Check done flag value
