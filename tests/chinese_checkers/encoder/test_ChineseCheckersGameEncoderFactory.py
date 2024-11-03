import unittest
from src.chinese_checkers.encoder.ChineseCheckersGameEncoderFactory import ChineseCheckersGameEncoderFactory
from src.chinese_checkers.encoder.GridPositionTargetEncoder import GridPositionTargetEncoder
from src.chinese_checkers.encoder.IChineseCheckersGameEncoder import IChineseCheckersGameEncoder

class TestChineseCheckersGameEncoderFactory(unittest.TestCase):
    def test_create_valid_encoder(self):
        """
        Test that a valid encoder name returns an instance of the correct encoder class.
        """
        encoder = ChineseCheckersGameEncoderFactory.create("grid_position_target")
        self.assertIsInstance(encoder, GridPositionTargetEncoder)
        self.assertIsInstance(encoder, IChineseCheckersGameEncoder)

    def test_create_invalid_encoder(self):
        """
        Test that an invalid encoder name raises a ValueError.
        """
        with self.assertRaises(ValueError) as context:
            ChineseCheckersGameEncoderFactory.create("invalid_encoder")
        self.assertEqual(str(context.exception), "Encoder strategy 'invalid_encoder' is not available.")
