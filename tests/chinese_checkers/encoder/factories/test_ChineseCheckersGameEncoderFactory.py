import unittest
from unittest.mock import MagicMock
from src.chinese_checkers.encoder.factories import ChineseCheckersGameEncoderFactory, MoveEncoderFactory
from src.chinese_checkers.encoder.implementations import GridPositionTargetEncoder, SpatialBoardStateEncoder, SpatialBoardMetricsEncoder

class TestChineseCheckersGameEncoderFactory(unittest.TestCase):
    def setUp(self):
        self.board_size = 4
        self.move_encoder_factory = MagicMock(spec=MoveEncoderFactory)
        self.move_encoder_factory.create.return_value = MagicMock()  # Mock MoveEncoder

        # Initialize the factory
        self.factory = ChineseCheckersGameEncoderFactory(
            move_encoder_factory=self.move_encoder_factory,
            board_size=self.board_size
        )

    def test_create_grid_position_250_encoder(self):
        encoder = self.factory.create("grid_position_250")
        self.assertIsInstance(encoder, GridPositionTargetEncoder,
                              "Expected 'grid_position_250' to create an instance of GridPositionTargetEncoder")
        self.assertEqual(encoder.max_moves, 250,
                         "Expected max_moves of 250 for 'grid_position_250' encoder")
        self.assertEqual(encoder.board_size, self.board_size,
                         "Expected board size to match factory initialization for 'grid_position_250' encoder")
        self.move_encoder_factory.create.assert_called_once_with("basic_move_encoder")

    def test_create_grid_position_100_encoder(self):
        encoder = self.factory.create("grid_position_100")
        self.assertIsInstance(encoder, GridPositionTargetEncoder,
                              "Expected 'grid_position_100' to create an instance of GridPositionTargetEncoder")
        self.assertEqual(encoder.max_moves, 100,
                         "Expected max_moves of 100 for 'grid_position_100' encoder")
        self.assertEqual(encoder.board_size, self.board_size,
                         "Expected board size to match factory initialization for 'grid_position_100' encoder")
        self.move_encoder_factory.create.assert_called_with("basic_move_encoder")

    def test_create_spatial_board_state_encoder(self):
        encoder = self.factory.create("spatial_board_state")
        self.assertIsInstance(encoder, SpatialBoardStateEncoder,
                              "Expected 'spatial_board_state' to create an instance of SpatialBoardStateEncoder")
        self.assertEqual(encoder.board_size, self.board_size,
                         "Expected board size to match factory initialization for 'spatial_board_state' encoder")

    def test_create_spatial_board_metrics_encoder(self):
        encoder = self.factory.create("spatial_board_metrics")
        self.assertIsInstance(encoder, SpatialBoardMetricsEncoder,
                              "Expected 'spatial_board_metrics' to create an instance of SpatialBoardMetricsEncoder")
        self.assertEqual(encoder.board_size, self.board_size,
                         "Expected board size to match factory initialization for 'spatial_board_metrics' encoder")

    def test_invalid_encoder_name(self):
        with self.assertRaises(ValueError, msg="Expected ValueError for invalid encoder name"):
            self.factory.create("invalid_encoder")
