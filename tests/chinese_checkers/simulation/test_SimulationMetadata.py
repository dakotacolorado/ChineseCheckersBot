import unittest
from pathlib import Path

from src.chinese_checkers.simulation.SimulationMetadata import SimulationMetadata


class TestSimulationMetadata(unittest.TestCase):

    def test_init(self):
        metadata = SimulationMetadata(player_count=2, board_size=4, max_game_length=10,
                                      winning_player="Player1", name="TestGame", version="1.0")
        self.assertEqual(metadata.player_count, 2)
        self.assertEqual(metadata.board_size, 4)
        self.assertEqual(metadata.max_game_length, 10)
        self.assertEqual(metadata.winning_player, "Player1")
        self.assertEqual(metadata.name, "TestGame")
        self.assertEqual(metadata.version, "1.0")

    def test_to_path(self):
        metadata = SimulationMetadata(player_count=2, board_size=4, max_game_length=10,
                                      winning_player="Player1", name="TestGame", version="1.0")
        expected_path = Path("player_count=2", "board_size=4", "max_game_length=10",
                             "name=TestGame", "version=1.0", "winning_player=Player1")
        self.assertEqual(metadata.to_path(), expected_path)
