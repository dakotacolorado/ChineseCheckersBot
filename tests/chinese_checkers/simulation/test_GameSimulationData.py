import unittest
import numpy as np

from src.chinese_checkers.simulation.GameSimulationData import GamePositions, GameMetadata, GameSimulationData


class TestGameMetadata(unittest.TestCase):

    def test_init(self):
        metadata = GameMetadata(player_count=2, board_size=4, max_game_length=10,
                                winning_player="Player1", name="TestGame", version="1.0")
        self.assertEqual(metadata.player_count, 2)
        self.assertEqual(metadata.board_size, 4)
        self.assertEqual(metadata.max_game_length, 10)
        self.assertEqual(metadata.winning_player, "Player1")
        self.assertEqual(metadata.name, "TestGame")
        self.assertEqual(metadata.version, "1.0")


class TestGamePositions(unittest.TestCase):

    def setUp(self):
        self.game_positions = GamePositions(
            player_ids=["P1", "P2"],
            player_target_positions=[(1, 2), (3, 4)],
            player_historical_positions=[[(1, 2), (2, 3)], [(3, 4), (4, 5)]]
        )

    def test_to_storable(self):
        storable = GamePositions.to_storable(self.game_positions)
        self.assertTrue(isinstance(storable["player_ids"], np.ndarray))
        self.assertTrue(isinstance(storable["player_target_positions"], np.ndarray))
        self.assertTrue(isinstance(storable["player_historical_positions"], np.ndarray))

    def test_from_storable(self):
        storable = GamePositions.to_storable(self.game_positions)
        restored = GamePositions.from_storable(storable)
        self.assertEqual(restored, self.game_positions)


class TestGameSimulationData(unittest.TestCase):

    def setUp(self):
        self.metadata = GameMetadata(player_count=2, board_size=4, max_game_length=10,
                                     winning_player="Player1", name="TestGame", version="1.0")
        self.game_positions = GamePositions(
            player_ids=["P1", "P2"],
            player_target_positions=[(1, 2), (3, 4)],
            player_historical_positions=[[(1, 2), (2, 3)], [(3, 4), (4, 5)]]
        )
        self.simulation_data = GameSimulationData(metadata=self.metadata, positions=self.game_positions)

    def test_init(self):
        self.assertEqual(self.simulation_data.metadata, self.metadata)
        self.assertEqual(self.simulation_data.positions, self.game_positions)
