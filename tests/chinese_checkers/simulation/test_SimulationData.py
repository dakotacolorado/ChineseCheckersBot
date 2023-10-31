import unittest

import numpy as np

from src.chinese_checkers.game.Position import Position
from src.chinese_checkers.game.Move import Move
from src.chinese_checkers.simulation.SimulationData import SimulationData


class TestSimulationData(unittest.TestCase):

    def setUp(self):
        self.game_positions = SimulationData(
            player_ids=["P1", "P2"],
            player_start_positions=[[Position(0, 1), Position(0, 2)], [Position(2, 3), Position(2, 4)]],
            player_target_positions=[[Position(1, 2), Position(1, 3)], [Position(3, 4), Position(3, 5)]],
            historical_moves=[Move(0, 1, Position(0, 1)), Move(2, 3, Position(2, 3))]
        )

    def test_to_storable(self):
        """Test if the to_storable method returns appropriate numpy arrays."""
        storable = SimulationData.to_storable(self.game_positions)
        self.assertTrue(isinstance(storable["player_ids"], np.ndarray))
        self.assertTrue(isinstance(storable["player_start_positions"], np.ndarray))
        self.assertTrue(isinstance(storable["player_target_positions"], np.ndarray))
        self.assertTrue(isinstance(storable["historical_moves"], np.ndarray))

        self.assertEqual(storable["player_ids"].tolist(), [b"P1", b"P2"])

    def test_from_storable(self):
        """Test if the from_storable method restores the original SimulationData object."""
        storable = SimulationData.to_storable(self.game_positions)
        restored = SimulationData.from_storable(storable)
        self.assertEqual(restored, self.game_positions)

        self.assertEqual(restored.player_ids, self.game_positions.player_ids)

    def test_from_storable_with_missing_key(self):
        """Test the behavior when a required key is missing."""
        storable = SimulationData.to_storable(self.game_positions)
        del storable["player_ids"]
        with self.assertRaises(ValueError):
            SimulationData.from_storable(storable)
