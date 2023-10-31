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
        storable = SimulationData.to_storable(self.game_positions)
        self.assertTrue(isinstance(storable["player_ids"], np.ndarray))
        self.assertTrue(isinstance(storable["player_start_positions"], np.ndarray))
        self.assertTrue(isinstance(storable["player_target_positions"], np.ndarray))
        self.assertTrue(isinstance(storable["historical_moves"], np.ndarray))

    def test_from_storable(self):
        storable = SimulationData.to_storable(self.game_positions)
        restored = SimulationData.from_storable(storable)
        self.assertEqual(restored, self.game_positions)
