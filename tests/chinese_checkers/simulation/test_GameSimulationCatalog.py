import unittest
import tempfile
import shutil

from src.chinese_checkers.game.Move import Move
from src.chinese_checkers.game.Position import Position
from src.chinese_checkers.simulation.GameSimulation import GameSimulation
from src.chinese_checkers.simulation.SimulationData import SimulationData
from src.chinese_checkers.simulation.SimulationMetadata import SimulationMetadata

from src.chinese_checkers.simulation.SimulationCatalog import SimulationCatalog


class TestGameSimulationCatalog(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory for the catalog
        self.temp_dir = tempfile.mkdtemp()
        self.catalog = SimulationCatalog(self.temp_dir)

        # Sample metadata and data
        self.sample_metadata = SimulationMetadata(
            player_count=6,
            board_size=121,
            max_game_length=100,
            winning_player="player1",
            name="sample_game",
            version="1.0"
        )

        self.sample_data = SimulationData(
            player_ids=["player1", "player2"],
            player_start_positions=[[Position(0, 0)], [Position(1, 1)]],
            player_target_positions=[[Position(5, 5)], [Position(6, 6)]],
            historical_moves=[Move(1, 1, Position(0, 0))]
        )

    def tearDown(self):
        # Remove the temporary directory
        shutil.rmtree(self.temp_dir)

    def test_save_and_load_simulation(self):
        # Save a simulation to the catalog
        simulation = GameSimulation(self.sample_metadata, self.sample_data)
        self.catalog.save_simulation(simulation)

        # Load the saved simulation
        loaded_simulations = list(self.catalog.load_simulations_by_metadata(self.sample_metadata))

        # Check if only one simulation is loaded
        self.assertEqual(len(loaded_simulations), 1)

        # Check if the loaded simulation matches the saved one
        self.assertEqual(loaded_simulations[0].metadata, self.sample_metadata)
        self.assertEqual(loaded_simulations[0].data, self.sample_data)

    def test_list_available_metadata(self):
        # Initially, there should be no metadata
        self.assertEqual(self.catalog.list_available_metadata(), [])

        # Save a simulation
        simulation = GameSimulation(self.sample_metadata, self.sample_data)
        self.catalog.save_simulation(simulation)

        # Check if the saved metadata is available
        available_metadata = self.catalog.list_available_metadata()
        self.assertEqual(len(available_metadata), 1)
        self.assertEqual(available_metadata[0], self.sample_metadata)

