import unittest
from pathlib import Path
import shutil
import h5py

from src.chinese_checkers.game.Position import Position
from src.chinese_checkers.game.Player import Player
from src.chinese_checkers.simulation.GameSimulationData import GameSimulationData, DirectoryAttributes, GameData
from src.chinese_checkers.simulation.DataCatalog import DataCatalog


class TestDataCatalog(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_dir = Path("test_catalog")
        cls.test_dir.mkdir()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_dir)

    def test_init(self):
        catalog = DataCatalog(str(self.test_dir))
        self.assertEqual(catalog.catalog_path, self.test_dir)

    def test_save_dataset(self):
        catalog = DataCatalog(str(self.test_dir))
        directory_attributes = DirectoryAttributes(
            player_count=4,
            board_size=10,
            max_game_length=50,
            name="TestGame",
            version="1.0",
            winning_player="Player1"
        )
        game_data = GameData([[
            Player(
                [Position(1, 1)],
                [Position(0, 0)],
                "Player1"
            )]])
        game_simulation_data = GameSimulationData(
            directory=directory_attributes, data=game_data)

        catalog.save_dataset(game_simulation_data)
        # dataset_path = catalog._construct_path(directory_attributes) / DataCatalog.FILENAME
        # self.assertTrue(dataset_path.is_file())
        #
        # with h5py.File(dataset_path, 'r') as h5file:
        #     saved_data = h5file['GameData'][:]
        # self.assertEqual(saved_data.tolist(), game_data.player_position_history)

    def test_construct_path(self):
        catalog = DataCatalog(str(self.test_dir))
        directory_attributes = DirectoryAttributes(
            player_count=4,
            board_size=10,
            max_game_length=50,
            name="TestGame",
            version="1.0",
            winning_player="Player1"
        )
        expected_path = (
                self.test_dir
                / f'player_count={directory_attributes.player_count}'
                / f'board_size={directory_attributes.board_size}'
                / f'max_game_length={directory_attributes.max_game_length}'
                / f'winning_player={directory_attributes.winning_player}'
                / f'name={directory_attributes.name}'
                / f'version={directory_attributes.version}'
        )
        self.assertEqual(catalog._construct_path(directory_attributes), expected_path)



    def test_list_datasets(self):
        catalog = DataCatalog(str(self.test_dir))
        self.assertEqual(catalog.list_datasets(), [])

        # Additional code to create a dataset, then confirm it's listed
        directory_attributes = DirectoryAttributes(
            player_count=4,
            board_size=10,
            max_game_length=50,
            name="TestGame",
            version="1.0",
            winning_player="Player1"
        )
        game_data = GameData([[Player([], [], "Player1")]])
        game_simulation_data = GameSimulationData(directory=directory_attributes, data=game_data)
        catalog.save_dataset(game_simulation_data)

        self.assertEqual(len(catalog.list_datasets()), 1)

    def test_get_dataset(self):
        catalog = DataCatalog(str(self.test_dir))
        directory_attributes = DirectoryAttributes(
            player_count=4,
            board_size=10,
            max_game_length=50,
            name="TestGame",
            version="1.0",
            winning_player="Player1"
        )
        game_data = GameData([[Player([Position(1, 1)], [Position(0, 0)], "Player1")] * 4])
        game_simulation_data = GameSimulationData(directory=directory_attributes, data=game_data)
        catalog.save_dataset(game_simulation_data)

        retrieved_dataset = catalog.get_dataset(directory_attributes)
        self.assertIsInstance(retrieved_dataset, GameSimulationData)
        self.assertEqual(retrieved_dataset.directory, directory_attributes)
