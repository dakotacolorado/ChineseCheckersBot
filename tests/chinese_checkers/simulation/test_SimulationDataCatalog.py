from pathlib import Path
from unittest import TestCase

import h5py
import numpy as np

from src.chinese_checkers.simulation.DataCatalog import DataCatalog, SimulationDataSet


class TestDataCatalog(TestCase):
    """
    Test class for DataCatalog.

    If the tests are unexpectedly failing, empty the test_data and try again.
    """

    directory = Path("test_data")
    test_catalog = DataCatalog(directory)

    def test_list_datasets(self):
        """Test that datasets can be listed correctly."""
        expected_datasets = []

        def generate_path(player_count, board_size, game_length, name, version) -> Path:
            return (self.directory / f'player_count={player_count}' / f'board_size={board_size}'
                    / f'game_length={game_length}' / f'name={name}' / f'version={version}')

        for player_count in ['1', '2']:
            for board_size in ['1', '2']:
                for game_length in ['1', '2']:
                    for name in ["test", "train"]:
                        for version in ["v1", "v2"]:
                            dataset_path = generate_path(player_count, board_size, game_length, name, version)
                            expected_datasets.append({
                                "player_count": player_count,
                                "board_size": board_size,
                                "game_length": game_length,
                                "name": name,
                                "version": version,
                            })
                            dataset_path.mkdir(parents=True, exist_ok=True)

                            with h5py.File(dataset_path / DataCatalog.FILENAME, 'w') as file:
                                data = np.array([[1, 2, 3], [4, 5, 6]])
                                labels = np.array([0, 1])
                                file.create_dataset('data', data=data)
                                file.create_dataset('labels', data=labels)

        datasets = self.test_catalog.list_datasets()
        self.assertEqual(expected_datasets, datasets)

    def test_save_dataset(self):
        """Test that datasets can be saved correctly."""
        # Create a sample dataset.
        player_count, board_size, game_length, name, version = '1', '1', '1', 'save_sample', 'v1'
        sample_data = np.array([[7, 8, 9], [10, 11, 12]])
        sample_labels = np.array([2, 3])
        sample_dataset = SimulationDataSet(player_count, board_size, game_length, name, version, '', sample_data,
                                           sample_labels)

        # Define dataset path.
        dataset_path = (self.directory / f'player_count={player_count}' / f'board_size={board_size}'
                        / f'game_length={game_length}' / f'name={name}' / f'version={version}' / DataCatalog.FILENAME)
        if dataset_path.exists():
            dataset_path.unlink()

        # Save and verify dataset.
        self.test_catalog.save_dataset(sample_dataset)
        self.assertTrue(dataset_path.exists())
        with h5py.File(dataset_path, 'r') as h5file:
            saved_data = h5file['data'][:]
            saved_labels = h5file['labels'][:]
            np.testing.assert_array_equal(saved_data, sample_data)
            np.testing.assert_array_equal(saved_labels, sample_labels)

        if dataset_path.exists():
            dataset_path.unlink()

    def test_retrieve_dataset(self):
        """Test retrieving a dataset."""
        player_count, board_size, game_length, name, version = '1', '1', '1', 'test', 'v1'
        expected_data = np.array([[1, 2, 3], [4, 5, 6]])
        expected_labels = np.array([0, 1])

        retrieved_dataset = self.test_catalog.get_dataset(player_count, board_size, game_length, name, version)
        np.testing.assert_array_equal(retrieved_dataset.data, expected_data)
        np.testing.assert_array_equal(retrieved_dataset.labels, expected_labels)

    def test_get_dataset_not_found(self):
        """Test retrieving a non-existent dataset."""
        player_count, board_size, game_length, name, version = '1', '1', '1', 'not_found', 'v1'
        with self.assertRaises(FileNotFoundError):
            self.test_catalog.get_dataset(player_count, board_size, game_length, name, version)
