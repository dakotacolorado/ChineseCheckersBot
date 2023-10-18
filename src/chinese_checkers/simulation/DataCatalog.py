from pathlib import Path
from typing import List

import h5py

from .SimulationDataSet import SimulationDataSet


class DataCatalog:
    """A class for managing datasets stored in a catalog."""

    FILENAME = 'data.h5'

    def __init__(self, catalog_dir: str):
        """Initialize the catalog with the given directory."""
        self.catalog_dir = Path(catalog_dir)

    def list_datasets(self) -> List[dict]:
        """Return a list of all datasets in the catalog."""
        dataset_paths = list(self.catalog_dir.rglob(self.FILENAME))

        def extract_metadata_from_path(dataset_path: Path) -> dict:
            """Extract metadata from a given dataset path."""
            return {key: value for key, value in [part.split('=') for part in dataset_path.parts if '=' in part]}

        return [extract_metadata_from_path(path) for path in dataset_paths]

    def get_dataset(
            self,
            player_count: int,
            board_size: int,
            game_length: int,
            name: str,
            version: str
    ) -> SimulationDataSet:
        """Retrieve a dataset from the catalog."""
        dataset_path = self._construct_path(player_count, board_size, game_length, name, version) / self.FILENAME

        with h5py.File(dataset_path, 'r') as h5file:
            data = h5file['data'][:]
            labels = h5file['labels'][:]

        return SimulationDataSet(player_count, board_size, game_length, name, version, '', data, labels)

    def save_dataset(self, dataset: SimulationDataSet):
        """Save a dataset to the catalog."""
        dataset_path = self._construct_path(
            dataset.player_count,
            dataset.board_size,
            dataset.game_length,
            dataset.name,
            dataset.version
        )
        dataset_path.mkdir(parents=True, exist_ok=True)

        with h5py.File(dataset_path / self.FILENAME, 'w') as h5file:
            h5file.create_dataset('data', data=dataset.data)
            h5file.create_dataset('labels', data=dataset.labels)

    def _construct_path(
            self,
            player_count: int,
            board_size: int,
            game_length: int,
            name: str,
            version: str
    ) -> Path:
        """Construct the file path for a dataset based on its attributes."""
        return (self.catalog_dir / f'player_count={player_count}' / f'board_size={board_size}'
                / f'game_length={game_length}' / f'name={name}' / f'version={version}')
