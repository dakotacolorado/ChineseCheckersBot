from pathlib import Path
from typing import List, Optional
import logging
import h5py
from .GameSimulationData import GameSimulationData, DirectoryAttributes, GameData

logger = logging.getLogger(__name__)


class DataCatalog:
    """A class for managing datasets stored in a catalog."""
    FILENAME = 'GameSimulationData.h5'

    def __init__(self, catalog_directory: str):
        """Initialize the catalog with the given directory."""
        self.catalog_path = Path(catalog_directory)
        self.catalog_path.mkdir(parents=True, exist_ok=True)

    def save_dataset(self, dataset: GameSimulationData) -> None:
        """Save a dataset to the catalog."""
        dataset_path = self._construct_path(dataset.directory)
        dataset_path.mkdir(parents=True, exist_ok=True)

        with h5py.File(dataset_path / self.FILENAME, 'w') as h5file:
            h5file.create_dataset('GameData', data=dataset.data)

    def _construct_path(self, directory: DirectoryAttributes) -> Path:
        """Construct the file path for a dataset based on its attributes."""
        return self.catalog_path.joinpath(
            f'player_count={directory.player_count}',
            f'board_size={directory.board_size}',
            f'max_game_length={directory.max_game_length}',
            f'winning_player={directory.winning_player}',
            f'name={directory.name}',
            f'version={directory.version}',
        )

    def list_datasets(self) -> List[DirectoryAttributes]:
        """Return a list of all datasets in the catalog."""
        dataset_paths = list(self.catalog_path.rglob(self.FILENAME))
        return [self.extract_metadata_from_path(path) for path in dataset_paths]

    @staticmethod
    def extract_metadata_from_path(dataset_path: Path) -> DirectoryAttributes:
        """Extract metadata from a given dataset path."""
        try:
            return DirectoryAttributes(
                part.split('=')[1]
                for part in dataset_path.parts
                if '=' in part
            )
        except IndexError as e:
            logger.error(f"Failed to extract metadata from path {dataset_path}: {e}")
            raise

    def get_dataset(self, directory: DirectoryAttributes) -> Optional[GameSimulationData]:
        """Retrieve a dataset from the catalog."""
        dataset_path = self._construct_path(directory) / self.FILENAME
        if not dataset_path.is_file():
            logger.error(f"Dataset file {dataset_path} does not exist.")
            return None

        with h5py.File(dataset_path, 'r') as h5file:
            data = GameData(h5file['GameData'][:])
        return GameSimulationData(directory, data)

