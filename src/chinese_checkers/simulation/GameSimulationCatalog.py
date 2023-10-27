from pathlib import Path
import logging
from typing import List, Union, Tuple, Optional, Iterator

import h5py
from .GameSimulationData import GameSimulationData, GameMetadata, GamePositions

logger = logging.getLogger(__name__)


class GameSimulationCatalog:
    FILENAME = 'GameSimulationData.h5'

    def __init__(self, catalog_directory: str):
        self.catalog_path = Path(catalog_directory)
        self.catalog_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized GameSimulationCatalog at {self.catalog_path}")

    def _construct_path(self, metadata: GameMetadata) -> Path:
        return self.catalog_path.joinpath(
            f'player_count={metadata.player_count}',
            f'board_size={metadata.board_size}',
            f'max_game_length={metadata.max_game_length}',
            f'winning_player={metadata.winning_player}',
            f'name={metadata.name}',
            f'version={metadata.version}',
            self.FILENAME
        )

    def save_simulation(self, simulation: GameSimulationData) -> None:
        file_path = self._construct_path(simulation.metadata)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        storable_data = simulation.positions.to_storable()

        with h5py.File(file_path, 'a') as h5file:
            group = h5file.create_group(f'simulation_{len(h5file.keys())}')
            for key, value in storable_data.items():
                group.create_dataset(key, data=value)

        logger.info(f"Saved simulation at {file_path}")

    def load_simulations_by_metadata(
            self,
            metadata: GameMetadata,
            index: Optional[Union[int, Tuple[int, int]]] = None
    ) -> Iterator[GameSimulationData]:
        """
        Load the list of game simulations matching the provided metadata.
        If an index or range of indexes is provided, only the simulations at
        those positions will be loaded.

        :param metadata: Metadata to filter simulations by.
        :param index: Optional index or range (start, end) to load specific simulations.
        :return: An iterator over the matched GameSimulationData objects.
        """
        # Construct the path to the file
        dataset_path = self._construct_path(metadata)

        # Check if the file exists
        if not dataset_path.is_file():
            logger.error(f"Dataset file {dataset_path} does not exist.")
            return

        # Load simulations from the h5py file
        with h5py.File(dataset_path, 'r') as h5file:
            # If an index is provided, filter the groups (simulations) accordingly
            if index is not None:
                if isinstance(index, int):
                    keys = [f"simulation_{index}"]
                elif isinstance(index, tuple) and len(index) == 2:
                    start, end = index
                    keys = [f"simulation_{i}" for i in range(start, end)]
                else:
                    raise ValueError(f"Invalid index format: {index}")
            else:
                # If no index is provided, load all simulations
                keys = list(h5file.keys())

            # For each key (simulation), retrieve the data and yield a GameSimulationData object
            for key in keys:
                if key in h5file:
                    group = h5file[key]
                    storable_data = {k: group[k][:] for k in group.keys()}
                    positions = GamePositions.from_storable(storable_data)
                    yield GameSimulationData(metadata, positions)
                else:
                    logger.warning(f"Simulation {key} not found in {dataset_path}")

    def list_available_metadata(self) -> List[GameMetadata]:
        """
        Returns a list of all unique metadata options available in the catalog.
        This does not return the actual simulations but rather the metadata
        configurations that are stored.
        """
        dataset_paths = list(self.catalog_path.rglob(self.FILENAME))
        return [self.extract_metadata_from_path(path.parent) for path in dataset_paths]

    @staticmethod
    def extract_metadata_from_path(directory_path: Path) -> GameMetadata:
        try:
            parts = {part.split('=')[0]: part.split('=')[1] for part in directory_path.parts if '=' in part}
            return GameMetadata(
                player_count=int(parts['player_count']),
                board_size=int(parts['board_size']),
                max_game_length=int(parts['max_game_length']),
                winning_player=parts['winning_player'],
                name=parts['name'],
                version=parts['version']
            )
        except IndexError as e:
            logger.error(f"Failed to extract metadata from path {directory_path}: {e}")
            raise
