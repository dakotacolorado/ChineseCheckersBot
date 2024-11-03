from pathlib import Path
import logging
from typing import List, Union, Tuple, Optional, Iterator
import h5py
import numpy as np
from .GameSimulation import GameSimulation
from .SimulationData import SimulationData
from .SimulationMetadata import SimulationMetadata

logger = logging.getLogger(__name__)


class SimulationCatalog:
    FILENAME = 'GameSimulation.h5'

    def __init__(self, catalog_directory: str):
        self.catalog_path = Path(catalog_directory)
        self.catalog_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized GameSimulationCatalog at {self.catalog_path}")

    def _construct_path(self, metadata: SimulationMetadata) -> Path:
        return self.catalog_path.joinpath(metadata.to_path(), self.FILENAME)

    def save_simulation(self, simulation: GameSimulation) -> None:
        file_path = self._construct_path(simulation.metadata)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        storable_data = simulation.data.to_storable()

        with h5py.File(file_path, 'a') as h5file:
            group = h5file.create_group(f'simulation_{len(h5file.keys())}')
            for key, value in storable_data.items():
                group.create_dataset(key, data=value)

        logger.info(f"Saved simulation at {file_path}")

    def load_simulations_by_metadata(
            self,
            metadata: SimulationMetadata,
            index: Optional[Union[int, Tuple[int, int]]] = None
    ) -> Iterator[GameSimulation]:
        """
        Load the list of game simulations matching the provided metadata.
        """
        dataset_path = self._construct_path(metadata)

        if not dataset_path.is_file():
            logger.error(f"Dataset file {dataset_path} does not exist.")
            return

        with h5py.File(dataset_path, 'r') as h5file:
            if index is not None:
                if isinstance(index, int):
                    keys = [f"simulation_{index}"]
                elif isinstance(index, tuple) and len(index) == 2:
                    start, end = index
                    keys = [f"simulation_{i}" for i in range(start, end)]
                else:
                    raise ValueError(f"Invalid index format: {index}")
            else:
                keys = list(h5file.keys())

            for key in keys:
                if key in h5file:
                    group = h5file[key]
                    storable_data = {k: group[k][:] for k in group.keys()}
                    data = SimulationData.from_storable(storable_data)
                    yield GameSimulation(metadata, data)
                else:
                    logger.warning(f"Simulation {key} not found in {dataset_path}")

    def list_available_metadata(self) -> List[SimulationMetadata]:
        dataset_paths = list(self.catalog_path.rglob(self.FILENAME))
        return [self._extract_metadata_from_path(path.parent) for path in dataset_paths]

    @staticmethod
    def _extract_metadata_from_path(directory_path: Path) -> SimulationMetadata:
        try:
            parts = {part.split('=')[0]: part.split('=')[1] for part in directory_path.parts if '=' in part}
            return SimulationMetadata(
                player_count=int(parts['player_count']),
                board_size=int(parts['board_size']),
                max_game_length=int(parts['max_game_length']),
                name=parts['name'],
                version=parts['version'],
                winning_player=parts['winning_player'],
            )
        except IndexError as e:
            logger.error(f"Failed to extract metadata from path {directory_path}: {e}")
            raise
