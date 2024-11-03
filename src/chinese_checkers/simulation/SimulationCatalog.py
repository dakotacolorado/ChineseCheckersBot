import logging
from typing import List, Iterator

from .GameSimulation import GameSimulation
from .SimulationData import SimulationData
from .SimulationMetadata import SimulationMetadata
from ..catalog.LocalH5Catalog import LocalH5Catalog

logger = logging.getLogger(__name__)


class SimulationCatalog(LocalH5Catalog[SimulationMetadata, SimulationData]):
    @property
    def filename(self) -> str:
        return 'GameSimulation.h5'

    def save_simulation(self, simulation: GameSimulation) -> None:
        """Saves a game simulation, using the simulation's metadata as the key."""
        self.save_data(simulation.metadata, simulation.data)
        logger.info(f"Saved simulation with metadata {simulation.metadata}")

    def load_simulations_by_metadata(self, metadata: SimulationMetadata) -> Iterator[GameSimulation]:
        """Loads all game simulations that match the given metadata."""
        for data in self.load_data(metadata, SimulationData):
            yield GameSimulation(metadata, data)

    def list_available_metadata(self) -> List[SimulationMetadata]:
        """Lists all metadata instances available in the catalog."""
        return super().list_available_metadata(SimulationMetadata)
