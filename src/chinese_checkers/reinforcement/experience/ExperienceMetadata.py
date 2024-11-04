from pathlib import Path
from dataclasses import dataclass
from chinese_checkers.simulation import SimulationMetadata

@dataclass(frozen=True)
class ExperienceMetadata(SimulationMetadata):
    generator_name: str

    def to_path(self) -> Path:
        """Generates a file path one level deeper than SimulationMetadata, adding generator_name as a subdirectory."""
        base_path = super().to_path()  # Get path from SimulationMetadata
        return base_path / f'generator_name={self.generator_name}'

    @classmethod
    def from_simulation_metadata(cls, simulation_metadata: SimulationMetadata, generator_name: str) -> "ExperienceMetadata":
        """
        Creates an ExperienceMetadata instance from a SimulationMetadata instance and a generator_name.

        Args:
            simulation_metadata (SimulationMetadata): The existing simulation metadata.
            generator_name (str): The name of the generator to include in the experience metadata.

        Returns:
            ExperienceMetadata: A new instance of ExperienceMetadata with the provided generator name.
        """
        return cls(
            player_count=simulation_metadata.player_count,
            board_size=simulation_metadata.board_size,
            max_game_length=simulation_metadata.max_game_length,
            winning_player=simulation_metadata.winning_player,
            name=simulation_metadata.name,
            version=simulation_metadata.version,
            generator_name=generator_name
        )
