from dataclasses import dataclass
from pathlib import Path
from chinese_checkers.simulation import SimulationMetadata

@dataclass(frozen=True)
class ExperienceMetadata(SimulationMetadata):
    generator_name: str
    current_player: str

    def to_path(self) -> Path:
        """Generates a file path one level deeper than SimulationMetadata, adding generator_name and current_player as subdirectories."""
        base_path = super().to_path()
        return base_path / f'generator_name={self.generator_name}' / f'current_player={self.current_player}'

    @classmethod
    def from_simulation_metadata(cls, simulation_metadata: SimulationMetadata, generator_name: str, current_player: str) -> "ExperienceMetadata":
        return cls(
            player_count=simulation_metadata.player_count,
            board_size=simulation_metadata.board_size,
            max_game_length=simulation_metadata.max_game_length,
            winning_player=simulation_metadata.winning_player,
            name=simulation_metadata.name,
            version=simulation_metadata.version,
            generator_name=generator_name,
            current_player=current_player
        )
