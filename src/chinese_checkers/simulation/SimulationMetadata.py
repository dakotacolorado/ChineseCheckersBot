from dataclasses import dataclass
from pathlib import Path

from ..catalog import IMetadata


@dataclass(frozen=True)
class SimulationMetadata(IMetadata):
    """Metadata about the game simulation."""
    player_count: int
    board_size: int
    max_game_length: int
    winning_player: str
    name: str
    version: str

    def to_path(self) -> Path:
        """Generates a unique file path for storage organization based on metadata properties."""
        return Path(
            f'player_count={self.player_count}',
            f'board_size={self.board_size}',
            f'max_game_length={self.max_game_length}',
            f'name={self.name}',
            f'version={self.version}',
            f'winning_player={self.winning_player}',
        )

    def __hash__(self) -> int:
        """Generates a hash based on all metadata fields, allowing it to serve as a unique dictionary key."""
        return hash((self.player_count, self.board_size, self.max_game_length,
                     self.winning_player, self.name, self.version))

    def __eq__(self, other: object) -> bool:
        """Checks equality between two SimulationMetadata instances based on all fields."""
        if not isinstance(other, SimulationMetadata):
            return False
        return (
            self.player_count == other.player_count and
            self.board_size == other.board_size and
            self.max_game_length == other.max_game_length and
            self.winning_player == other.winning_player and
            self.name == other.name and
            self.version == other.version
        )
