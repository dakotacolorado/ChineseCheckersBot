from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class SimulationMetadata:
    """Metadata about the game simulation."""
    player_count: int
    board_size: int
    max_game_length: int
    winning_player: str
    name: str
    version: str

    def to_path(self: "SimulationMetadata") -> Path:
        return Path(
            f'player_count={self.player_count}',
            f'board_size={self.board_size}',
            f'max_game_length={self.max_game_length}',
            f'name={self.name}',
            f'version={self.version}',
            f'winning_player={self.winning_player}',
        )





