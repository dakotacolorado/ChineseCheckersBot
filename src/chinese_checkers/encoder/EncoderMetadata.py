
from dataclasses import dataclass
from pathlib import Path

from ..catalog import IMetadata

@dataclass(frozen=True)
class EncoderMetadata(IMetadata):
    board_size: int
    encoder_name: str

    def to_path(self) -> Path:
        return Path(
            f'board_size={self.board_size}',
            f'encoder_name={self.encoder_name}',
        )

    def __hash__(self) -> int:
        return hash((self.board_size, self.encoder_name))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, EncoderMetadata):
            return False
        return (
            self.board_size == other.board_size and
            self.encoder_name == other.encoder_name
        )

