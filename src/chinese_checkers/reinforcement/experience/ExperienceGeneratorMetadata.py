from pathlib import Path

from chinese_checkers.catalog import IMetadata


class ExperienceEncoderMetadata(IMetadata):
    encoder_name: str

    def to_path(self) -> Path:
        return Path(
            f'encoder_name={self.encoder_name}',
        )

    def __hash__(self) -> int:
        return hash((self.encoder_name))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ExperienceEncoderMetadata):
            return False
        return (
            self.encoder_name == other.encoder_name
        )