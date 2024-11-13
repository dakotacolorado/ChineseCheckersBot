from dataclasses import dataclass

import torch

from .ExperienceData import ExperienceData
from .ExperienceMetadata import ExperienceMetadata
from chinese_checkers.catalog import IDataMetadata


@dataclass(frozen=True)
class Experience(IDataMetadata[ExperienceData, ExperienceMetadata]):

    data: ExperienceData
    metadata: ExperienceMetadata

    @staticmethod
    def from_data_metadata(data: ExperienceData, metadata: ExperienceMetadata) -> 'Experience':
        return Experience(data, metadata)

    def action_eq(self, other):
        if not isinstance(other, Experience):
            return False
        return (

                torch.equal(self.data.action, other.data.action)
        )

    def action_hash(self) -> int:
        """Generates a hash based on all metadata fields, allowing it to serve as a unique dictionary key."""
        return hash(self.data.action)