from dataclasses import dataclass

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
