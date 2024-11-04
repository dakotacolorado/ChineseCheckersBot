import logging
from typing import List, Iterator, Type

from .Experience import Experience
from .ExperienceData import ExperienceData
from .ExperienceMetadata import ExperienceMetadata
from ...catalog.LocalH5Catalog import LocalH5Catalog


class ExperienceCatalog(LocalH5Catalog[ExperienceMetadata, ExperienceData, Experience]):

    logger = logging.getLogger(__name__)

    @property
    def filename(self) -> str:
        return 'ExperienceData.h5'

    @property
    def metadata_cls(self) -> Type[ExperienceMetadata]:
        return ExperienceMetadata

    @property
    def data_cls(self) -> Type[ExperienceData]:
        return ExperienceData

    @property
    def data_metadata_cls(self) -> Type[Experience]:
        return Experience

    def save_experience(self, experience: Experience) -> None:
        """Saves an experience, using the experience's metadata as the key."""
        self.create_dataset(experience.metadata)
        self.add_record(experience.metadata, experience.data)
        self.logger.info(f"Saved experience with metadata {experience.metadata}")

    def load_experiences_by_metadata(self, metadata: ExperienceMetadata) -> Iterator[Experience]:
        """Loads all experiences that match the given metadata."""
        for data_metadata in self.load_dataset(metadata):
            yield data_metadata  # data_metadata is already an Experience instance

    def list_available_metadata(self) -> List[ExperienceMetadata]:
        """Lists all metadata instances available in the catalog."""
        return self.list_datasets()
