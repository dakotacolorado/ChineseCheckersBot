from dataclasses import dataclass
from typing import List
import torch

from .ExperienceData import ExperienceData
from .ExperienceMetadata import ExperienceMetadata
from ..encode import SpatialMoveMetricsEncoder, RewardEncoder, SpatialBoardMetricsEncoder
from ...catalog import IDataMetadata
from ...game import ChineseCheckersGame
from ...simulation import GameSimulation, SimulationMetadata
from ..encode.RewardEncoderV2 import RewardEncoderV2

@dataclass(frozen=True)
class Experience(IDataMetadata[ExperienceData, ExperienceMetadata]):

    data: ExperienceData
    metadata: ExperienceMetadata

    @staticmethod
    def from_data_metadata(data: ExperienceData, metadata: ExperienceMetadata) -> 'Experience':
        return Experience(data, metadata)
