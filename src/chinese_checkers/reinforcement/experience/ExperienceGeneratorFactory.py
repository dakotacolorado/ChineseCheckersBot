from typing import Callable, Dict

from .BaseExperienceGenerator import BaseExperienceGenerator
from .DerivedFeaturesExperienceGenerator import DerivedFeaturesExperienceGenerator
from .ExperienceGeneratorMetadata import ExperienceEncoderMetadata
from ...simulation import GameSimulation


class ExperienceGeneratorFactory:

    @property
    def _encoders(self) -> Dict[str, Callable[[GameSimulation], BaseExperienceGenerator]]:
        return {
            "derived_features_generator": lambda game_simulation: DerivedFeaturesExperienceGenerator(),
        }

    def __init__(
            self,
            metadata: ExperienceEncoderMetadata
    ):
        self.metadata = metadata


    def create_encoder(self, game_simulation: GameSimulation) -> BaseExperienceGenerator:
        name = self.metadata.encoder_name
        encoder_class = self._encoders.get(name)
        if not encoder_class:
            raise ValueError(f"Encoder strategy '{name}' is not available.")
        return encoder_class(game_simulation)