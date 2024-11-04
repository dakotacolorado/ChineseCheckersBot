from abc import abstractmethod, ABC
from typing import List

from .Experience import Experience
from chinese_checkers.simulation import GameSimulation


class BaseExperienceGenerator(ABC):

    @abstractmethod
    def generate(self, simulation: GameSimulation) -> List[Experience]:
        pass
