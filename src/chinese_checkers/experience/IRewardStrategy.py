from abc import ABC, abstractmethod
from typing import List
from ..game.ChineseCheckersGame import ChineseCheckersGame

class IRewardStrategy(ABC):
    """
    Interface for reward strategies in reinforcement learning with Chinese Checkers.

    A reward strategy defines how rewards are calculated at each turn in a game simulation.
    Different strategies can be implemented to assign rewards based on factors like
    game progress, distance to winning, or specific move quality. This interface allows
    for flexible reward design, enabling experimentation with various training incentives.

    Methods:
        calculate_reward (abstract): Calculate the reward for a specific turn based on the
                                     current game state and overall game context.
    """

    @abstractmethod
    def calculate_reward(
            self,
            game_sequence: List[ChineseCheckersGame],
            winning_player: int,
            turn_index: int
    ) -> float:
        """
        Calculate the reward for a specific turn in the game simulation.

        This method computes the reward associated with a particular turn index, given the
        entire game sequence up to that point, the identity of the winning player, and
        the turn number. Different implementations of this method may assign rewards based
        on the player's progress towards winning, moves that improve or worsen their position,
        or other contextual game factors.

        Args:
            game_sequence (List[ChineseCheckersGame]): The ordered sequence of game states
                                                       throughout the simulation. Each state
                                                       represents a specific turn in the game.
            winning_player (int): The player ID of the winning player, if determined. This
                                  can be used to provide higher rewards for the winning player.
            turn_index (int): The index of the turn for which the reward is calculated.
                              This index refers to the position within the game sequence.

        Returns:
            float: The calculated reward value for the given turn, which may be positive
                   for favorable moves, negative for unfavorable moves, or zero for neutral actions.
        """
        pass
