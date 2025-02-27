from typing import List, Dict
from abc import ABC, abstractmethod

from ..game import Player
from ..game.ChineseCheckersGame import ChineseCheckersGame
from ..game.Move import Move


class IModel(ABC):
    """
    Interface for a model that can be used to play Chinese Checkers.
    """

    @abstractmethod
    def _choose_next_move(self, game: ChineseCheckersGame, move_history: Dict[str, List[Move]] = None) -> Move:
        """
        Chooses the next move to make.
        Args:
            game (ChineseCheckersGame): The current game state.

        Returns:
            Next move to make.
        """
        pass

    def make_move(self, game: ChineseCheckersGame, move_history: Dict[str, List[Move]]) -> (ChineseCheckersGame, Move):
        move: Move = self._choose_next_move(game, move_history)
        return game.apply_move(move), move
