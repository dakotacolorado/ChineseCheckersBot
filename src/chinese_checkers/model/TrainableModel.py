from typing import List
from abc import ABC, abstractmethod
import torch

from . import IModel
from ..game.ChineseCheckersGame import ChineseCheckersGame
from ..game.Move import Move
from chinese_checkers.simulation import GameSimulation


class TrainableModel(IModel, ABC):
    """
    Abstract base class for trainable models used in Chinese Checkers simulations.
    Extends the IModel interface to include methods for encoding and training.
    """

    @abstractmethod
    def encode_game(self, game: ChineseCheckersGame) -> torch.Tensor:
        """
        Encodes the current game state into a tensor representation.

        Args:
            game (ChineseCheckersGame): The current game state.

        Returns:
            torch.Tensor: Encoded representation of the game.
        """
        pass

    @abstractmethod
    def encode_move(self, move: Move) -> torch.Tensor:
        """
        Encodes a move into a tensor representation.

        Args:
            move (Move): The move to encode.

        Returns:
            torch.Tensor: Encoded representation of the move.
        """
        pass

    @abstractmethod
    def encode_reward(self, simulation: GameSimulation) -> List[torch.Tensor]:
        """
        Encodes the rewards for a simulation into tensor representations.

        Args:
            simulation (GameSimulation): The game simulation to evaluate.

        Returns:
            List[torch.Tensor]: A list of reward tensors, one for each step in the simulation.
        """
        pass

    @abstractmethod
    def train(self, simulations: List[GameSimulation]):
        """
        Trains the model using a list of simulations.

        Args:
            simulations (List[GameSimulation]): A list of game simulations to train on.
        """
        pass

    def make_move(self, game: ChineseCheckersGame) -> (ChineseCheckersGame, Move):
        """
        Implements move-making behavior from IModel by using the `_chose_next_move` method.

        Args:
            game (ChineseCheckersGame): The current game state.

        Returns:
            (ChineseCheckersGame, Move): The updated game state and the chosen move.
        """
        move = self._chose_next_move(game)
        return game.apply_move(move), move
