from typing import List

from torch import zeros_like, stack

from ..game.ChineseCheckersGame import ChineseCheckersGame
from ..model.IModel import IModel


class GameSimulation:

    def __init__(
            self,
            models: List[IModel],
            max_turns: int = 1000,
            board_size: int = 4,
            print_period: bool = 0,
            print_coordinates: bool = False
    ):
        """
        Simulates a game between the given models.

        Args:
            models: models to play the game
            max_turns: maximum number of turns to play
            board_size: size of the board
            print_period: period to print the game
            print_coordinates: whether to add the coordinates to the printed board
        """
        self.max_turns = max_turns
        self.game = ChineseCheckersGame.start_game(
            number_of_players=len(models), board_size=board_size)

        self.print_period = print_period
        self.game.update_printer_settings(print_coordinates=print_coordinates)
        self._print_game()
        self.models = models
        self.games = []

    def simulate_game(self) -> List[ChineseCheckersGame]:
        """
        Starts the game simulation.

        Returns:
            The winning game state, or None if the game was not won.
        """
        while not self.game.is_game_won() and self.game.turn < self.max_turns:
            self.games.append(self.game)
            self.game = self.models[self.game.turn % len(self.models)].make_move(self.game)
            self._print_game()

        if self.game.turn >= self.max_turns:
            return None

        self.games.append(self.game)
        return self.games

    def _print_game(self):
        if self.print_period and self.game.turn % self.print_period == 0:
            self.game.print()

    def tensor(self, max_turns=400):
        game_sequence_tensors = [game.tensor() for game in self.games]

        padding_size = max_turns - len(self.games)
        if padding_size > 0:
            padding = [zeros_like(game_sequence_tensors[0])] * padding_size
            game_sequence_tensors.extend(padding)

        return stack(game_sequence_tensors)

