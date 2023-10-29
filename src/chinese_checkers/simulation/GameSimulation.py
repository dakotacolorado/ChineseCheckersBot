from typing import List

from .GameSimulationData import GameMetadata, GamePositions, GameSimulationData
from ..game.ChineseCheckersGame import ChineseCheckersGame
from ..game.Move import Move
from ..game.Player import Player
from ..model.IModel import IModel


class GameSimulation:

    def __init__(
            self,
            models: List[IModel],
            max_turns: int = 1000,
            board_size: int = 4,
            print_period: int = 0,
            show_coordinates: bool = False
    ):
        """
        Initializes the game setup. The actual game progression (players making moves)
        starts only when the `simulate_game()` method is explicitly called.

        Args:
            models (List[IModel]): Models that will take turns playing the game.
            max_turns (int): Maximum allowable turns before the game is considered unfinished.
            board_size (int): Diameter of the board (measured by spaces or positions).
            print_period (int): Interval specifying how often the game state is printed.
            show_coordinates (bool): If True, game prints will include coordinates.
        """

        # set up the game
        self.max_turns: int = max_turns
        self.game: ChineseCheckersGame = ChineseCheckersGame.start_game(
            number_of_players=len(models), board_size=board_size)
        self.models: List[IModel] = models

        # set up the printer
        self.print_period: int = print_period
        self.game = self.game.update_printer_settings(show_coordinates=show_coordinates)

        # initialize the game state history
        self.start_positions: List[Player] = self.game.players
        self.move_history: List[Move] = []

    def simulate_game(self) -> Player:
        """
        Run the simulation where each model plays in turns until the game is won or max_turns is reached.

        Returns:
            Player: The winning player, or None if the game did not conclude.
        """
        while not self.game.is_game_won() and self.game.turn < self.max_turns:
            current_model = self.models[self.game.turn % len(self.models)]
            self.game, move = current_model.make_move(self.game)
            self._print_game_state_if_required()
            self.move_history.append(move)

        if self.game.turn >= self.max_turns:
            raise Exception("Game did not finish within the maximum allowed turns.")

        return self.game.get_winner()

    def get_simulation_data(self, name: str, version: str) -> GameSimulationData:
        """
        Generate a record of the played game for storage purposes.

        Args:
            name (str): Descriptive name for the game simulation.
            version (str): Version identifier for the game simulation.

        Returns:
            GameSimulationData: Data structure encapsulating details of the played game.
        """
        game_metadata = GameMetadata(
            len(self.game.players),
            self.game.board.radius,
            self.max_turns,
            self.game.get_winner().player_id,
            name,
            version
        )
        game_positions = GamePositions(
            [player.player_id for player in self.game.players],
            [player.positions for player in self.start_positions],
            [player.target_positions for player in self.game.players],
            self.move_history
        )

        return GameSimulationData(game_metadata, game_positions)

    def _print_game_state_if_required(self):
        """Prints the current game state if the turn is a multiple of print_period."""
        if self.print_period and self.game.turn % self.print_period == 0:
            self.game.print()
