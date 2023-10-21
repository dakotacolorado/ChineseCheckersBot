from typing import List

from .GameSimulationData import GameData, DirectoryAttributes, GameSimulationData
from ..game.ChineseCheckersGame import ChineseCheckersGame
from ..game.Player import Player
from ..model.IModel import IModel


class GameSimulation:

    def __init__(
            self,
            models: List[IModel],
            max_turns: int = 1000,
            board_size: int = 4,
            print_period: int = 0,
            print_coordinates: bool = False
    ):
        """
        Initializes the game setup. The actual game progression (players making moves)
        starts only when the `simulate_game()` method is explicitly called.

        Args:
            models (List[IModel]): Models that will take turns playing the game.
            max_turns (int): Maximum allowable turns before the game is considered unfinished.
            board_size (int): Diameter of the board (measured by spaces or positions).
            print_period (int): Interval specifying how often the game state is printed.
            print_coordinates (bool): If True, game prints will include coordinates.
        """
        self.max_turns = max_turns
        self.game = ChineseCheckersGame.start_game(
            number_of_players=len(models), board_size=board_size
        )
        self.print_period = print_period
        self.game.update_printer_settings(print_coordinates=print_coordinates)
        self._print_initial_game_state()
        self.models = models
        self.game_state_history = []  # Records the state of players after every turn.

    def _print_initial_game_state(self):
        """Print the initial state of the game."""
        if self.print_period:
            self.game.print()

    def simulate_game(self) -> Player:
        """
        Run the simulation where each model plays in turns until the game is won or max_turns is reached.

        Returns:
            Player: The winning player, or None if the game did not conclude.
        """
        while not self.game.is_game_won() and self.game.turn < self.max_turns:
            self.game_state_history.append(self.game.players)
            current_model = self.models[self.game.turn % len(self.models)]
            self.game = current_model.make_move(self.game)
            self._print_game_state_if_required()

        if self.game.turn >= self.max_turns:
            raise Exception("Game did not finish within the maximum allowed turns.")

        return self.game.get_winner()

    def _print_game_state_if_required(self):
        """Prints the current game state if the turn is a multiple of print_period."""
        if self.print_period and self.game.turn % self.print_period == 0:
            self.game.print()

    def create_game_simulation_record(self, name: str, version: str) -> GameSimulationData:
        """
        Generate a record of the played game for storage purposes.

        Args:
            name (str): Descriptive name for the game simulation.
            version (str): Version identifier for the game simulation.

        Returns:
            GameSimulationData: Data structure encapsulating details of the played game.
        """
        directory_attributes = DirectoryAttributes(
            len(self.game.players),
            self.game.board.radius,
            self.max_turns,
            name,
            version,
            self.game.get_winner().player_id
        )
        game_data = GameData(self.game_state_history)

        return GameSimulationData(directory_attributes, game_data)

    # Ill add this back when I finish storing the game data.  This will be used to load it into the NN
    # def get_game_tensor(self, max_turns: int = 400):
    #     """
    #     Convert games into tensor representation, padding with zeros if needed.
    #
    #     Args:
    #         max_turns (int): Maximum number of turns for tensor padding.
    #
    #     Returns:
    #         Tensor: Stacked tensor representation of the games.
    #     """
        # game_sequence_tensors = [player.tensor() for game in self.player_history]
        #
        # padding_size = max_turns - len(self.games)
        # if padding_size > 0:
        #     padding = [zeros_like(game_sequence_tensors[0])] * padding_size
        #     game_sequence_tensors.extend(padding)
        #
        # return stack(game_sequence_tensors)
