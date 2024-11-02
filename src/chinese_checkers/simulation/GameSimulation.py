import re
from dataclasses import dataclass
from typing import List

from .SimulationData import SimulationData
from .SimulationMetadata import SimulationMetadata
from ..game.ChineseCheckersGame import ChineseCheckersGame
from ..model.IModel import IModel


@dataclass(frozen=True)
class GameSimulation:
    """Represents a game simulation, including metadata and simulation data.

    Attributes:
        metadata (SimulationMetadata): Metadata about the simulation, such as board size and game settings.
        data (SimulationData): Contains the data for the simulation, including player positions and historical moves.
    """

    metadata: SimulationMetadata
    data: SimulationData

    def to_game_sequence(self, sample_period: int = 1) -> List[ChineseCheckersGame]:
        """Converts the simulation data into a sequence of game states.

        Samples game states based on a specified period. Only frames where the frame
        index is a multiple of the sample period are included in the sequence.

        Args:
            sample_period (int, optional): Interval at which frames are sampled. Defaults to 1.

        Returns:
            List[ChineseCheckersGame]: List of game states at sampled frames.
        """
        game = ChineseCheckersGame.start_game(
            number_of_players=len(self.data.player_ids),
            board_size=self.metadata.board_size
        )

        for player, start_positions in zip(game.players, self.data.player_start_positions):
            player.positions = start_positions

        game_sequence = [game]
        current_turn = 0

        for move in self.data.historical_moves:
            game = game.apply_move(move)
            current_turn += 1

            if current_turn % sample_period == 0:
                game_sequence.append(game)

        return game_sequence

    @staticmethod
    def simulate_game(
            models: List[IModel],
            name: str,
            version: str,
            max_turns: int = 1000,
            board_size: int = 4,
            print_period: int = 0,
            show_coordinates: bool = False,
    ) -> "GameSimulation":
        """Runs a simulation where models play the game in turns until a player wins or max turns is reached.

        Args:
            models (List[IModel]): List of models that control the players in the simulation.
            name (str): Name of the simulation.
            version (str): Version string following the format 'v<major>.<minor>.<patch>'.
            max_turns (int, optional): Maximum number of turns to run the simulation. Defaults to 1000.
            board_size (int, optional): Size of the game board. Defaults to 4.
            print_period (int, optional): Frequency of print statements during simulation. Defaults to 0.
            show_coordinates (bool, optional): Whether to show coordinates on the board. Defaults to False.

        Returns:
            GameSimulation: A GameSimulation instance containing the metadata and data of the simulation.

        Raises:
            ValueError: If any of the input parameters are invalid.
        """
        GameSimulation._validate_input(models, name, version, max_turns, board_size, print_period)

        game = GameSimulation._setup_game(models, board_size, show_coordinates)
        move_history = GameSimulation._run_game_simulation(game, models, max_turns, print_period)

        return GameSimulation._construct_simulation_data(game, move_history, name, version, max_turns)

    @staticmethod
    def _validate_input(models, name, version, max_turns, board_size, print_period):
        """Validates the input parameters for the simulation.

        Args:
            models (List[IModel]): List of models to participate in the simulation.
            name (str): Name of the simulation.
            version (str): Version string.
            max_turns (int): Maximum number of turns.
            board_size (int): Size of the game board.
            print_period (int): Interval for print statements.

        Raises:
            ValueError: If any of the input parameters are invalid.
        """
        valid_lengths = [2, 3, 4, 6]
        version_pattern = r"^v\d+\.\d+\.\d+$"
        name_pattern = r"^[a-zA-Z0-9_-]+$"

        if len(models) not in valid_lengths:
            raise ValueError("Invalid number of models. Must be 2, 3, 4, or 6.")
        if not re.match(version_pattern, version):
            raise ValueError("Version must follow the 'v<major>.<minor>.<patch>' format.")
        if max_turns > 1000:
            raise ValueError("Max turns cannot exceed 1000.")
        if not re.match(name_pattern, name):
            raise ValueError("Name can only contain letters, numbers, underscores (_), and hyphens (-).")
        if print_period <= 0:
            raise ValueError("Print period must be greater than 0.")
        if board_size <= 1:
            raise ValueError("Board size must be greater than 1.")

    @staticmethod
    def _setup_game(models, board_size, show_coordinates):
        """Initializes a game with the given models and board settings.

        Args:
            models (List[IModel]): List of player models.
            board_size (int): Size of the game board.
            show_coordinates (bool): Whether to display coordinates on the board.

        Returns:
            ChineseCheckersGame: The initialized game instance.
        """
        game = ChineseCheckersGame.start_game(
            number_of_players=len(models),
            board_size=board_size
        ).update_printer_settings(
            show_coordinates=show_coordinates
        )
        return game

    @staticmethod
    def _run_game_simulation(game, models, max_turns, print_period):
        """Simulates a game by alternating moves among models until a win or max turns.

        Args:
            game (ChineseCheckersGame): The game instance.
            models (List[IModel]): List of player models.
            max_turns (int): Maximum number of turns for the simulation.
            print_period (int): Interval for printing game state.

        Returns:
            List: History of moves made during the simulation.

        Raises:
            Exception: If the game does not complete within the specified turns.
        """
        move_history = []

        while not game.is_game_won() and game.turn < max_turns:
            current_model = models[game.turn % len(models)]
            game, move = current_model.make_move(game)
            move_history.append(move)

        if game.turn >= max_turns:
            raise Exception(f"Game did not finish within {max_turns} turns.")

        return move_history

    @staticmethod
    def _construct_simulation_data(game, move_history, name, version, max_turns):
        """Constructs simulation data from the game results.

        Args:
            game (ChineseCheckersGame): The final game state.
            move_history (List): List of moves made during the simulation.
            name (str): Name of the simulation.
            version (str): Version string.
            max_turns (int): Maximum number of turns in the simulation.

        Returns:
            GameSimulation: The constructed GameSimulation instance.
        """
        start_positions = game.players
        game_metadata = SimulationMetadata(
            len(game.players),
            game.board.radius,
            max_turns,
            game.get_winner().player_id if game.is_game_won() else None,
            name,
            version
        )
        game_positions = SimulationData(
            [player.player_id for player in game.players],
            [player.positions for player in start_positions],
            [player.target_positions for player in game.players],
            move_history
        )

        return GameSimulation(game_metadata, game_positions)
