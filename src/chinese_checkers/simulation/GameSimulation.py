import logging
import re
from dataclasses import dataclass
from typing import List
from IPython.display import Image

from .GameSimulationAnimation import GameSimulationAnimation
from .SimulationData import SimulationData
from .SimulationMetadata import SimulationMetadata
from ..game.ChineseCheckersGame import ChineseCheckersGame
from ..model.IModel import IModel

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GameSimulation:
    """Represents a game simulation, including metadata and simulation data.

    Attributes:
        metadata (SimulationMetadata): Metadata about the simulation, such as board size and game settings.
        data (SimulationData): Contains the data for the simulation, including player positions and historical moves.
    """

    metadata: SimulationMetadata
    data: SimulationData

    @staticmethod
    def simulate_game(
            models: List[IModel],
            name: str,
            version: str,
            max_turns: int = 1000,
            board_size: int = 4,
            print_period: int = None,
            show_coordinates: bool = False,
    ) -> "GameSimulation":
        """Runs a simulation where models play the game in turns until a player wins or max turns is reached."""
        logger.debug("Starting game simulation with parameters: "
                    f"name={name}, version={version}, max_turns={max_turns}, board_size={board_size}")

        GameSimulation._validate_input(models, name, version, max_turns, board_size, print_period)
        start_game = GameSimulation._setup_game(models, board_size, show_coordinates)
        move_history, final_game = GameSimulation._run_game_simulation(start_game, models, max_turns, print_period)
        return GameSimulation._construct_simulation_data(final_game, move_history, name, version, max_turns)

    @staticmethod
    def _validate_input(models, name, version, max_turns, board_size, print_period):
        """Validates the input parameters for the simulation."""
        logger.debug("Validating input parameters.")
        valid_lengths = [2, 3, 4, 6]
        version_pattern = r"^v\d+\.\d+\.\d+$"
        name_pattern = r"^[a-zA-Z0-9_-]+$"

        if len(models) not in valid_lengths:
            logger.error(f"Invalid number of models: {len(models)}")
            raise ValueError("Invalid number of models. Must be 2, 3, 4, or 6.")
        if not re.match(version_pattern, version):
            logger.error(f"Invalid version format: {version}")
            raise ValueError("Version must follow the 'v<major>.<minor>.<patch>' format.")
        if max_turns > 10000:
            logger.error(f"Max turns exceeded: {max_turns}")
            raise ValueError("Max turns cannot exceed 10000.")
        if not re.match(name_pattern, name):
            logger.error(f"Invalid simulation name: {name}")
            raise ValueError("Name can only contain letters, numbers, underscores (_), and hyphens (-).")
        if print_period and print_period <= 0:
            logger.error(f"Invalid print period: {print_period}")
            raise ValueError("Print period must be greater than 0.")
        if board_size <= 1:
            logger.error(f"Invalid board size: {board_size}")
            raise ValueError("Board size must be greater than 1.")
        logger.info("Input validation completed successfully.")

    @staticmethod
    def _setup_game(models, board_size, show_coordinates):
        """Initializes a game with the given models and board settings."""
        logger.debug(f"Setting up game with board_size={board_size} and show_coordinates={show_coordinates}")
        game = ChineseCheckersGame.start_game(
            number_of_players=len(models),
            board_size=board_size
        ).update_printer_settings(
            show_coordinates=show_coordinates
        )
        return game

    @staticmethod
    def _run_game_simulation(game, models, max_turns, print_period):
        """Simulates a game by alternating moves among models until a win or max turns."""
        logger.debug("Running game simulation.")
        move_history = []
        while not game.is_game_won() and game.turn < max_turns:
            current_model = models[game.turn % len(models)]
            logger.debug(f"Turn {game.turn}, current player: {game.get_current_player().player_id}")
            game, move = current_model.make_move(game)
            move_history.append(move)
            if print_period and game.turn % print_period == 1:
                game.print()

        if game.is_game_won():
            logger.info(f"Game won by player {game.get_winner().player_id} in {game.turn} turns.")
        else:
            logger.warning(f"Game did not finish within the maximum of {max_turns} turns.")
        return move_history, game

    @staticmethod
    def _construct_simulation_data(game, move_history, name, version, max_turns):
        """Constructs simulation data from the game results."""
        start_positions = game.players
        is_won = game.is_game_won()
        winner = game.get_winner().player_id if is_won else None
        logger.info(f"Constructing simulation data. Winner: {winner} (is_game_won={is_won})")

        game_metadata = SimulationMetadata(
            len(game.players),
            game.board.radius,
            max_turns,
            winner,
            name,
            version
        )
        game_positions = SimulationData(
            [player.player_id for player in game.players],
            [player.positions for player in start_positions],
            [player.target_positions for player in game.players],
            move_history
        )

        logger.debug("Simulation data constructed successfully.")
        return GameSimulation(game_metadata, game_positions)

    def _to_game_sequence(self, sample_period: int = 1) -> List[ChineseCheckersGame]:
        """Converts the simulation data into a sequence of game states."""
        logger.debug(f"Generating game sequence with sample_period={sample_period}")
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

        logger.debug("Game sequence generation completed.")
        return game_sequence

    def display(self, sample_period: int = 10) -> Image:
        logger.info("Displaying game animation.")
        game_animation = GameSimulationAnimation(self._to_game_sequence(sample_period=sample_period))
        return game_animation.display()

    def save_animation(self, file_path: str = None, sample_period: int = 10, fps: int = 10):
        if file_path is None:
            file_path = f"{self.metadata.name}-{self.metadata.version}.mp4"
        logger.info(f"Saving game animation to {file_path}")
        game_animation = GameSimulationAnimation(self._to_game_sequence(sample_period=sample_period))
        game_animation.save_to_file(file_path=file_path, fps=fps)
        logger.info("Game animation saved successfully.")
