import re
from dataclasses import dataclass
from typing import List

from .SimulationData import SimulationData
from .SimulationMetadata import SimulationMetadata
from ..game.ChineseCheckersGame import ChineseCheckersGame
from ..model.IModel import IModel


@dataclass(frozen=True)
class GameSimulation:
    metadata: SimulationMetadata
    data: SimulationData

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
        """
        Run the simulation where each model plays in turns until the game is won or max_turns is reached.
        """
        GameSimulation._validate_input(models, name, version, max_turns, board_size, print_period)

        game = GameSimulation._setup_game(models, board_size, show_coordinates)
        move_history = GameSimulation._run_game_simulation(game, models, max_turns, print_period)

        return GameSimulation._construct_simulation_data(game, move_history, name, version, max_turns)

    @staticmethod
    def _validate_input(models, name, version, max_turns, board_size, print_period):
        valid_lengths = [2, 3, 4, 6]
        version_pattern = r"^v\d+\.\d+\.\d+$"
        name_pattern = r"^[a-zA-Z0-9_-]+$"

        if len(models) not in valid_lengths:
            raise ValueError("Invalid number of models. Must be 2, 3, 4, or 6.")
        if not re.match(version_pattern, version):
            raise ValueError(
                "Version must follow the 'v<major-version-int>.<minor-version-int>.<patch-version-int>' template.")
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
        game = ChineseCheckersGame.start_game(
            number_of_players=len(models),
            board_size=board_size
        ).update_printer_settings(
            show_coordinates=show_coordinates
        )
        return game

    @staticmethod
    def _run_game_simulation(game, models, max_turns, print_period):
        move_history = []

        while not game.is_game_won() and game.turn < max_turns:
            current_model = models[game.turn % len(models)]
            game, move = current_model.make_move(game)
            move_history.append(move)

        if game.turn >= max_turns:
            raise Exception("Game did not finish within the specified {} turns.".format(max_turns))

        return move_history

    @staticmethod
    def _construct_simulation_data(game, move_history, name, version, max_turns):
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
