from dataclasses import dataclass
from typing import List

from .SimulationData import SimulationData
from .SimulationMetadata import SimulationMetadata
from ..game.ChineseCheckersGame import ChineseCheckersGame
from ..game.Move import Move
from ..game.Player import Player
from ..model.IModel import IModel
from ..geometry.Hexagram import Hexagram


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

        Args:
            models (List[IModel]): Models that will take turns playing the game.
            name (str): Descriptive name for the game simulation.
            version (str): Version identifier for the game simulation.
            max_turns (int): Maximum allowable turns before the game is considered unfinished.
            board_size (int): Diameter of the board (measured by spaces or positions).
            print_period (int): Interval specifying how often the game state is printed.
            show_coordinates (bool): If True, game prints will include coordinates.
        """

        # set up the game
        game: ChineseCheckersGame = ChineseCheckersGame.start_game(
            number_of_players=len(models),
            board_size=board_size
        ).update_printer_settings(
            show_coordinates=show_coordinates
        )

        # run the simulation and record the moves
        start_positions: List[Player] = game.players
        move_history: List[Move] = []

        while not game.is_game_won() and game.turn < max_turns:
            current_model = models[game.turn % len(models)]
            game, move = current_model.make_move(game)
            GameSimulation._print_game_state_if_required(print_period, game)
            move_history.append(move)

            if game.turn >= max_turns:
                raise Exception("Game did not finish within the maximum allowed turns.")

        # format and return the simulation data
        game_metadata = SimulationMetadata(
            len(game.players),
            game.board.radius,
            max_turns,
            game.get_winner().player_id,
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

    @staticmethod
    def _print_game_state_if_required(print_period: int, game: ChineseCheckersGame):
        """Prints the current game state if the turn is a multiple of print_period."""
        if print_period and game.turn % print_period == 0:
            game.print()

    def to_game_sequence(self) -> List[ChineseCheckersGame]:
        """Converts the simulation to a list of ChineseCheckersGame objects."""
        players: List[Player] = [
            Player(start_positions, target_positions, player_id)
            for start_positions, target_positions, player_id
            in zip(
                self.data.player_start_positions,
                self.data.player_target_positions,
                self.data.player_ids,
            )
        ]

        game = ChineseCheckersGame(players, board=Hexagram(self.metadata.board_size))
        game_sequence = [game]

        for move in self.data.historical_moves:
            game: ChineseCheckersGame = game.apply_move(move)
            game_sequence.append(game)

        return game_sequence
