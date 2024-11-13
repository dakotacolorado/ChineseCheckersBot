from typing import List, Dict

from chinese_checkers.geometry import Vector
from .GameRuleEngine import GameRuleEngine
from .Move import Move
from .Player import Player
import pydash as _

from .Position import Position
from ..geometry.Hexagram import Hexagram
from ..geometry.Printer import Printer


class ChineseCheckersGame:

    @staticmethod
    def start_game(number_of_players: int = 2, board_size: int = 4) -> 'ChineseCheckersGame':
        """
        Starts a new game of Chinese Checkers.

        Chinese Checkers Wiki: https://en.wikipedia.org/wiki/Chinese_checkers

        Args:
            number_of_players: Number of players (2, 3, 4, or 6). default: 2
            board_size: Optionally you can provide a board size (> 0).

        Raises:
            ValueError: If number_of_players is not 2, 3, 4, or 6.
            ValueError: If board_size is not a positive integer.
        """

        # Validate number_of_players
        valid_player_counts = {2, 3, 4, 6}
        if number_of_players not in valid_player_counts:
            raise ValueError(
                f"Invalid number of players: {number_of_players}. "
                f"Must be one of {valid_player_counts}."
            )

        # Validate board_size
        if not isinstance(board_size, int) or board_size <= 0:
            raise ValueError(
                f"Invalid board size: {board_size}. "
                "Board size must be a positive integer."
            )

        hexagram = Hexagram(board_size)

        # Depending on the amount of players, a different subset of starting corners will be used.
        starting_player_corners: Dict[int, List[int]] = {
            2: [0, 3],
            3: [0, 2, 4],
            4: [0, 1, 3, 4],
            6: [0, 1, 2, 3, 4, 5],
        }

        # Create a player for each starting corner.
        players = [
            Player(
                [Position(v.i, v.j) for v in hexagram.hexagram_corner_points[corner_index]],
                [Position(v.i, v.j) for v in hexagram.hexagram_corner_points[(corner_index + 3) % 6]],
                str(corner_index)
            )
            for corner_index in starting_player_corners[number_of_players]
        ]

        return ChineseCheckersGame(players, 0, hexagram)

    def __init__(
            self,
            players: List[Player],
            turn: int = 0,
            board: Hexagram = Hexagram(4),
            printer: Printer = Printer()
    ):
        valid_player_counts = {2, 3, 4, 6}
        if len(players) not in valid_player_counts:
            raise ValueError(
                f"Invalid number of players: {len(players)}. "
                f"Must be one of {valid_player_counts}."
            )

        self.players = players
        self.turn = turn
        self.board = board
        self.printer = printer

        # Validate that all players have unique positions
        all_positions = self.get_all_positions()
        if len(all_positions) != len(set(all_positions)):
            raise ValueError("Players must have unique positions.")

    def __eq__(self, other: 'ChineseCheckersGame') -> bool:
        return self.players == other.players \
            and self.turn == other.turn \
            and self.board == other.board

    def __hash__(self):
        return hash((tuple(self.players), self.board.radius))

    def apply_move(self, move: Move) -> 'ChineseCheckersGame':
        return ChineseCheckersGame(
            [
                player.apply_move(move) if player == self.get_current_player() else player
                for player in self.players
            ],
            self.turn + 1,
            self.board,
            self.printer
        )

    def get_current_player(self) -> Player:
        """
        Returns the current player based on the turn number.
        Returns:
            Player: The current player.
        """
        return self.players[self.turn % len(self.players)]

    def get_other_players(self) -> List[Player]:
        return [p for p in self.players if p != self.get_current_player()]

    def get_all_positions(self):
        return [
            p for player in self.players
            for p in player.positions
        ]

    def get_next_moves(self, remove_backwards_moves: bool = False) -> List[Move]:
        engine = GameRuleEngine(
            self.get_current_player(),
            [p for p in self.players if p != self.get_current_player()],
            self.board
        )
        if remove_backwards_moves:
            return _.filter_(
                engine.get_next_moves_for_player(),
                lambda m: not self._is_move_backwards(m, self.get_current_player().player_id)
            )
        return engine.get_next_moves_for_player()

    normalized_player_directions = {
        "0": Vector(-0.7071068, -0.7071068),
        "1": Vector(0.4472136, -0.8944272),
        "2": Vector(0.8944272, -0.4472136),
        "3": Vector(0.7071068, 0.7071068),
        "4": Vector(-0.4472136, 0.8944272),
        "5": Vector(-0.8944272, 0.4472136),
    }

    @staticmethod
    def _is_move_backwards(move: Move, player_id: str) -> bool:
        return move.dot(ChineseCheckersGame.normalized_player_directions[player_id]) < 0
        pass

    def is_game_won(self) -> bool:
        """
        Returns true if the game is won.
        Returns:
            bool: True if the game is won.
        """
        return _.some(self.players, lambda p: p.has_player_reached_target())

    def get_winner(self) -> Player:
        """
        Returns the winner of the game.
        Returns:
            Player: The winner of the game.
        """
        return _.find(self.players, lambda p: p.has_player_reached_target())

    def update_printer_settings(self, **printer_settings) -> "ChineseCheckersGame":
        return ChineseCheckersGame(
            self.players,
            self.turn,
            self.board,
            Printer(**printer_settings)
        )

    def print(self, **print_settings):
        self.printer.print(
            self.board.hexagram_points,
            *[p.positions for p in self.players],
            game_metrics={
                "Turn": self.turn,
                "Current Player": self.get_current_player().player_id,
                "Next Moves": len(self.get_next_moves()),
            },
            **print_settings
        )