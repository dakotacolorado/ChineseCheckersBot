from typing import List, Dict

from src.chinese_checkers.geometry.Hexagram import Hexagram
from .GameRuleEngine import GameRuleEngine
from .Move import Move
from .Player import Player
import pydash as _


class ChineseCheckersGame:

    @staticmethod
    def start_game(number_of_players: int = 2, board_size: int = 4) -> 'ChineseCheckersGame':
        """
        Starts a new game of Chinese Checkers.

        Chinese Checkers Wiki: https://en.wikipedia.org/wiki/Chinese_checkers

        Args:
            number_of_players: Number of players (2,3,4, or 6). default: 2
            board_size: Optionally you can provide a board size (> 0).
        """

        hexagram = Hexagram(board_size)

        # Depending on the amount of players a different subset of starting corners will be used.
        starting_player_corners: Dict[int, List[int]] = {
            2: [1, 4],
            3: [1, 3, 5],
            4: [1, 2, 4, 5],
            6: [1, 2, 3, 4, 5, 6],
        }

        # Create a player for each starting corner.
        players = [
            Player(
                hexagram.hexagram_corner_points[corner_index],
                # opposite corner is the corner with the same index + 3 (mod 6)
                hexagram.hexagram_corner_points[(corner_index + 3) % 6],
            )
            for corner_index in starting_player_corners[number_of_players]
        ]

        return ChineseCheckersGame(players, 0, hexagram)

    def __init__(
            self,
            players: List[Player],
            turn: int = 0,
            board: Hexagram = Hexagram(4)
    ):
        self.players = players
        self.turn = turn
        self.board = board

    def apply_move(self, move: Move) -> 'ChineseCheckersGame':
        return ChineseCheckersGame(
            [
                player.apply_move(move) if player == self.get_current_player() else player
                for player in self.players
            ],
            self.turn + 1,
            self.board
        )

    def get_current_player(self) -> Player:
        """
        Returns the current player based on the turn number.
        Returns:
            Player: The current player.
        """
        return self.players[self.turn % len(self.players)]

    def get_all_positions(self):
        return [
            p for player in self.players
            for p in player.positions
        ]

    def get_next_moves(self) -> List[Move]:
        engine = GameRuleEngine(
            self.get_current_player(),
            [p for p in self.players if p != self.get_current_player()],
            self.board
        )
        return engine.get_next_moves_for_player()

    def is_game_won(self) -> bool:
        """
        Returns true if the game is won.
        Returns:
            bool: True if the game is won.
        """
        return _.some(self.players, lambda p: p.has_player_reached_target())
