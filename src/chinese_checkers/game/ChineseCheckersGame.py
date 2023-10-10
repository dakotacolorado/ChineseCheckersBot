from typing import List, Dict

from src.chinese_checkers.geometry.hexagram import Hexagram
from .position import Position
from .move import Move
from .player import Player


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
                hexagram.hexagram_corner_points[(corner_index + 3) % 6]
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

    def __get_all_positions(self):
        return [
            p for player in self.players
            for p in player.positions
        ]

    def __str__(self):
        j_min = min(self.board.hexagon_points, key=lambda p: p.j)
        j_max = max(self.board.hexagon_points, key=lambda p: p.j)

        i_min = min(self.board.hexagon_points, key=lambda p: p.i)
        i_max = max(self.board.hexagon_points, key=lambda p: p.i)


        def get_character_for_position(i: int, j: int) -> str:
            pos = Position(i, j)
            if pos in self.__get_all_positions():
                return "X"
            elif pos in self.board.hexagram_points:
                return "O"
            else:
                return "_"

        board_string = "\n".join([
            "".join([
                get_character_for_position(i, j)
                for j in range(-self.board.radius-2, self.board.radius + 3)
            ])
            for i in range(-self.board.radius-2, self.board.radius + 3)
        ])

        return f"""
Turn: {self.turn}
Current Player: {self.get_current_player()}
Board:
{board_string}
Hexagram: 
{self.board}
"""
