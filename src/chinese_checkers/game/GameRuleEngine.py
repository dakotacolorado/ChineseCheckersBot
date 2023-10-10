from typing import List

from src.chinese_checkers.geometry.hexagram import Hexagram
from .move import Move
from .player import Player
from .position import Position


class GameRuleEngine:
    """
    This class is responsible for checking the validity of moves and generating the next possible moves for a player.
    """

    def __init__(
            self,
            current_player: Player,
            other_players: List[Player],
            hexagram: Hexagram
    ):
        self.board_positions = hexagram.hexagram_points
        self.occupied_positions = [
                                      pos for p in other_players for pos in p.positions
                                  ] + current_player.positions
        self.unit_moves = hexagram.hexagon_points
        self.current_player = current_player

    def get_next_moves_for_player(self) -> List[Move]:
        next_moves = []

        for position in self.current_player.positions:
            for unit_move in self.unit_moves:
                # Check for simple moves
                new_position = Position(position.i + unit_move.i, position.j + unit_move.j)
                if self.__is_position_valid(new_position):
                    next_moves.append(Move(unit_move.i, unit_move.j, position))

                # Check for hop moves and chain hops
                hops = self.__get_hop_moves(position, [], unit_move)
                next_moves.extend(hops)

        return next_moves

    def __get_hop_moves(
            self,
            position: Position,
            visited: List[Position],
            unit_move: Position
    ) -> List[Move]:
        hops = []
        adjacent_position = Position(position.i + unit_move.i, position.j + unit_move.j)
        jump_position = Position(position.i + 2 * unit_move.i, position.j + 2 * unit_move.j)

        if self.__is_position_valid(
                jump_position) and adjacent_position in self.occupied_positions and jump_position not in visited:
            hops.append(Move(2 * unit_move.i, 2 * unit_move.j, position))
            visited.append(jump_position)
            # Recursive call for chained hops
            for unit in self.unit_moves:
                hops.extend(self.__get_hop_moves(jump_position, visited, unit))

        return hops

    def __is_position_in_bounds(self, position: Position) -> bool:
        return position in self.board_positions

    def __is_position_open(self, position: Position) -> bool:
        return position not in self.occupied_positions

    def __is_position_valid(self, position: Position) -> bool:
        return self.__is_position_in_bounds(position) and self.__is_position_open(position)
