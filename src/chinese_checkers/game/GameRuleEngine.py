from typing import List, Set

from .Move import Move
from .Player import Player
from .Position import Position
from ..geometry.Vector import Vector
from ..geometry.Hexagram import Hexagram


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
        self.board_positions = set(hexagram.hexagram_points)  
        self.occupied_positions = {
            pos for p in other_players for pos in p.positions
        } | set(current_player.positions)
        self.unit_moves = hexagram.hexagon_points
        self.current_player = current_player

    def get_next_moves_for_player(self) -> List[Move]:
        """
        Generates the next possible moves for the current player.

        Returns:
            List[Move]: The list of possible moves.
        """
        next_moves = []

        for position in self.current_player.positions:
            for unit_move in self.unit_moves:
                # Check for simple moves
                new_position = Position(position.i + unit_move.i, position.j + unit_move.j)
                if self._is_position_valid(new_position):
                    next_moves.append(Move(unit_move.i, unit_move.j, position))

                # Check for hop moves and chain hops
                hops = self._get_hop_moves(position, {position}, unit_move)
                next_moves.extend(hops)

        return next_moves

    def _get_hop_moves(
            self,
            position: Position,
            visited: Set[Position],
            unit_move: Vector,
            root_position: Position = None
    ) -> List[Move]:
        """
        Recursively finds all possible hop moves from a given position.

        Note. Ensure that the visited set is initialized with the starting position.

        Args:
            position: position to check for hop moves from
            visited: set of visited positions
            unit_move: the unit move to check for hops in

        Returns:
            The list of hop moves from the given position.
        """
        if root_position is None:
            root_position = position

        hops = []
        adjacent_position = position + unit_move
        jump_position = position + unit_move * 2

        # Check for hop moves
        if self._is_position_valid(jump_position) \
                and adjacent_position in self.occupied_positions \
                and jump_position not in visited:
            # Add the hop move
            move_from_root_position = jump_position - root_position
            hops.append(Move(move_from_root_position.i, move_from_root_position.j, root_position))
            visited.add(jump_position)

            # Recursive call for chained hops
            for unit in self.unit_moves:
                hops.extend(self._get_hop_moves(jump_position, visited, unit, root_position))
        return hops

    def _is_position_in_bounds(self, position: Position) -> bool:
        """
        Checks if a position is within the bounds of the board.
        Args:
            position: position to check

        Returns:
            True if the position is within the bounds of the board, False otherwise.
        """
        return position in self.board_positions

    def _is_position_open(self, position: Position) -> bool:
        """
        Checks if a position is open.
        Args:
            position: position to check

        Returns:
            True if the position is open, False otherwise.
        """
        return position not in self.occupied_positions

    def _is_position_valid(self, position: Position) -> bool:
        """
        Checks if a position is valid.
        Args:
            position: position to check

        Returns:
            True if the position is valid, False otherwise.
        """
        return self._is_position_in_bounds(position) and self._is_position_open(position)
