from typing import List
import numpy as np

from .IModel import IModel
from ..game import Player, Move, Position
from ..geometry.Vector import Vector


class CentroidModel(IModel):

    def _chose_next_move(self, current_player: Player, other_players: List[Player], moves: List[Move]) -> Move:
        """
        Greedy algorithm to choose the move that minimizes the distance between the centroid of the
        current player's positions and the target positions.

        Args:
            current_player: Current player.
            other_players: Other players in the game.
            moves: List of possible moves for current player.

        Returns:
            Next move to make.
        """
        closest_distance = float('inf')
        next_move = None
        target_centroid = CentroidModel._calculate_centroid(current_player.target_positions)
        for move in moves:
            new_position = move.apply()
            distance = new_position.distance(target_centroid)
            if distance < closest_distance:
                closest_distance = distance
                next_move = move

        if next_move is None:
            raise Exception("No next move found")

        return next_move


    @staticmethod
    def _calculate_centroid(positions: List[Position]) -> Vector:
        """
        Calculate the centroid of a list of positions.

        Args:
            positions: List of positions.

        Returns:
            Centroid of positions.
        """
        if not positions:
            return None

        vector_array = np.array([(vector.i, vector.j) for vector in positions])

        centroid = np.mean(vector_array, axis=0)

        return Vector(centroid[0], centroid[1])
