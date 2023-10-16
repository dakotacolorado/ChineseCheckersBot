import random
from typing import List
from random import choice
import numpy as np

from .IModel import IModel
from ..game import Player, Move, Position
from ..geometry.Vector import Vector


class CentroidModel(IModel):

    def _chose_next_move(self, current_player: Player, other_players: List[Player], moves: List[Move]) -> Move:
        """
        Simple converging algorithm to generate the bootstrap data for the neural network.

        Greedy algorithm to choose the move that minimizes the distance between the centroid of the
        current player's positions and the target positions. Simple converging algorithm to generate
        the bootstrap data for the neural network.

        Chose the move that has the minimum distance from a target position, if it is not already in a
        target position. Defensive moves are not considered.

        Args:
            current_player: Current player.
            other_players: Other players in the game.
            moves: List of possible moves for current player.

        Returns:
            Next move to make.
        """
        closest_distance = float('inf')
        next_moves = []

        # Calculate the centroid of the current player's positions
        target_position = sorted(
            current_player.target_positions,
            key=lambda p: p.distance(Vector(0, 0)),
            reverse=True
        )[0]

        other_player_targets = [
            p
            for player in other_players
            for p in player.target_positions
        ]
        priority_positions = [
            p
            for p in current_player.positions
            if p in other_player_targets
        ]

        # Find the closest target position for each piece
        for move in moves:
            if move.position in current_player.target_positions:
                if random.Random().random() > 0.05:
                    continue
            if len(priority_positions) > 0 and move.position not in priority_positions:
                if random.Random().random() > 0.1:
                    continue

            new_position = move.apply()

            # Calculate the distance from the new position to the target positions
            distance = new_position.distance(target_position)

            if distance < closest_distance:
                closest_distance = distance
                next_moves = [move]
            elif distance == closest_distance:
                next_moves.append(move)

        if not next_moves:
            raise Exception("No next move found")

        next_move = choice(next_moves)

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
