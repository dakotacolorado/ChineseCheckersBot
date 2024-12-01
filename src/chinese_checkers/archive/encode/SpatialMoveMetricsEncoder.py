import torch
from typing import List

from chinese_checkers.game import ChineseCheckersGame, Move, Position, Player
from chinese_checkers.geometry import Vector


class SpatialMoveMetricsEncoder:

    def encode(self, move: Move, game: ChineseCheckersGame) -> torch.Tensor:
        """
        Encodes a move as a tensor with three normalized metrics:
        - Move length (normalized by board radius)
        - Start position distance from target centroid (normalized by board radius)
        - End position distance from target centroid (normalized by board radius)

        Args:
            move (Move): The move to encode.
            game (ChineseCheckersGame): The game state to provide player context, including board radius.

        Returns:
            torch.Tensor: A tensor [move_length, start_distance, end_distance].
        """
        current_player = game.get_current_player()
        target_centroid = self._calculate_centroid(current_player.target_positions)
        board_radius = game.board.radius

        move_length = self._calculate_move_length(move, board_radius)
        start_distance = self._distance_from_target_centroid(move.position, target_centroid, board_radius)
        end_position = move.apply()
        end_distance = self._distance_from_target_centroid(end_position, target_centroid, board_radius)

        projection_onto_shortest_path = SpatialMoveMetricsEncoder._projection_onto_shortest_path(move, target_centroid)

        return torch.tensor([move_length, start_distance, end_distance, projection_onto_shortest_path], dtype=torch.float32)

    @staticmethod
    def _calculate_move_length(move: Move, board_radius: float) -> float:
        """
        Calculates the normalized length of the move based on Euclidean distance.

        Args:
            move (Move): The move to calculate length for.
            board_radius (float): The board radius for normalization.

        Returns:
            float: The Euclidean distance between start and end positions, normalized by board radius.
        """
        end_position = move.apply()
        distance = move.position.distance(end_position)
        return distance / board_radius

    @staticmethod
    def _calculate_centroid(positions: List[Position]) -> Vector:
        if not positions:
            return Vector(0.0, 0.0)
        avg_i = sum(pos.i for pos in positions) / len(positions)
        avg_j = sum(pos.j for pos in positions) / len(positions)
        return Vector(avg_i, avg_j)

    @staticmethod
    def _distance_from_target_centroid(position: Position, centroid: Vector, board_radius: float) -> float:
        distance = position.distance(centroid)
        return distance / board_radius

    @staticmethod
    def _projection_onto_shortest_path(move: Move, target_centroid: Vector) -> float:
        position: Position = move.position
        move_vector = move.apply() - position
        shortest_path = target_centroid - position

        if shortest_path.norm() == 0:
            return 0.0

        projection = move_vector.dot(shortest_path) / shortest_path.norm()
        return projection
