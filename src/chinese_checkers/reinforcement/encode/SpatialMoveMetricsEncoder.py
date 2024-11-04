import torch
from chinese_checkers.game import Move, ChineseCheckersGame, Position
from typing import List

from chinese_checkers.geometry.Vector import Vector


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
        board_radius = game.board.radius  # Assuming game.board provides the radius

        move_length = self._calculate_move_length(move, board_radius)
        start_distance = self._distance_from_target_centroid(move.position, target_centroid, board_radius)
        end_position = move.apply()
        end_distance = self._distance_from_target_centroid(end_position, target_centroid, board_radius)

        return torch.tensor([move_length, start_distance, end_distance], dtype=torch.float32)

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
    def _distance_from_target_centroid(position: Position, centroid: Position, board_radius: float) -> float:
        distance = position.distance(centroid)
        return distance / board_radius
