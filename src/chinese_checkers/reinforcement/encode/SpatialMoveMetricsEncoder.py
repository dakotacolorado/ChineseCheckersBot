import torch
from chinese_checkers.game import Move, ChineseCheckersGame, Position
from typing import List

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
        target_centroid = self._calculate_target_centroid(current_player.target_positions)
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
    def _calculate_target_centroid(target_positions: List[Position]) -> Position:
        """
        Calculates the centroid of target positions.

        Args:
            target_positions (List[Position]): List of target positions for the player.

        Returns:
            Position: The centroid of the target positions.
        """
        x_coords = [pos.i for pos in target_positions]
        y_coords = [pos.j for pos in target_positions]
        centroid_x = sum(x_coords) / len(target_positions)
        centroid_y = sum(y_coords) / len(target_positions)
        return Position(int(centroid_x), int(centroid_y))

    @staticmethod
    def _distance_from_target_centroid(position: Position, centroid: Position, board_radius: float) -> float:
        """
        Calculates the normalized distance between a given position and the target centroid.

        Args:
            position (Position): The position to measure.
            centroid (Position): The target centroid.
            board_radius (float): The board radius for normalization.

        Returns:
            float: The Euclidean distance between the position and centroid, normalized by board radius.
        """
        distance = position.distance(centroid)
        return distance / board_radius
