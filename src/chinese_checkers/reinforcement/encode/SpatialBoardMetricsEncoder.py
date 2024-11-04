import logging
from typing import Tuple, List
import torch
from chinese_checkers.game import ChineseCheckersGame, Player, Position
from chinese_checkers.geometry.Vector import Vector  # Assuming Vector is defined as in previous examples

class SpatialBoardMetricsEncoder:
    """
    Encodes spatial metrics of the board into a tensor with the following normalized metrics:
    - Centroid distance from target for the current player
    - Average move size
    - Max move size
    - Farthest position from target centroid
    - Shortest position from target centroid
    - Number of completed positions
    The values are normalized by the board size (radius).
    """

    def __init__(self, board_size: int):
        self.board_size = board_size

    def encode(self, game: ChineseCheckersGame) -> torch.Tensor:
        current_player = game.get_current_player()

        centroid_distance = self._calculate_centroid_distance(current_player, self.board_size)
        avg_move_size, max_move_size = self._calculate_move_sizes(game, self.board_size)
        farthest_distance = self._calculate_farthest_position(current_player, self.board_size)
        shortest_distance = self._calculate_shortest_position(current_player, self.board_size)
        completed_positions = self._calculate_completed_positions(current_player)

        return torch.tensor([
            centroid_distance,
            avg_move_size,
            max_move_size,
            farthest_distance,
            shortest_distance,
            completed_positions
        ], dtype=torch.float32)

    @staticmethod
    def _calculate_centroid_distance(player: Player, board_size: int) -> float:
        current_centroid = SpatialBoardMetricsEncoder._calculate_centroid(player.positions)
        target_centroid = SpatialBoardMetricsEncoder._calculate_centroid(player.target_positions)
        distance = current_centroid.distance(target_centroid) / board_size
        return distance

    @staticmethod
    def _calculate_move_sizes(game: ChineseCheckersGame, board_size: int) -> Tuple[float, float]:
        moves = game.get_next_moves()
        move_distances = [move.apply().distance(move.position) / board_size for move in moves]
        avg_move_size = sum(move_distances) / len(move_distances)
        max_move_size = max(move_distances)
        return avg_move_size, max_move_size

    @staticmethod
    def _calculate_farthest_position(player: Player, board_size: int) -> float:
        target_centroid = SpatialBoardMetricsEncoder._calculate_centroid(player.target_positions)
        farthest_distance = max(position.distance(target_centroid) for position in player.positions)
        return farthest_distance / board_size

    @staticmethod
    def _calculate_shortest_position(player: Player, board_size: int) -> float:
        target_centroid = SpatialBoardMetricsEncoder._calculate_centroid(player.target_positions)
        shortest_distance = min(position.distance(target_centroid) for position in player.positions)
        return shortest_distance / board_size

    @staticmethod
    def _calculate_completed_positions(player: Player) -> int:
        current_positions = set(player.positions)
        target_positions = set(player.target_positions)
        intersection = current_positions & target_positions
        return len(intersection)

    @staticmethod
    def _calculate_centroid(positions: List[Position]) -> Vector:
        if not positions:
            return Vector(0.0, 0.0)
        avg_i = sum(pos.i for pos in positions) / len(positions)
        avg_j = sum(pos.j for pos in positions) / len(positions)
        return Vector(avg_i, avg_j)
