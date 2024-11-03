import logging
from typing import Tuple, List
import torch
from .IChineseCheckersGameEncoder import IChineseCheckersGameEncoder
from ...game import ChineseCheckersGame, Player, Position


class SpatialBoardMetricsEncoder(IChineseCheckersGameEncoder):
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

    @property
    def input_type(self) -> type:
        return ChineseCheckersGame

    @property
    def shape(self) -> Tuple[int, ...]:
        return (6,)

    def encode(self, game: ChineseCheckersGame) -> torch.Tensor:
        current_player = game.get_current_player()
        logging.debug(f"Current player positions: {current_player.positions}")
        logging.debug(f"Target positions: {current_player.target_positions}")

        # Calculate each metric
        centroid_distance = self._calculate_centroid_distance(current_player, self.board_size)
        logging.debug(f"Centroid distance: {centroid_distance}")

        avg_move_size, max_move_size = self._calculate_move_sizes(game, self.board_size)
        logging.debug(f"Average move size: {avg_move_size}, Max move size: {max_move_size}")

        farthest_distance = self._calculate_farthest_position(current_player, self.board_size)
        logging.debug(f"Farthest position distance: {farthest_distance}")

        shortest_distance = self._calculate_shortest_position(current_player, self.board_size)
        logging.debug(f"Shortest position distance: {shortest_distance}")

        completed_positions = self._calculate_completed_positions(current_player)
        logging.debug(f"Completed positions count: {completed_positions}")

        # Compile metrics into a tensor
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
        logging.debug(
            f"Centroid calculation - Current centroid: {current_centroid}, Target centroid: {target_centroid}, Distance: {distance}")
        return distance

    @staticmethod
    def _calculate_move_sizes(game: ChineseCheckersGame, board_size: int) -> Tuple[float, float]:
        moves = game.get_next_moves()
        move_distances = [move.apply().distance(move.position) / board_size for move in moves]
        avg_move_size = sum(move_distances) / len(move_distances)
        max_move_size = max(move_distances)
        logging.debug(f"Move sizes - Distances: {move_distances}, Average: {avg_move_size}, Max: {max_move_size}")
        return avg_move_size, max_move_size

    @staticmethod
    def _calculate_farthest_position(player: Player, board_size: int) -> float:
        target_centroid = SpatialBoardMetricsEncoder._calculate_centroid(player.target_positions)
        farthest_distance = max(position.distance(target_centroid) for position in player.positions)
        logging.debug(f"Farthest position - Target centroid: {target_centroid}, Farthest distance: {farthest_distance}")
        return farthest_distance / board_size

    @staticmethod
    def _calculate_shortest_position(player: Player, board_size: int) -> float:
        target_centroid = SpatialBoardMetricsEncoder._calculate_centroid(player.target_positions)
        shortest_distance = min(position.distance(target_centroid) for position in player.positions)
        logging.debug(f"Shortest position - Target centroid: {target_centroid}, Shortest distance: {shortest_distance}")
        return shortest_distance / board_size

    @staticmethod
    def _calculate_completed_positions(player: Player) -> int:
        current_positions = set(player.positions)
        target_positions = set(player.target_positions)
        intersection = current_positions & target_positions
        logging.debug(
            f"Completed positions - Current positions: {current_positions}, Target positions: {target_positions}, Intersection: {intersection}")
        return len(intersection)

    @staticmethod
    def _calculate_centroid(positions: List[Position]) -> Position:
        if not positions:
            return Position(0, 0)
        avg_i = sum(pos.i for pos in positions) / len(positions)
        avg_j = sum(pos.j for pos in positions) / len(positions)
        centroid = Position(int(avg_i), int(avg_j))
        logging.debug(f"Centroid - Positions: {positions}, Centroid: {centroid}")
        return centroid
