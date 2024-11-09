import torch
from ...game import Move, ChineseCheckersGame


class SpatialMoveEncoder:
    """
    Encodes moves in Chinese Checkers, converting them into start and end coordinates as a tensor.
    Also provides a dummy action for padding purposes when fewer moves are available.
    """

    def encode(self, move: Move) -> torch.Tensor:
        """
        Encodes a single move as a torch.Tensor with start and end positions.

        Args:
            move (Move): The move to encode.

        Returns:
            torch.Tensor: A tensor [start_x, start_y, end_x, end_y] representing the move.
        """
        start_x, start_y = move.i, move.j
        end_x, end_y = move.apply().i, move.apply().j
        return torch.tensor([start_x, start_y, end_x, end_y], dtype=torch.int32)

