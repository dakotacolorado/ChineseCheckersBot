from typing import Tuple
import torch
from chinese_checkers.encoder.interfaces.IMoveEncoder import IMoveEncoder
from chinese_checkers.game.Move import Move


class BasicMoveEncoder(IMoveEncoder):
    """
    Encodes moves in Chinese Checkers, converting them into start and end coordinates as a tensor.
    Also provides a dummy action for padding purposes when fewer moves are available.
    """

    MOVE_DIM: int = 4  # Dimension of the move encoding [start_x, start_y, end_x, end_y]

    @property
    def output_shape(self) -> Tuple[int, ...]:
        """
        Returns the shape of the output tensor from the encoding.

        Returns:
            Tuple[int, ...]: Shape of the encoded torch.Tensor.
        """
        return (self.MOVE_DIM,)

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

    def create_dummy_action(self) -> torch.Tensor:
        """
        Creates a dummy action encoding to use as padding when fewer moves are available.

        Returns:
            torch.Tensor: A tensor [0, 0, 0, 0] representing a no-op or dummy move.
        """
        return torch.zeros(self.output_shape, dtype=torch.int32)
