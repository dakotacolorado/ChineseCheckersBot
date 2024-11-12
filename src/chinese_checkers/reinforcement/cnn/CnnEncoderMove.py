from typing import Tuple

import numpy as np
import torch

from chinese_checkers.game import Move


class CnnEncoderMove:
    def __init__(self, board_size: int):
        self.board_size = board_size
        self.board_dim = 2 * (board_size + 1)

    def shape(self) -> Tuple[int, int, int]:
        return 2, self.board_dim * 2, self.board_dim * 2

    def encode(self, move: Move) -> torch.Tensor:
        """
        Encodes a move into a 2-channel tensor representing the start and end positions.

        Args:
            move (Move): The move to encode.

        Returns:
            torch.Tensor: A (2, board_dim, board_dim) tensor with start and end positions.
        """
        move_tensor = np.zeros((2, self.board_dim * 2 , self.board_dim * 2), dtype=np.float32)

        # Channel 0: Start position
        start_x, start_y = move.i + self.board_size, move.j + self.board_size
        move_tensor[0, start_x, start_y] = 1

        # Channel 1: End position
        end_position = move.apply()
        end_x, end_y = end_position.i + self.board_size, end_position.j + self.board_size
        move_tensor[1, end_x, end_y] = 1

        return torch.tensor(move_tensor)