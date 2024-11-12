from typing import Tuple

import numpy as np
import torch
from ...game.ChineseCheckersGame import ChineseCheckersGame

class CnnEncoderState:
    def __init__(self, board_size: int):
        self.board_size = board_size
        self.board_dim = 2 * board_size + 1

    def shape(self) -> Tuple[int, int, int]:
        return 3, self.board_dim * 2, self.board_dim * 2

    def encode(self, game: ChineseCheckersGame) -> torch.Tensor:
        """
        Encodes the board state into a tensor with channels for:
            - Current player's positions
            - Current player's target positions
            - Other players' positions

        Args:
            game (ChineseCheckersGame): The current game state.

        Returns:
            torch.Tensor: A (3, board_dim * 2, board_dim * 2) tensor.
        """
        # Initialize the board tensor with the new dimensions
        board_tensor = np.zeros((3, self.board_dim * 2, self.board_dim * 2), dtype=np.float32)

        # Channel 0: Current player’s positions
        current_player = game.get_current_player()
        for position in current_player.positions:
            x, y = position.i + self.board_size, position.j + self.board_size
            board_tensor[0, x, y] = 1

        # Channel 1: Current player’s target positions
        for position in current_player.target_positions:
            x, y = position.i + self.board_size, position.j + self.board_size
            board_tensor[1, x, y] = 1

        # Channel 2: Other players' positions
        for player in game.get_other_players():
            for position in player.positions:
                x, y = position.i + self.board_size, position.j + self.board_size
                board_tensor[2, x, y] = 1

        return torch.tensor(board_tensor)