from typing import Tuple
import torch
import numpy as np
from .IChineseCheckersGameEncoder import IChineseCheckersGameEncoder
from ..ITensorEncoder import ITensorEncoder
from ...game.ChineseCheckersGame import ChineseCheckersGame


class SpatialBoardStateEncoder(IChineseCheckersGameEncoder[ChineseCheckersGame, torch.Tensor], ITensorEncoder[ChineseCheckersGame]):
    def __init__(self, board_size: int):
        """
        Initializes the encoder with the given board size (radius).

        Args:
            board_size (int): The radius of the board, used to calculate dimensions.
        """
        self.board_size = board_size
        self.board_dim = 2 * board_size + 1  # Calculate board dimension

    @property
    def shape(self) -> Tuple[int, ...]:
        """
        Returns the shape of the encoded board state: (3, board_dim, board_dim).
        """
        return 3, self.board_dim, self.board_dim

    def encode(self, game: ChineseCheckersGame) -> torch.Tensor:
        """
        Encodes the board state into a 3-channel tensor, each channel representing:
            - Current player's positions
            - Current player's target positions
            - Other players' positions

        Args:
            game (ChineseCheckersGame): The current game state.

        Returns:
            torch.Tensor: A (3, board_dim, board_dim) tensor.
        """
        board_tensor = np.zeros((3, self.board_dim, self.board_dim), dtype=np.float32)

        # Encode current player's positions in channel 0
        current_player = game.get_current_player()
        for position in current_player.positions:
            x, y = position.i + self.board_size, position.j + self.board_size
            board_tensor[0, x, y] = 1

        # Encode current player's target positions in channel 1
        for position in current_player.target_positions:
            x, y = position.i + self.board_size, position.j + self.board_size
            board_tensor[1, x, y] = 1

        # Encode other players' positions in channel 2
        for player in game.get_other_players():
            for position in player.positions:
                x, y = position.i + self.board_size, position.j + self.board_size
                board_tensor[2, x, y] = 1

        return torch.tensor(board_tensor)
