import warnings
from typing import Tuple

import torch

from ..ITensorEncoder import ITensorEncoder
from ..move.IMoveEncoder import IMoveEncoder
from .BaseChineseCheckersGameEncoder import BaseChineseCheckersGameEncoder
from ...game import ChineseCheckersGame


class GridPositionTargetEncoder(BaseChineseCheckersGameEncoder):
    def __init__(
            self,
            move_encoder: IMoveEncoder,
            max_moves: int,
            board_size: int,
    ):
        warnings.warn(
            f"{self.__class__.__name__} is deprecated and will be removed in a future version. "
            "Consider using SpatialBoardStateEncoder instead.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(move_encoder, max_moves)
        self.board_size = board_size
        self.board_dim = self._calculate_board_dim(board_size)

    @staticmethod
    def _calculate_board_dim(radius: int) -> int:
        return 2 * radius * 2 + 1

    def encode_positions(self, game: ChineseCheckersGame) -> torch.Tensor:
        """
        Encodes the positions of the current player and opponents on the board,
        preserving spatial structure. Combines both the current player's and opponents'
        positions into a single tensor for spatial relationship processing.

        Args:
            game (ChineseCheckersGame): The game state.

        Returns:
            torch.Tensor: A tensor with shape (2, board_dim, board_dim), where:
                          - The first channel encodes the current player's positions.
                          - The second channel encodes the opponents' positions.
        """
        current_player_matrix = self._encode_current_player(game, self.board_dim)
        opponent_matrix = self._encode_opponents(game, self.board_dim)

        # Combine current player and opponent matrices into a single tensor with two channels
        return torch.stack([current_player_matrix, opponent_matrix], dim=0)

    @property
    def positions_shape(self) -> Tuple[int, int, int]:
        """
        Returns the shape of the positional encoding output, which includes two channels
        (current player and opponents) and the board dimensions.

        Returns:
            Tuple[int, int, int]: Shape of the encoded positions tensor as (2, board_dim, board_dim).
        """
        return 2, self.board_dim, self.board_dim

    @staticmethod
    def _encode_current_player(game: ChineseCheckersGame, board_dim: int) -> torch.Tensor:
        """
        Encodes the current player's positions on the board.

        Args:
            game (ChineseCheckersGame): The game state.
            board_dim (int): The board dimension.

        Returns:
            torch.Tensor: A tensor of shape (board_dim, board_dim) with the current player's positions.
        """
        current_player_matrix = torch.zeros((board_dim, board_dim), dtype=torch.int32)
        current_player = game.get_current_player()

        for position in current_player.positions:
            x, y = position.i + board_dim // 2, position.j + board_dim // 2
            current_player_matrix[x, y] = 1

        return current_player_matrix

    @staticmethod
    def _encode_opponents(game: ChineseCheckersGame, board_dim: int) -> torch.Tensor:
        """
        Encodes all opponents' positions on the board.

        Args:
            game (ChineseCheckersGame): The game state.
            board_dim (int): The board dimension.

        Returns:
            torch.Tensor: A tensor of shape (board_dim, board_dim) with the opponents' positions.
        """
        opponent_matrix = torch.zeros((board_dim, board_dim), dtype=torch.int32)
        current_player = game.get_current_player()

        for player in game.players:
            if player != current_player:
                for position in player.positions:
                    x, y = position.i + board_dim // 2, position.j + board_dim // 2
                    opponent_matrix[x, y] = 1

        return opponent_matrix
