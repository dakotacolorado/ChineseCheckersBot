from typing import Tuple

import numpy as np
from ..game.ChineseCheckersGame import ChineseCheckersGame
from ..geometry.Hexagram import Hexagram
from .IChineseCheckersGameEncoder import IChineseCheckersGameEncoder

class GridPositionTargetEncoder(IChineseCheckersGameEncoder):
    """
    Encodes the game state of Chinese Checkers into a grid representation, including player positions
    and target indicators.
    """

    def encode(self, game: ChineseCheckersGame) -> Tuple[np.ndarray, int]:
        """
        Encodes a single `ChineseCheckersGame` state into a grid representation.

        Args:
            game (ChineseCheckersGame): The game state to encode.

        Returns:
            Tuple[np.ndarray, int]: A tuple containing:
                - np.ndarray: Encoded board state with player and target positions.
                - int: Current player index.
        """
        # Determine board dimensions based on hexagram points
        board_dim = self._calculate_board_dim(game.board)

        # Initialize an empty grid to represent the board state
        board_state = np.zeros((board_dim, board_dim), dtype=int)

        # Encode each player's positions on the board
        for player_index, player in enumerate(game.players, start=1):
            for position in player.positions:
                x, y = position.i + board_dim // 2, position.j + board_dim // 2
                board_state[x, y] = player_index  # Unique encoding for each player

        # Encode target positions with negative values for each player
        for player_index, player in enumerate(game.players, start=1):
            for target_position in player.target_positions:
                x, y = target_position.i + board_dim // 2, target_position.j + board_dim // 2
                board_state[x, y] = -player_index  # Use negative index for target

        # Encode the current player as an additional feature
        current_player_index = game.players.index(game.get_current_player()) + 1

        # Return the board state and current player index as a tuple
        return board_state, current_player_index

    @staticmethod
    def _calculate_board_dim(hexagram: Hexagram) -> int:
        """
        Calculates the board dimension based on the range of coordinates in hexagram points.

        Args:
            hexagram (Hexagram): The hexagram board structure.

        Returns:
            int: Calculated dimension of the board.
        """
        # Get the min and max values of i and j coordinates in hexagram_points
        min_i = min(point.i for point in hexagram.hexagram_points)
        max_i = max(point.i for point in hexagram.hexagram_points)
        min_j = min(point.j for point in hexagram.hexagram_points)
        max_j = max(point.j for point in hexagram.hexagram_points)

        # Calculate the dimension needed to fit all points
        board_dim = max(max_i - min_i, max_j - min_j) + 1  # Adding 1 to account for zero indexing

        return board_dim
