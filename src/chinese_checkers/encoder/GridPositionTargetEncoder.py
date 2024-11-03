import numpy as np
from typing import Tuple
from ..game.ChineseCheckersGame import ChineseCheckersGame
from .IChineseCheckersGameEncoder import IChineseCheckersGameEncoder

class GridPositionTargetEncoder(IChineseCheckersGameEncoder):
    """
    Encodes the game state of Chinese Checkers into a single array:
    combining the current player, opponents, and a one-hot encoded vector for player ID.
    """

    def encode(self, game: ChineseCheckersGame) -> np.ndarray:
        """
        Encodes a single `ChineseCheckersGame` state into a single array by
        combining the current player's positions, all other players' positions,
        and a one-hot encoded player ID.

        Args:
            game (ChineseCheckersGame): The game state to encode.

        Returns:
            np.ndarray: A single array containing:
                - Flattened current player's positions.
                - Flattened opponents' positions.
                - One-hot encoded player ID.
        """
        # Determine board dimensions dynamically
        board_dim = self._calculate_board_dim(game.board)

        # Initialize matrices for current player and opponents
        current_player_matrix = np.zeros((board_dim, board_dim), dtype=int)
        opponent_matrix = np.zeros((board_dim, board_dim), dtype=int)

        # Get the current player and the ID as an integer
        current_player = game.get_current_player()
        current_player_id = int(current_player.player_id)

        # Fill current player positions
        for position in current_player.positions:
            x, y = position.i + board_dim // 2, position.j + board_dim // 2
            current_player_matrix[x, y] = 1

        # Fill opponent positions
        for player in game.players:
            if player != current_player:
                for position in player.positions:
                    x, y = position.i + board_dim // 2, position.j + board_dim // 2
                    opponent_matrix[x, y] = 1

        # Create a one-hot encoded vector for the player ID (up to 6 players)
        player_id_one_hot = np.zeros(6, dtype=int)
        player_id_one_hot[current_player_id] = 1

        # Concatenate the flattened matrices and the one-hot vector into a single array
        encoded_state = np.concatenate(
            [current_player_matrix.flatten(), opponent_matrix.flatten(), player_id_one_hot]
        )

        return encoded_state

    @staticmethod
    def _calculate_board_dim(hexagram) -> int:
        """
        Calculates the board dimension based on the range of coordinates in hexagram points.

        Args:
            hexagram: The hexagram board structure.

        Returns:
            int: Calculated dimension of the board.
        """
        min_i = min(point.i for point in hexagram.hexagram_points)
        max_i = max(point.i for point in hexagram.hexagram_points)
        min_j = min(point.j for point in hexagram.hexagram_points)
        max_j = max(point.j for point in hexagram.hexagram_points)

        return max(max_i - min_i, max_j - min_j) + 1  # Adjust for zero indexing
