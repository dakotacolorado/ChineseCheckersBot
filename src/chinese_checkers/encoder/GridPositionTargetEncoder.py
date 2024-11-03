import numpy as np
from ..game.ChineseCheckersGame import ChineseCheckersGame
from .IChineseCheckersGameEncoder import IChineseCheckersGameEncoder
from .MoveEncoder import MoveEncoder


class GridPositionTargetEncoder(IChineseCheckersGameEncoder):
    """
    Encodes the game state of Chinese Checkers into a single array:
    combining the current player, opponents, a one-hot encoded vector for player ID,
    and encoded moves for the current state.
    """

    def __init__(self, max_moves: int = 100):
        """
        Initializes the encoder with a maximum number of moves to encode.

        Args:
            max_moves (int): Maximum number of moves to encode in the state representation.
        """
        self.move_encoder = MoveEncoder()
        self.max_moves = max_moves
        self.dummy_action_encoding = self.move_encoder.create_dummy_action()

    def encode(self, game: ChineseCheckersGame) -> np.ndarray:
        """
        Encodes a `ChineseCheckersGame` state into a single array by combining:
            - Flattened current player's positions matrix (1D array of length board_dim^2)
            - Flattened opponents' positions matrix (1D array of length board_dim^2)
            - One-hot encoded player ID vector (1D array of length 6)
            - Encoded moves for current state (2D array of shape (max_moves, move_encoding_size))

        Args:
            game (ChineseCheckersGame): The game state to encode.

        Returns:
            np.ndarray: Encoded game state as a single flattened array.
        """
        # Calculate board dimension
        board_dim = self._calculate_board_dim(game.board)

        # Encode each part of the game state
        current_player_matrix = self._encode_current_player(game, board_dim)
        opponent_matrix = self._encode_opponents(game, board_dim)
        player_id_one_hot = self._encode_player_id(game)
        moves_encoding = self._encode_moves(game)

        # Concatenate all parts into a single array
        encoded_state = np.concatenate([
            current_player_matrix.flatten(),
            opponent_matrix.flatten(),
            player_id_one_hot,
            moves_encoding.flatten()
        ])

        return encoded_state

    @staticmethod
    def _calculate_board_dim(hexagram) -> int:
        """
        Calculates the board dimension based on the range of coordinates in hexagram points.

        Args:
            hexagram: The hexagram board structure.

        Returns:
            int: Dimension of the square board.
        """
        min_i = min(point.i for point in hexagram.hexagram_points)
        max_i = max(point.i for point in hexagram.hexagram_points)
        min_j = min(point.j for point in hexagram.hexagram_points)
        max_j = max(point.j for point in hexagram.hexagram_points)

        return max(max_i - min_i, max_j - min_j) + 1  # Adjust for zero indexing

    @staticmethod
    def _encode_current_player(game: ChineseCheckersGame, board_dim: int) -> np.ndarray:
        """
        Encodes the current player's positions as a 2D board matrix.

        Args:
            game (ChineseCheckersGame): The game state.
            board_dim (int): The board dimension.

        Returns:
            np.ndarray: Flattened array representing the current player's positions on the board.
        """
        current_player_matrix = np.zeros((board_dim, board_dim), dtype=int)
        current_player = game.get_current_player()

        for position in current_player.positions:
            x, y = position.i + board_dim // 2, position.j + board_dim // 2
            current_player_matrix[x, y] = 1

        return current_player_matrix

    @staticmethod
    def _encode_opponents(game: ChineseCheckersGame, board_dim: int) -> np.ndarray:
        """
        Encodes all opponents' positions as a 2D board matrix.

        Args:
            game (ChineseCheckersGame): The game state.
            board_dim (int): The board dimension.

        Returns:
            np.ndarray: Flattened array representing opponents' positions on the board.
        """
        opponent_matrix = np.zeros((board_dim, board_dim), dtype=int)
        current_player = game.get_current_player()

        for player in game.players:
            if player != current_player:
                for position in player.positions:
                    x, y = position.i + board_dim // 2, position.j + board_dim // 2
                    opponent_matrix[x, y] = 1

        return opponent_matrix

    @staticmethod
    def _encode_player_id(game: ChineseCheckersGame) -> np.ndarray:
        """
        One-hot encodes the current player's ID.

        Args:
            game (ChineseCheckersGame): The game state.

        Returns:
            np.ndarray: One-hot encoded vector of length 6 for the current player's ID.
        """
        player_id_one_hot = np.zeros(6, dtype=int)
        current_player_id = int(game.get_current_player().player_id)
        player_id_one_hot[current_player_id] = 1

        return player_id_one_hot

    def _encode_moves(self, game: ChineseCheckersGame) -> np.ndarray:
        """
        Encodes a limited number of moves (top N moves) as flattened arrays.
        Pads with dummy actions if fewer than max_moves are available.

        Args:
            game (ChineseCheckersGame): The game state.

        Returns:
            np.ndarray: 2D array of shape (max_moves, move_encoding_size) representing encoded moves.
        """
        moves = game.get_next_moves()
        # TODO: Do better than random selection of the top N moves
        # This works for now, but should be optimized.
        encoded_moves = [self.move_encoder.encode(move) for move in moves[:self.max_moves]]

        # Pad with dummy actions if fewer moves than max_moves
        while len(encoded_moves) < self.max_moves:
            encoded_moves.append(self.dummy_action_encoding)

        return np.array(encoded_moves)
