import numpy as np

from ..game.Move import Move


class MoveEncoder:
    """
    Encodes moves in Chinese Checkers, converting them into start and end coordinates as a vector.
    Also provides a dummy action for padding purposes when fewer moves are available.
    """

    MOVE_DIM: int = 4

    @staticmethod
    def encode(move: Move) -> np.ndarray:
        """
        Encodes a single move as a numpy array with start and end positions.

        Args:
            move (Move): The move to encode.

        Returns:
            np.ndarray: An array [start_x, start_y, end_x, end_y] representing the move.
        """
        start_x, start_y = move.i, move.j
        end_x, end_y = move.apply().i, move.apply().j
        return np.array([start_x, start_y, end_x, end_y])

    @staticmethod
    def create_dummy_action() -> np.ndarray:
        """
        Creates a dummy action encoding to use when fewer than the maximum moves are available.
        This is used as padding in the action set.

        Returns:
            np.ndarray: An array [0, 0, 0, 0] representing a no-op or dummy move.
        """
        return np.zeros(MoveEncoder.MOVE_DIM, dtype=int)
