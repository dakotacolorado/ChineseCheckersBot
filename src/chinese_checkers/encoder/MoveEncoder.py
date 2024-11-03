import numpy as np

from ..game.Move import Move


class MoveEncoder:
    """
    Encodes moves in Chinese Checkers, converting them into start and end coordinates as a vector.
    """

    @staticmethod
    def encode(move: Move) -> np.ndarray:
        """
        Encodes a single move as a numpy array with start and end positions.

        Args:
            move (Move): The move to encode.

        Returns:
            np.ndarray: An array [start_x, start_y, end_x, end_y].
        """
        start_x, start_y = move.i, move.j
        end_x, end_y = move.apply().i, move.apply().j
        return np.array([start_x, start_y, end_x, end_y])