from abc import ABC, abstractmethod
import torch
from typing import Tuple
from ..move import IMoveEncoder
from .IChineseCheckersGameEncoder import IChineseCheckersGameEncoder
from ...game import ChineseCheckersGame


class BaseChineseCheckersGameEncoder(IChineseCheckersGameEncoder, ABC):
    # Shape for one-hot encoded player ID vector
    PLAYER_ID_SHAPE: Tuple[int] = (6,)

    def __init__(self, move_encoder: IMoveEncoder, max_moves: int):
        self.move_encoder = move_encoder
        self.max_moves = max_moves
        self.dummy_action_encoding = self.move_encoder.create_dummy_action()
        self.move_encoding_shape = (self.max_moves, *self.move_encoder.shape)

    @staticmethod
    def _encode_player_id(game: ChineseCheckersGame) -> torch.Tensor:
        """
        One-hot encodes the current player's ID. This helps the model understand
        which player is making the moves, enabling it to distinguish between players
        based on ID.

        Args:
            game (ChineseCheckersGame): The game state.

        Returns:
            torch.Tensor: A one-hot encoded tensor for the player ID with shape (6,).
        """
        player_id_one_hot = torch.zeros(BaseChineseCheckersGameEncoder.PLAYER_ID_SHAPE, dtype=torch.int32)
        current_player_id = int(game.get_current_player().player_id)
        player_id_one_hot[current_player_id] = 1
        return player_id_one_hot

    def _encode_moves(self, game: ChineseCheckersGame) -> torch.Tensor:
        """
        Encodes a limited number of moves (up to max_moves) as a tensor. This helps
        the model recognize available moves and their sequence in the game,
        preserving their spatial and temporal structure.

        Args:
            game (ChineseCheckersGame): The game state.

        Returns:
            torch.Tensor: A 2D tensor representing encoded moves, with shape
            (max_moves, move_encoder.shape).
        """
        moves = game.get_next_moves()
        encoded_moves = [self.move_encoder.encode(move) for move in moves[:self.max_moves]]
        while len(encoded_moves) < self.max_moves:
            encoded_moves.append(self.dummy_action_encoding)
        return torch.stack(encoded_moves)

    @abstractmethod
    def encode_positions(self, game: ChineseCheckersGame) -> torch.Tensor:
        """
        Encodes the positions of players on the board. Each subclass can implement
        specific positional encoding strategies, preserving spatial relationships.

        Args:
            game (ChineseCheckersGame): The game state.

        Returns:
            torch.Tensor: A tensor representing the encoded positions on the board.
        """
        pass

    @property
    @abstractmethod
    def positions_shape(self) -> Tuple[int, ...]:
        """
        Defines the shape of the positional encoding output.

        Returns:
            Tuple[int, ...]: Shape of the encoded positions.
        """
        pass

    @property
    def shape(self) -> Tuple[int, ...]:
        """
        Expected shape of the combined encoding:

        - (6,) for the player ID encoding (one-hot vector for 6 players).
        - (max_moves, move_encoder.shape) for the encoded moves. This preserves
          the structure of each move in the sequence, facilitating temporal
          processing in the model.
        - positions_shape for the encoded board positions, retaining the spatial
          configuration of players on the board.

        Expected final shape: (6, max_moves, move_encoder.shape[1], ..., positions_shape)

        Returns:
            Tuple[int, ...]: Combined shape for the full encoding.
        """
        return (
            self.PLAYER_ID_SHAPE[0],
            *self.move_encoding_shape,
            *self.positions_shape,
        )

    def encode(self, game: ChineseCheckersGame) -> torch.Tensor:
        """
        Encodes the game state into a structured tensor containing player ID, moves,
        and positional information. The structured format preserves spatial and temporal
        relationships, suitable for processing by CNNs.

        - Player ID encoding provides a one-hot vector, helping the model distinguish
          the current player.
        - Move encoding keeps the sequential nature of moves, allowing the model to
          understand move order and availability.
        - Position encoding preserves board layout, enabling spatial processing
          through convolutions.

        Args:
            game (ChineseCheckersGame): The game state to encode.

        Returns:
            torch.Tensor: A structured tensor containing the encoded game state.
        """
        player_id_encoding = self._encode_player_id(game)
        moves_encoding = self._encode_moves(game)
        positions_encoding = self.encode_positions(game)

        return torch.cat([
            player_id_encoding.view(1, -1),  # Reshape to maintain structure
            moves_encoding.view(1, *self.move_encoding_shape),
            positions_encoding.view(1, *self.positions_shape)
        ], dim=0)
