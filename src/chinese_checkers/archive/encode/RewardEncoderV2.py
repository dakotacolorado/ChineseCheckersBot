
from typing import List
import torch

from chinese_checkers.game import ChineseCheckersGame, Move, Player
from chinese_checkers.geometry import Centroid
from chinese_checkers.simulation import GameSimulation


class RewardEncoderV2:

    def encode(self, simulation: GameSimulation) -> List[torch.Tensor]:
        moves = simulation.data.historical_moves
        game_sequence: List[ChineseCheckersGame] = simulation._to_game_sequence()
        game_length = len(game_sequence)
        winner_id = simulation.metadata.winning_player

        q_objective_list = []
        for turn, (move, game) in enumerate(zip(moves, game_sequence)):
            next_game = game.apply_move(move)
            q_objective = self._compose_features(game, move, next_game, turn, game_length, winner_id)
            q_objective_tensor = torch.tensor([q_objective], dtype=torch.float32)
            q_objective_list.append(q_objective_tensor)
        return q_objective_list

    @staticmethod
    def _compose_features(
            game: ChineseCheckersGame,
            move: Move,
            next_game: ChineseCheckersGame,
            turn: int,
            game_length: int,
            winner_id: str,
    ) -> float:
        return sum([
            RewardEncoderV2._player_distance_from_target(next_game)/4,
            RewardEncoderV2._distance_from_win_loss(game, turn, game_length, winner_id)
        ])

    # normalize the distance of group to target between 0 and 1
    # 1 is the closest to the target. 0 is the farthest
    MAX_INVERSE_DISTANCE = 40.0
    MIN_INVERSE_DISTANCE = 0.47140452079103173
    SHIFTED_MAX_INVERSE_DISTANCE = MAX_INVERSE_DISTANCE - MIN_INVERSE_DISTANCE
    @staticmethod
    def _player_distance_from_target(game: ChineseCheckersGame) -> float:
        board_size: int = game.board.radius
        current_player = game.get_current_player()
        target_positions = current_player.target_positions
        current_positions = current_player.positions
        current_centroid = Centroid.from_vectors(current_positions)
        target_centroid = Centroid.from_vectors(target_positions)
        distance = current_centroid.distance(target_centroid) / board_size
        if distance == 0:
            return 1
        else:
            return ( (1 / distance) - RewardEncoderV2.MIN_INVERSE_DISTANCE ) / RewardEncoderV2.SHIFTED_MAX_INVERSE_DISTANCE

    @staticmethod
    def _distance_from_win_loss(game: ChineseCheckersGame, turn: int, game_length: int, winner_id: str) -> float:
        current_player = game.get_current_player()
        if current_player.player_id == winner_id:
            return (turn + 1) / game_length
        else:
            return -1 * (turn + 1) / game_length