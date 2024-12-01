from typing import List
import torch

from chinese_checkers.game import ChineseCheckersGame, Move, Player
from chinese_checkers.geometry import Centroid
from chinese_checkers.simulation import GameSimulation

"""
This only works for games that are completed.  DNF games should be encoded differently.
"""
class RewardEncoder:

    def encode(self, simulation: GameSimulation, game_sequence) -> List[torch.Tensor]:
        moves = simulation.data.historical_moves
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
        # Precompute invariant factors
        # weight_factor = 1 / 4 * (turn / game_length)
        return sum([
            RewardEncoder._player_distance_from_target(game, move),
            RewardEncoder._player_positions_in_target(game, move),
            RewardEncoder._player_positions_not_in_start(game, move)/8
        ])

    MAX_INVERSE_DISTANCE = 40.0
    MIN_INVERSE_DISTANCE = 0.47140452079103173
    SHIFTED_MAX_INVERSE_DISTANCE = MAX_INVERSE_DISTANCE - MIN_INVERSE_DISTANCE

    @staticmethod
    def _player_distance_from_target(game: ChineseCheckersGame, move: Move) -> float:
        board_size: int = game.board.radius
        current_player = game.get_current_player()
        target_positions = current_player.target_positions
        new_positions = current_player.apply_move(move).positions
        current_centroid = Centroid.from_vectors(new_positions)
        target_centroid = Centroid.from_vectors(target_positions)
        distance = current_centroid.distance(target_centroid) / board_size
        return 1 if distance == 0 else ((1 / distance) - RewardEncoder.MIN_INVERSE_DISTANCE) / RewardEncoder.SHIFTED_MAX_INVERSE_DISTANCE

    @staticmethod
    def _distance_from_win_loss(game: ChineseCheckersGame, turn: int, game_length: int, winner_id: str) -> float:
        current_player = game.get_current_player()
        return (turn + 1) / game_length if current_player.player_id == winner_id else -1 * (turn + 1) / game_length

    @staticmethod
    def _player_positions_in_target(game: ChineseCheckersGame, move: Move) -> float:
        # Set the turn back 1 so the current player is the player that just moved
        current_player = game.get_current_player()
        target_positions = set(current_player.target_positions)  # Convert to set for faster lookup
        new_positions = current_player.apply_move(move).positions

        # Count positions in target using a generator expression
        score = sum(1 for pos in new_positions if pos in target_positions)
        return score / len(new_positions)

    START_GAME: ChineseCheckersGame = ChineseCheckersGame.start_game(number_of_players=6, board_size=4)

    @staticmethod
    def _player_positions_not_in_start(game: ChineseCheckersGame, move: Move) -> float:
        current_player = game.get_current_player()
        start_player = next(p for p in RewardEncoder.START_GAME.players if p.player_id == current_player.player_id)
        start_positions = set(start_player.positions)  # Convert to set for faster lookup
        new_positions = current_player.apply_move(move).positions

        # Count positions not in start using a generator expression
        score = sum(1 for pos in start_positions if pos not in new_positions)
        return score / len(new_positions)