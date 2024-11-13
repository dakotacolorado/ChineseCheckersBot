
from typing import List
import torch

from chinese_checkers.game import ChineseCheckersGame, Move, Player
from chinese_checkers.geometry import Centroid
from chinese_checkers.simulation import GameSimulation


class RewardEncoder:

    def encode(self, simulation: GameSimulation) -> List[torch.Tensor]:
        moves = simulation.data.historical_moves
        game_sequence: List[ChineseCheckersGame] = simulation.to_game_sequence()
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
            RewardEncoder._distance_from_win_loss(game, turn, game_length, winner_id),
            RewardEncoder._player_distance_from_target(next_game) / 4 * (turn / game_length),
            RewardEncoder._player_positions_in_target(next_game) / 4 * (turn / game_length),
            RewardEncoder._player_positions_not_in_start(next_game) / 4 * ( turn / game_length )
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
            return ( (1 / distance) - RewardEncoder.MIN_INVERSE_DISTANCE ) / RewardEncoder.SHIFTED_MAX_INVERSE_DISTANCE

    @staticmethod
    def _distance_from_win_loss(game: ChineseCheckersGame, turn: int, game_length: int, winner_id: str) -> float:
        current_player = game.get_current_player()
        if current_player.player_id == winner_id:
            return (turn + 1) / game_length
        else:
            return -1 * (turn + 1) / game_length

    @staticmethod
    def _player_positions_in_target(game: ChineseCheckersGame) -> float:
        # set the turn back 1 so the current player is the player that just moved
        game = ChineseCheckersGame(
            players=game.players,
            turn=game.turn - 1,
            board=game.board,
            printer=game.printer
        )
        current_player = game.get_current_player()
        target_positions = current_player.target_positions
        current_positions = current_player.positions
        score = 0
        for pos in current_positions:
            if pos in target_positions:
                score += 1
        return score / len(current_positions)

    @staticmethod
    def _player_positions_not_in_start(game: ChineseCheckersGame) -> float:
        # set the turn back 1 so the current player is the player that just moved
        game = ChineseCheckersGame(
            players=game.players,
            turn=game.turn - 1,
            board=game.board,
            printer=game.printer
        )
        start_game = ChineseCheckersGame.start_game(number_of_players=len(game.players), board_size=game.board.radius)
        start_positions = [p for p in start_game.players if p.player_id == game.get_current_player().player_id][0].positions
        current_player = game.get_current_player()
        current_positions = current_player.positions
        score = 0
        for pos in start_positions:
            if pos in current_positions:
                score -= 1
        return score / len(current_positions)