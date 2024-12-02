import time
from typing import List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import torch
from tqdm import tqdm

from chinese_checkers.simulation import GameSimulation
from .Experience import Experience
from chinese_checkers.game import ChineseCheckersGame, Move
from chinese_checkers.geometry import Centroid


def encode_game(game: ChineseCheckersGame) -> torch.Tensor:
    board_dim = 2 * (game.board.radius + 1)
    board_tensor = torch.zeros((3, board_dim * 2, board_dim * 2), dtype=torch.int8)

    current_player = game.get_current_player()
    for position in current_player.positions:
        x, y = position.i + board_dim, position.j + board_dim
        board_tensor[0, x, y] = 1

    for position in current_player.target_positions:
        x, y = position.i + board_dim, position.j + board_dim
        board_tensor[1, x, y] = 1

    for player in game.get_other_players():
        for position in player.positions:
            x, y = position.i + board_dim, position.j + board_dim
            board_tensor[2, x, y] = 1

    return board_tensor


def encode_game_from_experience(experience: Experience) -> torch.Tensor:
    return encode_game(experience.game)


def encode_move(move: Move, board_size: int) -> torch.Tensor:
    board_dim = 2 * (board_size + 1)
    move_tensor = torch.zeros((2, board_dim * 2, board_dim * 2), dtype=torch.float32)

    start_x, start_y = move.position.i + board_dim, move.position.j + board_dim
    move_tensor[0, start_x, start_y] = 1

    end_position = move.apply()
    end_x, end_y = end_position.i + board_dim, end_position.j + board_dim
    move_tensor[1, end_x, end_y] = 1

    return move_tensor


def encode_move_from_experience(experience: Experience) -> torch.Tensor:
    move = experience.move
    board_size = experience.board_size
    return encode_move(move, board_size)


def encode_reward(game: ChineseCheckersGame, move: Move, winner_id: str, total_turns: int) -> torch.Tensor:
    reward = 0.0
    reward += _distance_from_win_loss(game, game.turn, total_turns, winner_id)
    reward += _player_distance_from_target(game, move)
    reward += _player_positions_in_target(game, move)
    reward += _player_positions_not_in_start(game, move)

    return torch.tensor(reward, dtype=torch.float32)


def encode_reward_from_experience(experience: Experience) -> torch.Tensor:
    game = experience.game
    move = experience.move
    winner_id = experience.winning_player
    total_turns = experience.total_turns
    return encode_reward(game, move, winner_id, total_turns)


def _distance_from_win_loss(game: ChineseCheckersGame, turn: int, game_length: int, winner_id: str) -> float:
    current_player = game.get_current_player()
    return (turn + 1) / game_length if current_player.player_id == winner_id else -0.25 * (turn + 1) / game_length


def _player_distance_from_target(game: ChineseCheckersGame, move: Move) -> float:
    board_size: int = game.board.radius
    current_player = game.get_current_player()
    target_positions = current_player.target_positions
    new_positions = current_player.apply_move(move).positions
    current_centroid = Centroid.from_vectors(new_positions)
    target_centroid = Centroid.from_vectors(target_positions)
    distance = current_centroid.distance(target_centroid) / board_size
    if distance == 0:
        return 1.0
    else:
        # Normalize inverse distance
        normalized_distance = (1 / distance - 0.4714) / (40.0 - 0.4714)
        return normalized_distance


def _player_positions_in_target(game: ChineseCheckersGame, move: Move) -> float:
    current_player = game.get_current_player()
    target_positions = set(current_player.target_positions)
    new_positions = current_player.apply_move(move).positions
    return sum(1 for pos in new_positions if pos in target_positions) / len(new_positions)


def _player_positions_not_in_start(game: ChineseCheckersGame, move: Move) -> float:
    current_player = game.get_current_player()
    start_positions = set(current_player.positions)
    new_positions = current_player.apply_move(move).positions
    return sum(1 for pos in start_positions if pos not in new_positions) / len(new_positions)


class Encoder:
    def __init__(self, board_size: int, thread_count: int = 4):
        self.board_size = board_size
        self.thread_count = thread_count

    def batch_encode(self, experiences: List[Experience]) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        total_experiences = len(experiences)
        print(f"Starting encoding of {total_experiences} experiences using {self.thread_count} workers.")

        start_time = time.time()

        with ProcessPoolExecutor(max_workers=self.thread_count) as executor:
            # Submit all tasks to the executor
            state_futures = {executor.submit(encode_game_from_experience, exp): exp for exp in experiences}
            move_futures = {executor.submit(encode_move_from_experience, exp): exp for exp in experiences}
            reward_futures = {executor.submit(encode_reward_from_experience, exp): exp for exp in experiences}

            # Track progress with a loading bar
            for future in tqdm(
                    as_completed(list(state_futures.keys()) + list(move_futures.keys()) + list(reward_futures.keys())),
                    total=3 * total_experiences, desc="Encoding experiences", unit="tasks"):
                pass  # Future completion updates the progress bar

            # Collect results
            encoded_states = [future.result() for future in state_futures]
            encoded_moves = [future.result() for future in move_futures]
            encoded_rewards = [future.result() for future in reward_futures]

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Encoding complete. Encoded {total_experiences} experiences in {elapsed_time:.2f} seconds.")

        return encoded_states, encoded_moves, encoded_rewards

    def encode_game(self, game: ChineseCheckersGame) -> torch.Tensor:
        if game.board.radius != self.board_size:
            raise ValueError(f"Board size mismatch. Expected {self.board_size}, got {game.board.radius}.")
        game = encode_game(game)
        # remove this error after unit testing
        if game.shape != self.encoded_game_shape:
            raise ValueError(f"Game shape mismatch. Expected {self.encoded_game_shape}, got {game.shape}.")
        return game

    @property
    def encoded_game_shape(self) -> Tuple[int, int, int]:
        board_dim = 2 * (self.board_size + 1)
        return 3, board_dim * 2, board_dim * 2

    def encode_move(self, move: Move) -> torch.Tensor:
        move = encode_move(move, self.board_size)
        # remove this error after unit testing
        if move.shape != self.encoded_move_shape:
            raise ValueError(f"Move shape mismatch. Expected {self.encoded_move_shape}, got {move.shape}.")
        return move

    @property
    def encoded_move_shape(self) -> Tuple[int, int, int]:
        board_dim = 2 * (self.board_size + 1)
        return 2, board_dim * 2, board_dim * 2

    def encode_reward(self, simulation: GameSimulation) -> List[torch.Tensor]:
        if simulation.metadata.board_size != self.board_size:
            raise ValueError(f"Board size mismatch. Expected {self.board_size}, got {simulation.metadata.board_size}.")
        games = simulation.to_game_sequence()
        moves = simulation.data.historical_moves
        winner = simulation.metadata.winning_player
        total_turns = len(moves)
        return [
            encode_reward(game, move, winner, total_turns)
            for game, move in zip(games, moves)
        ]

    @property
    def encoded_reward_shape(self) -> Tuple[int]:
        return 1,