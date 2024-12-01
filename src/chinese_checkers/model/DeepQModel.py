import random
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
from typing import List
import numpy as np
import torch
import time
from torch import nn
from torch.utils.data import Dataset
from tqdm import tqdm

from chinese_checkers.game import ChineseCheckersGame, Move
from chinese_checkers.geometry import Centroid
from chinese_checkers.simulation import GameSimulation
from .DeepQModelNetwork import DeepQModelNetwork
from .TrainableModel import TrainableModel


def process_simulation(sim_and_board_dim_and_player):
    """Processes a single simulation: precomputes states, moves, and rewards."""
    simulation, board_dim, player_id = sim_and_board_dim_and_player

    # Filter the game sequence and moves to only include the player_id
    full_game_sequence = simulation.to_game_sequence()

    game_sequence = [
        game for game in full_game_sequence
        if game.get_current_player().player_id == player_id
    ]
    moves = [
        move for game, move in zip(full_game_sequence, simulation.data.historical_moves)
        if game.get_current_player().player_id == player_id
    ]

    # Encode states, moves, and rewards for the filtered sequence
    encoded_states = [encode_game(game, board_dim) for game in game_sequence]
    encoded_moves = [encode_move(move, board_dim) for move in moves]
    rewards = encode_reward(simulation, board_dim)  # Precompute all rewards

    return encoded_states, encoded_moves, rewards


class SimulationDataset(Dataset):
    def __init__(self, simulations: List[GameSimulation], board_dim: int, player_id: str, num_workers: int = None):
        start_time = time.time()  # Start the timer
        self.board_dim = board_dim
        self.player_id = player_id

        # Determine the number of workers (default to CPU count if not provided)
        if num_workers is None:
            num_workers = min(int(cpu_count() / 4), len(simulations))

        # Precompute all encodings in parallel
        self.encoded_data = self._compute_encoded_data_parallel(simulations, num_workers)

        # Flatten experiences and store them
        self.experiences = []
        for states, moves, rewards in self.encoded_data:
            self.experiences.extend(zip(states, moves, rewards))

        end_time = time.time()  # Stop the timer
        elapsed_time = end_time - start_time
        print(f"Dataset created with {len(self)} experiences in {elapsed_time:.2f} seconds.")

    def _compute_encoded_data_parallel(self, simulations, num_workers):
        print(f"Precomputing {len(simulations)} simulation encodings in parallel across {num_workers} workers...")
        args = [(sim, self.board_dim, self.player_id) for sim in simulations]
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            encoded_data = list(executor.map(process_simulation, args))
        return encoded_data

    def __len__(self):
        return len(self.experiences)

    def __getitem__(self, idx):
        state, move, reward = self.experiences[idx]

        # Convert data to torch tensors
        state = state.float()
        move = move.float()
        reward = reward.clone().detach().float()

        return state, move, reward


def encode_game(game: ChineseCheckersGame, board_dim: int) -> torch.Tensor:
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


def encode_move(move: Move, board_dim: int) -> torch.Tensor:
    move_tensor = torch.zeros((2, board_dim * 2, board_dim * 2), dtype=torch.float32)

    start_x, start_y = move.position.i + board_dim, move.position.j + board_dim
    move_tensor[0, start_x, start_y] = 1

    end_position = move.apply()
    end_x, end_y = end_position.i + board_dim, end_position.j + board_dim
    move_tensor[1, end_x, end_y] = 1

    return move_tensor


def encode_reward(simulation: GameSimulation, board_dim: int) -> torch.Tensor:
    moves = simulation.data.historical_moves
    game_sequence = simulation.to_game_sequence()
    game_length = len(game_sequence)
    winner_id = simulation.metadata.winning_player

    rewards = np.zeros(len(moves), dtype=np.float32)
    for idx, (move, game) in enumerate(zip(moves, game_sequence)):
        rewards[idx] = _calculate_move_reward(
            game, move, idx, game_length, board_dim, winner_id
        )

    return torch.tensor(rewards, dtype=torch.float32)


def _calculate_move_reward(game: ChineseCheckersGame, move: Move, turn: int, game_length: int, board_dim: int, winner_id: str) -> float:
    # weight_factor = 1 / 4 * (turn / game_length)
    return sum([
        _player_distance_from_target(game, move, board_dim),# * weight_factor,
        _player_positions_in_target(game, move),
        _player_positions_not_in_start(game, move)
        # _distance_from_win_loss(game, turn, game_length, winner_id)/4
    ])

def _distance_from_win_loss(game: ChineseCheckersGame, turn: int, game_length: int, winner_id: str) -> float:
    current_player = game.get_current_player()
    return (turn + 1) / game_length if current_player.player_id == winner_id else -0.25 * (turn + 1) / game_length


def _player_distance_from_target(game: ChineseCheckersGame, move: Move, board_dim: int) -> float:
    board_size: int = game.board.radius
    current_player = game.get_current_player()
    target_positions = current_player.target_positions
    new_positions = current_player.apply_move(move).positions
    current_centroid = Centroid.from_vectors(new_positions)
    target_centroid = Centroid.from_vectors(target_positions)
    distance = current_centroid.distance(target_centroid) / board_size
    return 1 if distance == 0 else ((1 / distance) - 0.47140452079103173) / (40.0 - 0.47140452079103173)


def _player_positions_in_target(game: ChineseCheckersGame, move: Move) -> float:
    current_player = game.get_current_player()
    target_positions = set(current_player.target_positions)
    new_positions = current_player.apply_move(move).positions
    # if len([pos for pos in new_positions if pos in target_positions]) == len(target_positions):
    #     return 10
    return sum(1 for pos in new_positions if pos in target_positions) / len(new_positions)


def _player_positions_not_in_start(game: ChineseCheckersGame, move: Move) -> float:
    current_player = game.get_current_player()
    start_positions = set(current_player.positions)
    new_positions = current_player.apply_move(move).positions
    return sum(1 for pos in start_positions if pos not in new_positions) / len(new_positions)


class DeepQModel(TrainableModel):
    def encode_reward(self, simulation: GameSimulation) -> List[torch.Tensor]:
        return encode_reward(simulation, self.board_dim)

    def __init__(
        self,
        board_size: int,
        player_id: str = "0",
        state_output_dim: int = 32,
        move_output_dim: int = 16,
        q_hidden_dim: int = 128,
        generation: int = 0,
        exploration_prob: int = 0.01,
    ):
        self.board_size = board_size
        self.board_dim = 2 * (board_size + 1)

        self.state_dim = 3  # Channels in state encoding (from encode_game)
        self.move_dim = 2   # Channels in move encoding (from encode_move)
        self.generation = generation
        self.exploration_prob = exploration_prob

        self.model_params = {
            'board_size': board_size,
            'state_output_dim': state_output_dim,
            'move_output_dim': move_output_dim,
            'q_hidden_dim': q_hidden_dim,
            'generation': generation+1,
            'player_id': player_id,
        }
        self.player_id = player_id

        self.network = DeepQModelNetwork(
            state_dim=self.state_dim,
            state_grid_h=self.board_dim * 2,
            state_grid_w=self.board_dim * 2,
            move_dim=self.move_dim,
            move_grid_h=self.board_dim * 2,
            move_grid_w=self.board_dim * 2,
            state_output_dim=state_output_dim,
            move_output_dim=move_output_dim,
            q_hidden_dim=q_hidden_dim,
        )

        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network.to(self.device)

    def train(self, simulations: List[GameSimulation], batch_size: int = 64, num_workers: int = None):
        dataset = SimulationDataset(simulations, self.board_dim, player_id=self.player_id, num_workers=num_workers)

        # Initialize accumulators for batching
        state_batch = []
        move_batch = []
        reward_batch = []

        for state, move, reward in tqdm(dataset, desc="Training model", unit="batch", leave=False, dynamic_ncols=True):
            state_batch.append(state)
            move_batch.append(move)
            reward_batch.append(reward)

            # Process in batches
            if len(state_batch) == batch_size:
                states = torch.stack(state_batch).to(self.device)
                moves = torch.stack(move_batch).to(self.device)
                rewards = torch.stack(reward_batch).to(self.device)

                # Forward pass
                predicted_qs = self.network(states, moves).squeeze(-1)
                loss = self.loss_fn(predicted_qs, rewards)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Clear the batch
                state_batch.clear()
                move_batch.clear()
                reward_batch.clear()

        # Process remaining items in the last batch if any
        if state_batch:
            states = torch.stack(state_batch).to(self.device)
            moves = torch.stack(move_batch).to(self.device)
            rewards = torch.stack(reward_batch).to(self.device)

            predicted_qs = self.network(states, moves).squeeze(-1)
            loss = self.loss_fn(predicted_qs, rewards)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def _chose_next_move(self, game: ChineseCheckersGame) -> Move:
        """
        Chooses the next move with tunable random noise based on self.exploration_prob.

        Returns:
            Move: The selected move.
        """
        encoded_state = self.encode_game(game).float().to(self.device)
        possible_moves = game.get_next_moves(remove_backwards_moves=True)

        # If random noise is triggered, select a random move
        if random.random() < self.exploration_prob:
            return random.choice(possible_moves)

        # Otherwise, select the best move based on Q-values
        best_move = None
        max_q_value = float("-inf")

        for move in possible_moves:
            encoded_move = self.encode_move(move).float().to(self.device)
            q_value = self.network(encoded_state.unsqueeze(0), encoded_move.unsqueeze(0)).item()
            if q_value > max_q_value:
                max_q_value = q_value
                best_move = move

        return best_move

    def encode_game(self, game: ChineseCheckersGame) -> torch.Tensor:
        return encode_game(game, self.board_dim)

    def encode_move(self, move: Move) -> torch.Tensor:
        return encode_move(move, self.board_dim)

    def save(self, path: str):
        checkpoint = {
            'model_params': self.model_params,
            'state_dict': self.network.state_dict()
        }
        torch.save(checkpoint, path)
        print(f"Model saved to {path}")

    @staticmethod
    def load(path: str) -> 'DeepQModel':
        checkpoint = torch.load(path, map_location=torch.device('cpu'), weights_only=True)

        model_params = checkpoint['model_params']
        state_dict = checkpoint['state_dict']

        model = DeepQModel(
            board_size=model_params['board_size'],
            state_output_dim=model_params['state_output_dim'],
            move_output_dim=model_params['move_output_dim'],
            q_hidden_dim=model_params['q_hidden_dim']
        )

        model.network.load_state_dict(state_dict)
        model.network.eval()

        print(f"Model loaded from {path}")
        return model
