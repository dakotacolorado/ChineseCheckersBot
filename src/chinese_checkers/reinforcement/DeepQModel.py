import random
import time
from multiprocessing import cpu_count
from typing import List

import torch
from torch import nn
from tqdm import tqdm

from chinese_checkers.game import ChineseCheckersGame, Move
from chinese_checkers.reinforcement.Encoder import Encoder
from chinese_checkers.reinforcement.Network import Network
from .Experience import Experience
from ..model import IModel


class DeepQModel(IModel):
    def __init__(
            self,
            board_size: int,
            player_id: str = "0",
            state_output_dim: int = 32,
            move_output_dim: int = 16,
            q_hidden_dim: int = 128,
            exploration_prob: float = 0.01,
            target_update_freq: int = 1000,
            gamma: float = 0.99
    ):
        """
        Initializes the DeepQModel.

        :param board_size: The size of the board.
        :param player_id: The ID of the player this model controls.
        :param state_output_dim: The output dimension of the state encoding.
        :param move_output_dim: The output dimension of the move encoding.
        :param q_hidden_dim: The hidden dimension of the Q-value computation.
        :param exploration_prob: The probability of choosing a random move for exploration.
        :param target_update_freq: The frequency of updating the target network (# of samples seen).
        """
        self.board_size = board_size
        self.board_dim = 2 * (board_size + 1)
        self.exploration_prob = exploration_prob
        self.training_samples_seen = 0  # Initialize counter for training samples

        # Define model parameters for saving/loading
        self.model_params = {
            'board_size': board_size,
            'state_output_dim': state_output_dim,
            'move_output_dim': move_output_dim,
            'q_hidden_dim': q_hidden_dim,
            'player_id': player_id,
            'gamma': gamma,
        }

        # Initialize the encoder
        self.encoder = Encoder(board_size = board_size, thread_count=max(1, cpu_count() // 4))

        # Initialize the network
        self.network = Network(
            state_dim=self.encoder.encoded_game_shape[0],
            state_grid_h=self.encoder.encoded_game_shape[1],
            state_grid_w=self.encoder.encoded_game_shape[2],
            move_dim=self.encoder.encoded_move_shape[0],
            move_grid_h=self.encoder.encoded_move_shape[1],
            move_grid_w=self.encoder.encoded_move_shape[2],
            state_output_dim=state_output_dim,
            move_output_dim=move_output_dim,
            q_hidden_dim=q_hidden_dim,
        )

        # Set up training components
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()

        # Configure device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network.to(self.device)
        self.gamma = gamma
        # Target network setup
        self.target_network = Network(
            state_dim=self.encoder.encoded_game_shape[0],
            state_grid_h=self.encoder.encoded_game_shape[1],
            state_grid_w=self.encoder.encoded_game_shape[2],
            move_dim=self.encoder.encoded_move_shape[0],
            move_grid_h=self.encoder.encoded_move_shape[1],
            move_grid_w=self.encoder.encoded_move_shape[2],
            state_output_dim=state_output_dim,
            move_output_dim=move_output_dim,
            q_hidden_dim=q_hidden_dim,
        ).to(self.device)

        # Initialize target network weights
        self.target_network.load_state_dict(self.network.state_dict())
        self.target_network.eval()  # Target network doesn't get updated during training

        # Target network update frequency
        self.target_update_freq = target_update_freq

    def train(self, experiences: List[Experience], batch_size: int = 64):
        """
        Train the model using the provided experiences.

        :param experiences: List of Experience objects to train on.
        :param batch_size: The size of batches to use for training.
        """
        num_samples = len(experiences)
        print(f"Starting training with {num_samples} experiences, batch size: {batch_size}.")

        # Use the encoder to process experiences into tensors
        start_time = time.time()
        encoded_states, encoded_moves, encoded_rewards = self.encoder.batch_encode(experiences)
        end_time = time.time()
        print(f"Encoding completed in {end_time - start_time:.2f} seconds.")

        # Initialize accumulators for batching
        state_batch, move_batch, reward_batch = [], [], []

        # Iterate over encoded data
        for state, move, reward in tqdm(
                zip(encoded_states, encoded_moves, encoded_rewards),
                desc="Training model",
                total=len(encoded_states),
                unit="sample",
                leave=False,
                dynamic_ncols=True
        ):
            state_batch.append(state)
            move_batch.append(move)
            reward_batch.append(reward)

            # Process batches
            if len(state_batch) == batch_size:
                self._process_batch(state_batch, move_batch, reward_batch)
                state_batch.clear()
                move_batch.clear()
                reward_batch.clear()

        # Process remaining items in the last batch
        if state_batch:
            self._process_batch(state_batch, move_batch, reward_batch)

        # Update the training samples counter
        self.training_samples_seen += num_samples
        print(f"Training completed. Total samples seen: {self.training_samples_seen}.")

    def _process_batch(self, state_batch: List[torch.Tensor], move_batch: List[torch.Tensor],
                       reward_batch: List[torch.Tensor]):
        # Convert lists to tensors and move them to the appropriate device
        states = torch.stack(state_batch).to(self.device)
        moves = torch.stack(move_batch).to(self.device)
        rewards = torch.stack(reward_batch).to(self.device)

        # Compute Q-values from the main network
        predicted_qs = self.network(states, moves).squeeze(-1)

        # Compute target Q-values using the target network
        with torch.no_grad():
            target_qs = self.target_network(states, moves).squeeze(-1)

        # Compute the loss
        loss = self.loss_fn(predicted_qs, rewards + self.gamma * target_qs)  # Use self.gamma

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Increment training steps and update target network if needed
        if self.training_samples_seen % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.network.state_dict())

    def _choose_next_move(self, game: ChineseCheckersGame) -> Move:
        """
        Chooses the next move with tunable random noise based on self.exploration_prob.

        :param game: Current game state.
        :return: The selected move.
        """
        # Encode the current game state using the encoder
        encoded_state = self.encoder.encode_game(game).float().to(self.device)

        # Retrieve all possible moves
        possible_moves = game.get_next_moves(remove_backwards_moves=True)

        # If exploration noise is triggered, select a random move
        if random.random() < self.exploration_prob:
            return random.choice(possible_moves)

        # Otherwise, evaluate moves to select the best one
        best_move = None
        max_q_value = float("-inf")

        for move in possible_moves:
            # Encode the move using the encoder
            encoded_move = self.encoder.encode_move(move).float().to(self.device)

            # Compute the Q-value for the current move
            q_value = self.network(encoded_state.unsqueeze(0), encoded_move.unsqueeze(0)).item()

            # Track the best move based on Q-value
            if q_value > max_q_value:
                max_q_value = q_value
                best_move = move

        return best_move

    def save(self, path: str):
        """
        Saves the model parameters, state dictionary, and training samples count to a file.

        :param path: Path to the file where the model will be saved.
        """
        checkpoint = {
            'model_params': self.model_params,
            'state_dict': self.network.state_dict(),
            'training_samples_seen': self.training_samples_seen,
        }
        torch.save(checkpoint, path)
        print(f"Model saved to {path}")

    @staticmethod
    def load(path: str) -> 'DeepQModel':
        """
        Loads a model from a checkpoint file.

        :param path: Path to the file from which the model will be loaded.
        :return: An instance of DeepQModel with the loaded parameters.
        """
        checkpoint = torch.load(path, map_location=torch.device('cpu'), weights_only=True)

        model_params = checkpoint['model_params']
        state_dict = checkpoint['state_dict']
        training_samples_seen = checkpoint.get('training_samples_seen',0)

        # Recreate the model with the loaded parameters
        model = DeepQModel(
            board_size=model_params['board_size'],
            state_output_dim=model_params['state_output_dim'],
            move_output_dim=model_params['move_output_dim'],
            q_hidden_dim=model_params['q_hidden_dim'],
            exploration_prob=model_params.get('exploration_prob', 0.01),
            gamma=model_params.get('gamma', 0.99),
        )

        # Load the network weights
        model.network.load_state_dict(state_dict)
        model.network.eval()

        # Restore training samples counter
        model.training_samples_seen = training_samples_seen

        # Reassign the device
        model.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.network.to(model.device)

        print(f"Model loaded from {path}. Training samples seen: {model.training_samples_seen}.")
        return model
