import logging
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from typing import List
from torch import Tensor

from build.lib.chinese_checkers.reinforcement.cnn import CnnEncoderExperience
from .DqlCnnNetwork import DqlCnnNetwork
from .CnnNetworkMove import CnnNetworkMove
from .CnnNetworkState import CnnNetworkState
from chinese_checkers.experience import ExperienceData
from ..simulation import GameSimulation


class DqlCnnAgent:
    logger = logging.getLogger(__name__)

    def __init__(self, state_encoder, move_encoder, experience_encoder: CnnEncoderExperience, gamma: float = 0.99, lr: float = 0.001,
                 epsilon_start: float = 1.0, epsilon_min: float = 0.1, epsilon_decay: float = 0.995,
                 buffer_size: int = 10000, batch_size: int = 64, weight_decay: float = 1e-4):

        # Check if CUDA is available and set the device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")

        # Initialize agent parameters
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.experience_encoder = experience_encoder

        # Retrieve dimensions from the encoders
        state_dim, state_grid_h, state_grid_w = state_encoder.shape()
        move_dim, move_grid_h, move_grid_w = move_encoder.shape()

        # Initialize CNN networks for state and move encodings
        self.state_cnn = CnnNetworkState(state_dim, state_grid_h, state_grid_w, 64).to(self.device)
        self.move_cnn = CnnNetworkMove(move_dim, move_grid_h, move_grid_w, 64).to(self.device)

        # Initialize DQL network with combined encoded outputs
        self.q_network = DqlCnnNetwork(state_output_dim=64, move_output_dim=64).to(self.device)
        self.target_network = DqlCnnNetwork(state_output_dim=64, move_output_dim=64).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Optimizer with weight decay for L2 regularization
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr, weight_decay=weight_decay)

        # Replay buffer
        self.replay_buffer = deque(maxlen=buffer_size)

    def select_action(self, encoded_state, possible_moves: List[Tensor]) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, len(possible_moves) - 1)  # Explore

        q_values = []
        for encoded_move in possible_moves:
            # Process state and move encodings through respective CNNs
            state_encoded = self.state_cnn(encoded_state.unsqueeze(0).to(self.device))
            move_encoded = self.move_cnn(encoded_move.unsqueeze(0).to(self.device))
            q_value = self.q_network(state_encoded, move_encoded).item()
            q_values.append(q_value)

        return int(torch.argmax(torch.tensor(q_values)))

    def store_experience(self, experience: ExperienceData):
        self.replay_buffer.append(experience)

    def sample_experiences(self) -> List[ExperienceData]:
        return random.sample(self.replay_buffer, self.batch_size)

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return  # Not enough samples to train

        batch = self.sample_experiences()

        # Stack experiences directly with expected shape and move to the device
        states = torch.stack([exp.state for exp in batch]).to(self.device)
        actions = torch.stack([exp.action for exp in batch]).to(self.device)
        rewards = torch.tensor([exp.reward for exp in batch], dtype=torch.float32).view(-1, 1).to(self.device)
        next_states = torch.stack([exp.next_state for exp in batch]).to(self.device)
        dones = torch.tensor([exp.done for exp in batch], dtype=torch.float32).view(-1, 1).to(self.device)

        # Process encodings through CNNs
        state_encodings = self.state_cnn(states)
        action_encodings = self.move_cnn(actions)
        q_values = self.q_network(state_encodings, action_encodings)

        with torch.no_grad():
            next_state_encodings = self.state_cnn(next_states)
            next_q_values = self._compute_batched_next_q_values(next_state_encodings, actions)

        # Calculate target Q-values
        target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        loss = nn.MSELoss()(q_values, target_q_values.detach())

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon for exploration
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # Periodically update target network
        if random.randint(1, 100) <= 10:
            self.target_network.load_state_dict(self.q_network.state_dict())

    def validate(self, test_set: List[ExperienceData], validation_set: List[ExperienceData], validation_simulations: List[GameSimulation]) -> dict:
        test_error = self._compute_average_error(test_set)
        validation_error = self._compute_average_error(validation_set)
        validation_error_discounted_returns = self._compute_average_error_discounted_returns(validation_simulations)

        return {
            "test_error": test_error,
            "validation_error": validation_error,
            "validation_error_discounted_returns": validation_error_discounted_returns
        }

    def save(self, save_path: str):
        """
        Saves the state of the CNN and Q-network models to the specified path.

        Args:
            save_path (str): The file path to save the model.
        """
        torch.save({
            'state_cnn': self.state_cnn.state_dict(),
            'move_cnn': self.move_cnn.state_dict(),
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, save_path)
        self.logger.info(f"Model saved to {save_path}")

    def _compute_average_error(self, experiences: List[ExperienceData]) -> float:
        total_error = 0.0
        loss_fn = nn.MSELoss()

        for experience in experiences:
            state_tensor = experience.state.unsqueeze(0).to(self.device)
            action_tensor = experience.action.unsqueeze(0).to(self.device)

            with torch.no_grad():
                state_encoded = self.state_cnn(state_tensor)
                action_encoded = self.move_cnn(action_tensor)
                predicted_q_value = self.q_network(state_encoded, action_encoded).squeeze()

            target_q_value = experience.reward.to(self.device).squeeze()
            error = loss_fn(predicted_q_value, target_q_value)
            total_error += error.item()

        return total_error / len(experiences) if experiences else 0.0

    def _compute_average_error_discounted_returns(self, game_simulations: List[GameSimulation]) -> float:
        """
        Computes the average error of discounted returns across multiple game simulations.

        Each simulation is processed sequentially, and the discounted returns are calculated
        based on the game sequence. The function computes the Mean Squared Error (MSE) between
        the predicted Q-values and the discounted returns for each simulation.

        Args:
            game_simulations (List[GameSimulation]): A list of game simulations in chronological order.

        Returns:
            float: The squared sum of average errors divided by the total number of games.
        """
        scores = []
        for simulation in game_simulations:
            # Encode experiences from the simulation
            experiences = [e.data for e in self.experience_encoder.encode(simulation)]
            total_error = 0.0
            loss_fn = nn.MSELoss()
            discounted_return = 0.0

            # Process experiences in reverse order to compute discounted returns
            for experience in reversed(experiences):
                discounted_return = experience.reward.to(self.device) + self.gamma * discounted_return

                # Prepare tensors for state and action
                state_tensor = experience.state.unsqueeze(0).to(self.device)
                action_tensor = experience.action.unsqueeze(0).to(self.device)

                # Predict Q-value
                with torch.no_grad():
                    state_encoded = self.state_cnn(state_tensor)
                    action_encoded = self.move_cnn(action_tensor)
                    predicted_q_value = self.q_network(state_encoded, action_encoded).squeeze()

                # Compute error
                error = loss_fn(predicted_q_value, discounted_return.squeeze())
                total_error += error.item()

            # Compute average error for the simulation
            scores.append(total_error / len(experiences) if experiences else 0.0)

        # Return the squared sum of scores divided by the total number of games
        return sum(score ** 2 for score in scores) / len(game_simulations) if game_simulations else 0.0

    def _compute_batched_next_q_values(self, next_state_encodings, actions) -> Tensor:
        all_q_values = []
        for move in actions:
            move_encoding = self.move_cnn(move.unsqueeze(0).to(self.device))
            q_values = self.target_network(next_state_encodings, move_encoding.expand(next_state_encodings.size(0), -1))
            all_q_values.append(q_values)

        all_q_values = torch.stack(all_q_values, dim=1)
        max_q_values, _ = torch.max(all_q_values, dim=1)
        return max_q_values
