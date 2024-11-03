import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from typing import Tuple, List

from .DqlNetwork import DQLNetwork

class DQLAgent:
    def __init__(self, state_dim: int, move_dim: int, gamma: float = 0.99,
                 lr: float = 0.001, epsilon_start: float = 1.0, epsilon_min: float = 0.1,
                 epsilon_decay: float = 0.995, buffer_size: int = 10000, batch_size: int = 64):
        self.state_dim = state_dim
        self.move_dim = move_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size

        # Initialize Q-network and target network with combined input dimension
        combined_input_dim = state_dim + move_dim
        self.q_network = DQLNetwork(combined_input_dim)
        self.target_network = DQLNetwork(combined_input_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        # Replay buffer
        self.replay_buffer = deque(maxlen=buffer_size)

    def select_action(self, state: np.ndarray, possible_moves: List[np.ndarray]) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, len(possible_moves) - 1)  # Explore

        state = torch.FloatTensor(state).unsqueeze(0)
        move_tensors = [torch.FloatTensor(move).unsqueeze(0) for move in possible_moves]

        # Compute Q-values for each move and select the best one
        q_values = [self.q_network(torch.cat((state, move), dim=1)).item() for move in move_tensors]
        return np.argmax(q_values)

    def store_experience(self, experience: Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]):
        self.replay_buffer.append(experience)

    def sample_experiences(self) -> List[Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]]:
        return random.sample(self.replay_buffer, self.batch_size)

    def _compute_next_q_values(self, next_states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        # Compute the maximum Q-value for each next state
        next_q_values = []
        for next_state in next_states:
            max_q_value = float('-inf')
            for move in actions:
                next_state_action = torch.cat((next_state.unsqueeze(0), move.unsqueeze(0)), dim=1)
                q_value = self.target_network(next_state_action).item()
                max_q_value = max(max_q_value, q_value)
            next_q_values.append(max_q_value)
        return torch.FloatTensor(next_q_values)

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return  # Not enough samples to train

        batch = self.sample_experiences()
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert batch data to tensors
        states = torch.FloatTensor(np.array(states))
        actions = torch.FloatTensor(np.array(actions))
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)

        # Compute Q-values for current state-action pairs
        state_actions = torch.cat((states, actions), dim=1)
        q_values = self.q_network(state_actions).squeeze()

        # Compute target Q-values using the helper function
        next_q_values = self._compute_next_q_values(next_states, actions)
        target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        # Calculate loss and perform optimization step
        loss = nn.MSELoss()(q_values, target_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update epsilon for exploration
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # Periodic update of target network weights
        if random.randint(1, 100) <= 10:
            self.target_network.load_state_dict(self.q_network.state_dict())

    def run_training_sample(self, encoded_sample: Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]):
        self.store_experience(encoded_sample)
        self.train()
