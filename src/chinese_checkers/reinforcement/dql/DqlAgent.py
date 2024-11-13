import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from typing import List
from torch import Tensor
from .DqlNetwork import DQLNetwork
from chinese_checkers.experience import ExperienceData


class DQLAgent:
    logger = logging.getLogger(__name__)

    def __init__(self, state_dim: int, move_dim: int, gamma: float = 0.99,
                 lr: float = 0.001, epsilon_start: float = 1.0, epsilon_min: float = 0.1,
                 epsilon_decay: float = 0.995, buffer_size: int = 10000, batch_size: int = 64,
                 weight_decay: float = 1e-4):
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

        # Optimizer with weight decay for L2 regularization
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr, weight_decay=weight_decay)

        # Replay buffer
        self.replay_buffer = deque(maxlen=buffer_size)

    def select_action(self, state: Tensor, possible_moves: List[Tensor]) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, len(possible_moves) - 1)  # Explore

        state = state.unsqueeze(0)
        q_values = [self.q_network(torch.cat((state, move.unsqueeze(0)), dim=1)).item() for move in possible_moves]
        return int(np.argmax(q_values))

    def store_experience(self, experience: ExperienceData):
        self.replay_buffer.append(experience)

    def sample_experiences(self) -> List[ExperienceData]:
        return random.sample(self.replay_buffer, self.batch_size)

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return  # Not enough samples to train

        batch = self.sample_experiences()

        # Extract data and convert to tensors
        states = torch.stack([exp.state for exp in batch])
        actions = torch.stack([exp.action for exp in batch])
        rewards = torch.stack([exp.reward for exp in batch]).view(-1, 1)
        next_states = torch.stack([exp.next_state for exp in batch])
        dones = torch.stack([exp.done for exp in batch]).view(-1, 1)

        self.logger.info(f"states shape: {states.shape}")
        self.logger.info(f"actions shape: {actions.shape}")
        self.logger.info(f"rewards shape: {rewards.shape}")
        self.logger.info(f"next_states shape: {next_states.shape}")
        self.logger.info(f"dones shape: {dones.shape}")

        # Compute Q-values for current state-action pairs
        state_actions = torch.cat((states, actions), dim=1)
        q_values = self.q_network(state_actions)
        self.logger.info(f"q_values shape: {q_values.shape}")

        # Compute target Q-values for next states
        next_q_values = self._compute_batched_next_q_values(next_states, actions).view(-1, 1)
        self.logger.info(f"next_q_values shape: {next_q_values.shape}")

        # Calculate target Q-values
        target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        self.logger.info(f"target_q_values shape: {target_q_values.shape}")

        # Calculate loss with L2 regularization and optimize
        loss = nn.MSELoss()(q_values, target_q_values.detach())
        self.logger.info(f"loss: {loss.item()}")
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update epsilon for exploration
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # Periodic target network update
        if random.randint(1, 100) <= 10:
            self.target_network.load_state_dict(self.q_network.state_dict())

    def _compute_batched_next_q_values(self, next_states: Tensor, actions: Tensor) -> Tensor:
        all_q_values = []
        for move in actions:
            next_state_actions = torch.cat((next_states, move.unsqueeze(0).expand(next_states.size(0), -1)), dim=1)
            q_values = self.target_network(next_state_actions)
            all_q_values.append(q_values)

        all_q_values = torch.stack(all_q_values, dim=1)
        max_q_values, _ = torch.max(all_q_values, dim=1)
        return max_q_values
