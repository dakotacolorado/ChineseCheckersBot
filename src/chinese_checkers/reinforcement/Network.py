import torch
import torch.nn as nn


class Network(nn.Module):
    def __init__(self, state_dim: int, state_grid_h: int, state_grid_w: int,
                 move_dim: int, move_grid_h: int, move_grid_w: int,
                 state_output_dim: int, move_output_dim: int, q_hidden_dim: int = 128):
        super(Network, self).__init__()

        # State encoder with dropout
        self.state_cnn = nn.Sequential(
            nn.Conv2d(state_dim, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(0.3)  # Dropout with 30% probability
        )
        dummy_state = torch.zeros(1, state_dim, state_grid_h, state_grid_w)
        flattened_state_size = self.state_cnn(dummy_state).numel()
        self.state_fc = nn.Sequential(
            nn.Linear(flattened_state_size, state_output_dim),
            nn.ReLU(),
            nn.Dropout(0.3)  # Dropout for the fully connected layer
        )

        # Move encoder with dropout
        self.move_cnn = nn.Sequential(
            nn.Conv2d(move_dim, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(0.3)  # Dropout with 30% probability
        )
        dummy_move = torch.zeros(1, move_dim, move_grid_h, move_grid_w)
        flattened_move_size = self.move_cnn(dummy_move).numel()
        self.move_fc = nn.Sequential(
            nn.Linear(flattened_move_size, move_output_dim),
            nn.ReLU(),
            nn.Dropout(0.3)  # Dropout for the fully connected layer
        )

        # Q-value prediction with dropout
        self.q_network = nn.Sequential(
            nn.Linear(state_output_dim + move_output_dim, q_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(q_hidden_dim, q_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(q_hidden_dim, 1)  # Output a single Q-value
        )

    def forward(self, state, move):
        if state.dtype == torch.int8:
            state = state.float()
        if move.dtype == torch.int8:
            move = move.float()

        # State encoding
        state_encoded = self.state_cnn(state)
        state_encoded = state_encoded.view(state_encoded.size(0), -1)
        state_encoded = self.state_fc(state_encoded)

        # Move encoding
        move_encoded = self.move_cnn(move)
        move_encoded = move_encoded.view(move_encoded.size(0), -1)
        move_encoded = self.move_fc(move_encoded)

        # Q-value prediction
        combined = torch.cat((state_encoded, move_encoded), dim=1)
        q_value = self.q_network(combined)
        return q_value
