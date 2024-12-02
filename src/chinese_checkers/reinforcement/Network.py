import torch
import torch.nn as nn


class Network(nn.Module):
    def __init__(self, state_dim: int, state_grid_h: int, state_grid_w: int,
                 move_dim: int, move_grid_h: int, move_grid_w: int,
                 state_output_dim: int, move_output_dim: int, q_hidden_dim: int = 128):
        """
        Unified network for state encoding, move encoding, and Q-value prediction.

        Args:
            state_dim (int): Number of input channels for state.
            state_grid_h (int): Height of the state input grid.
            state_grid_w (int): Width of the state input grid.
            move_dim (int): Number of input channels for moves.
            move_grid_h (int): Height of the move input grid.
            move_grid_w (int): Width of the move input grid.
            state_output_dim (int): Output dimension for the state encoder.
            move_output_dim (int): Output dimension for the move encoder.
            q_hidden_dim (int): Hidden dimension for Q-value prediction network.
        """
        super(Network, self).__init__()

        # State encoder
        self.state_cnn = nn.Sequential(
            nn.Conv2d(state_dim, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        dummy_state = torch.zeros(1, state_dim, state_grid_h, state_grid_w)
        flattened_state_size = self.state_cnn(dummy_state).numel()
        self.state_fc = nn.Linear(flattened_state_size, state_output_dim)

        # Move encoder
        self.move_cnn = nn.Sequential(
            nn.Conv2d(move_dim, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        dummy_move = torch.zeros(1, move_dim, move_grid_h, move_grid_w)
        flattened_move_size = self.move_cnn(dummy_move).numel()
        self.move_fc = nn.Linear(flattened_move_size, move_output_dim)

        # Q-value prediction
        self.q_network = nn.Sequential(
            nn.Linear(state_output_dim + move_output_dim, q_hidden_dim),
            nn.ReLU(),
            nn.Linear(q_hidden_dim, q_hidden_dim),
            nn.ReLU(),
            nn.Linear(q_hidden_dim, 1),  # Output a single Q-value
        )

    def forward(self, state, move):
        """
        Forward pass through the unified network.

        Args:
            state (Tensor): Input state tensor with shape (batch_size, state_dim, state_grid_h, state_grid_w).
            move (Tensor): Input move tensor with shape (batch_size, move_dim, move_grid_h, move_grid_w).

        Returns:
            Tensor: Predicted Q-value with shape (batch_size, 1).
        """
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
