import torch
import torch.nn as nn


class DqlCnnNetwork(nn.Module):
    def __init__(self, state_output_dim: int, move_output_dim: int):
        super(DqlCnnNetwork, self).__init__()
        self.fc1 = nn.Linear(state_output_dim + move_output_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)  # Output a single Q-value for scalar rewards

    def forward(self, state_encoded, move_encoded):
        x = torch.cat((state_encoded, move_encoded), dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)  # Output Q-value
