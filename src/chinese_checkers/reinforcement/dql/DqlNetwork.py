import torch
import torch.nn as nn

class DQLNetwork(nn.Module):
    def __init__(self, input_dim: int):
        super(DQLNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)  # Output a single Q-value for scalar rewards

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)  # Shape: (batch_size, 1)
