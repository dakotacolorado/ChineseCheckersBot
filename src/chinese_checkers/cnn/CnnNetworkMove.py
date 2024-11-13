import torch
import torch.nn as nn


class CnnNetworkMove(nn.Module):
    def __init__(self, move_dim: int, grid_size_h: int, grid_size_w: int, output_dim: int):
        super(CnnNetworkMove, self).__init__()
        self.conv1 = nn.Conv2d(move_dim, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)

        # Use a dummy input to compute the output size
        dummy_input = torch.zeros(1, move_dim, grid_size_h, grid_size_w)
        out = torch.relu(self.conv1(dummy_input))
        out = torch.relu(self.conv2(out))
        self.flattened_size = out.numel()

        self.fc = nn.Linear(self.flattened_size, output_dim)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        return self.fc(x)
