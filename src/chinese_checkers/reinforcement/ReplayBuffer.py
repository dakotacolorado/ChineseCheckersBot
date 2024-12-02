from collections import deque
from typing import List

import numpy as np

from .Experience import Experience
from chinese_checkers.simulation import GameSimulation



class ReplayBuffer:
    def __init__(self, buffer_size: int, player_id: str, board_size: int):
        self.buffer_size = buffer_size
        self.player_id = player_id
        self.board_size = board_size
        self.buffer = deque(maxlen=buffer_size)

    def add(self, simulation: GameSimulation):
        game_sequence = simulation.to_game_sequence()
        move_sequence = simulation.data.historical_moves
        for move, game in zip(move_sequence, game_sequence):
            self.buffer.append(Experience(
                winning_player=simulation.metadata.winning_player,
                board_size=self.board_size,
                game=game,
                move=move,
                turn=game.turn,
                total_turns=len(game_sequence),
            ))

    def sample(self, sample_size: int) -> List[Experience]:
        if len(self.buffer) < sample_size:
            raise ValueError(f"Not enough experiences in buffer to sample {sample_size}.  Only {len(self.buffer)} in buffer.")
        print(f"Sampling {sample_size} experiences from buffer of size {len(self.buffer)} ({sample_size / len(self.buffer) * 100:.2f}% of buffer).")
        idx = np.random.choice(np.arange(len(self.buffer)), size=sample_size, replace=False)
        return [self.buffer[i] for i in idx]

    def __len__(self):
        return len(self.buffer)

    def print_status(self):
        buffer_percentage = len(self.buffer) / self.buffer_size * 100
        print(f"ReplayBuffer status: {buffer_percentage:.2f}% full ({len(self.buffer)}/{self.buffer_size} experiences).")
