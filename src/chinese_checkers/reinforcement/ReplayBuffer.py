from collections import deque, defaultdict
from typing import List
import numpy as np
from .Experience import Experience
from chinese_checkers.simulation import GameSimulation


class ReplayBuffer:
    def __init__(self, buffer_size: int, player_id: str, board_size: int, win_bias: float = 2.0, max_hash_queue_size: int = 10):
        """
        Args:
            buffer_size (int): Maximum total number of experiences across all hash queues.
            player_id (str): ID of the player for biasing winning experiences.
            board_size (int): Size of the board.
            win_bias (float): Multiplier for prioritizing winning experiences.
            max_hash_queue_size (int): Maximum number of experiences allowed per hash queue.
        """
        self.buffer_size = buffer_size
        self.player_id = player_id
        self.board_size = board_size
        self.win_bias = win_bias
        self.max_hash_queue_size = max_hash_queue_size

        self.experience_dict = defaultdict(lambda: deque(maxlen=max_hash_queue_size))
        self.total_experience_count = 0

    def add(self, simulation: GameSimulation):
        game_sequence = simulation.to_game_sequence()
        move_sequence = simulation.data.historical_moves

        for move, game in zip(move_sequence, game_sequence):
            new_experience = Experience(
                winning_player=simulation.metadata.winning_player,
                board_size=self.board_size,
                game=game,
                move=move,
                turn=game.turn,
                total_turns=len(game_sequence),
            )
            exp_hash = hash(new_experience)

            # Add the experience to its corresponding hash queue
            queue = self.experience_dict[exp_hash]
            if len(queue) == self.max_hash_queue_size:
                popped_exp = queue.popleft()
                # print(f"Replaced duplicate move {popped_exp.move}")
                self.total_experience_count -= 1

            queue.append(new_experience)
            self.total_experience_count += 1

            # Ensure the total buffer size constraint
            if self.total_experience_count > self.buffer_size:
                self._evict_experience()

    def _evict_experience(self):
        # Randomly select a hash queue to evict from, weighted by queue lengths
        hash_keys = list(self.experience_dict.keys())
        queue_lengths = np.array([len(self.experience_dict[key]) for key in hash_keys])
        probabilities = queue_lengths / queue_lengths.sum()
        selected_hash = np.random.choice(hash_keys, p=probabilities)

        # Pop an experience from the selected hash queue
        evicted_exp = self.experience_dict[selected_hash].popleft()
        # print(f"Evicted move from overfull buffer: {evicted_exp.move}")
        self.total_experience_count -= 1

        # Remove the queue from the dict if it's empty
        if not self.experience_dict[selected_hash]:
            del self.experience_dict[selected_hash]

    def sample(self, sample_size: int) -> List[Experience]:
        if self.total_experience_count < sample_size:
            raise ValueError(f"Not enough experiences in buffer to sample {sample_size}. "
                             f"Only {self.total_experience_count} in buffer.")

        # Combine all queues into a single list
        all_experiences = [exp for queue in self.experience_dict.values() for exp in queue]

        # Weighted sampling with win bias
        weights = [
            self.win_bias if exp.winning_player == self.player_id else 1.0
            for exp in all_experiences
        ]
        weights = np.array(weights)
        weights /= weights.sum()  # Normalize weights to sum to 1

        sampled_indices = np.random.choice(len(all_experiences), size=sample_size, replace=False, p=weights)
        return [all_experiences[i] for i in sampled_indices]

    def __len__(self):
        return self.total_experience_count

    def print_status(self):
        buffer_percentage = self.total_experience_count / self.buffer_size * 100
        print(f"ReplayBuffer status: {buffer_percentage:.2f}% full "
              f"({self.total_experience_count}/{self.buffer_size} experiences).")
