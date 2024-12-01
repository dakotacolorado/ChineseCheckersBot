from collections import deque
import numpy as np


class ReplayBuffer:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)

    def add(self, state, action, reward, next_state, done):
        """Add a single experience to the buffer."""
        self.buffer.append((state, action, reward, next_state, done))

    def add_batch(self, states, actions, rewards, next_states, dones):
        """
        Add a batch of experiences to the buffer.
        :param states: A list or array of states.
        :param actions: A list or array of actions.
        :param rewards: A list or array of rewards.
        :param next_states: A list or array of next states.
        :param dones: A list or array of done flags.
        """
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Sample a batch of experiences from the buffer."""
        if len(self.buffer) < batch_size:
            raise ValueError("Not enough samples in buffer to generate a batch.")
        idx = np.random.choice(np.arange(len(self.buffer)), size=batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in idx])
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

    def __len__(self):
        """Return the current size of the buffer."""
        return len(self.buffer)
