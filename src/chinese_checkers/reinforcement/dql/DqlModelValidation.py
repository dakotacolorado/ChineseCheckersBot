import torch
from typing import List, Tuple
import numpy as np

class DqlModelValidation:
    def __init__(self, agent, validation_data: List[Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]]):
        """
        Initialize the DqlModelValidation with a trained DQLAgent and a validation dataset.

        Args:
            agent (DQLAgent): The trained DQLAgent to be validated.
            validation_data (List[Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]]):
                Encoded validation data consisting of tuples (state, action, simulation, next_state, done).
        """
        self.agent = agent
        self.validation_data = validation_data

    def evaluate(self) -> float:
        """
        Evaluate the agent's performance on the validation dataset.

        Returns:
            float: The average difference between predicted and actual Q-values.
        """
        total_difference = 0.0
        num_samples = len(self.validation_data)

        for experience in self.validation_data:
            state, action, reward, next_state, done = experience

            # Convert numpy arrays to tensors
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_tensor = torch.FloatTensor(action).unsqueeze(0)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)

            # Get the predicted Q-value from the agent for the state-action pair
            state_action_combined = torch.cat((state_tensor, action_tensor), dim=1)
            predicted_q_value = self.agent.q_network(state_action_combined).item()

            # Calculate the expected Q-value
            if done:
                expected_q_value = reward  # No future simulation because the episode is done
            else:
                # Compute the maximum Q-value for the next state
                next_q_values = [
                    self.agent.q_network(torch.cat((next_state_tensor, torch.FloatTensor(move).unsqueeze(0)), dim=1)).item()
                    for move in self.agent.replay_buffer
                ]
                max_next_q_value = max(next_q_values)
                expected_q_value = reward + self.agent.gamma * max_next_q_value

            # Accumulate the difference
            total_difference += abs(predicted_q_value - expected_q_value)

        # Return the average difference
        return total_difference / num_samples