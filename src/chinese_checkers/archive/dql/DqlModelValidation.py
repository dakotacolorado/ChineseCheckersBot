import torch
from torch import nn
from typing import List
from chinese_checkers.archive.experience import ExperienceData

class DqlModelValidation:
    def __init__(self, model: nn.Module, gamma: float, test_set: List[ExperienceData], validation_set: List[ExperienceData]):
        self.model = model
        self.gamma = gamma
        self.test_set = test_set
        self.validation_set = validation_set

    def validate(self) -> dict:
        test_error = self._compute_average_error(self.test_set)
        validation_error = self._compute_average_error(self.validation_set)
        validation_error_discounted_returns = self._compute_average_error_discounted_returns(self.validation_set)

        return {
            "test_error": test_error,
            "validation_error": validation_error,
            "validation_error_discounted_returns": validation_error_discounted_returns
        }

    def _compute_average_error(self, experiences: List[ExperienceData]) -> float:
        total_error = 0.0
        loss_fn = nn.MSELoss()

        for experience in experiences:
            # Compute Q-value for the current state-action pair
            state_action = torch.cat((experience.state.unsqueeze(0), experience.action.unsqueeze(0)), dim=1)
            predicted_q_value = self.model(state_action).squeeze()

            # Use the reward as the target Q-value since we have a 1D reward
            target_q_value = experience.reward.squeeze()

            # Calculate error between predicted Q-value and actual reward
            error = loss_fn(predicted_q_value, target_q_value)
            total_error += error.item()

        return total_error / len(experiences) if experiences else 0.0

    def _compute_average_error_discounted_returns(self, experiences: List[ExperienceData]) -> float:
        total_error = 0.0
        loss_fn = nn.MSELoss()
        discounted_return = 0.0

        # Process experiences in reverse to calculate the discounted return for each step
        for experience in reversed(experiences):
            # Update the cumulative discounted return: R = reward + gamma * R
            discounted_return = experience.reward + self.gamma * discounted_return

            # Compute Q-value for the current state-action pair
            state_action = torch.cat((experience.state.unsqueeze(0), experience.action.unsqueeze(0)), dim=1)
            predicted_q_value = self.model(state_action).squeeze()

            # Calculate error between predicted Q-value and cumulative discounted return
            error = loss_fn(predicted_q_value, torch.tensor(discounted_return))
            total_error += error.item()

        # Average the error across all experiences
        return total_error / len(experiences) if experiences else 0.0

