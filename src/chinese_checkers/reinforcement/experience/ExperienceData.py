from dataclasses import dataclass
from typing import Dict, Any
import torch
from torch import Tensor
import numpy as np
from chinese_checkers.catalog import IData


@dataclass(frozen=True)
class ExperienceData(IData):
    state: Tensor
    action: Tensor
    reward: Tensor
    next_state: Tensor
    done: Tensor

    def to_storable(self) -> Dict[str, Any]:
        """
        Converts ExperienceData to a storable format compatible with h5py by converting tensors to numpy arrays.
        Ensures that all arrays have at least one dimension to avoid scalar dataspace errors.
        """
        return {
            "state": self.state.numpy().reshape(-1),  # Ensures at least 1D
            "action": self.action.numpy().reshape(-1),  # Ensures at least 1D
            "reward": self.reward.numpy().reshape(-1),  # Ensures at least 1D
            "next_state": self.next_state.numpy().reshape(-1),  # Ensures at least 1D
            "done": self.done.numpy().reshape(-1)  # Ensures at least 1D
        }

    @staticmethod
    def from_storable(data: Dict[str, Any]) -> 'ExperienceData':
        """
        Reconstructs an ExperienceData instance from stored h5py data by converting numpy arrays back to tensors.
        Ensures that the shapes of the arrays are compatible with the expected tensor format.
        """
        if not all(key in data for key in ["state", "action", "reward", "next_state", "done"]):
            raise ValueError("The provided data dictionary is missing required keys.")

        return ExperienceData(
            state=torch.tensor(np.atleast_1d(data["state"])),
            action=torch.tensor(np.atleast_1d(data["action"])),
            reward=torch.tensor(np.atleast_1d(data["reward"])),
            next_state=torch.tensor(np.atleast_1d(data["next_state"])),
            done=torch.tensor(np.atleast_1d(data["done"]))
        )
