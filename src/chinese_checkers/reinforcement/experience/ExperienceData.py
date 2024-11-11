from dataclasses import dataclass
from typing import Dict, Any

import torch
from torch import Tensor

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

        def ensure_min_dim(tensor: Tensor):
            array = tensor.numpy()
            return array if array.ndim > 0 else array.reshape(1)  # Ensures at least 1D

        return {
            "state": ensure_min_dim(self.state),
            "action": ensure_min_dim(self.action),
            "reward": ensure_min_dim(self.reward),
            "next_state": ensure_min_dim(self.next_state),
            "done": ensure_min_dim(self.done)
        }

    @staticmethod
    def from_storable(data: Dict[str, Any]) -> 'ExperienceData':
        """
        Reconstructs an ExperienceData instance from stored data by converting numpy arrays back to tensors.
        """
        required_keys = ["state", "action", "reward", "next_state", "done"]

        # Check that all required keys are present
        if not all(key in data for key in required_keys):
            raise ValueError(f"The provided data dictionary is missing required keys: {required_keys}")

        def restore_tensor(array):
            return torch.tensor(array) if array.ndim > 0 else torch.tensor(
                array[0])  # Restore scalars if 1D with single element

        return ExperienceData(
            state=restore_tensor(data["state"]),
            action=restore_tensor(data["action"]),
            reward=restore_tensor(data["reward"]),
            next_state=restore_tensor(data["next_state"]),
            done=restore_tensor(data["done"])
        )
