import json
from dataclasses import dataclass
from typing import Dict, Any

import numpy as np
import pandas as pd
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

    def to_dataframe(self) -> pd.DataFrame:
        """
        Serializes ExperienceData to a Pandas DataFrame suitable for Parquet storage.
        - `state` and `action` are serialized as JSON strings.
        - `reward` is stored as a scalar float.
        - `next_state` is serialized as a JSON string.
        - `done` is stored as an integer (0 for False, 1 for True).
        """
        data = {
            'state': json.dumps(self.state.numpy().tolist()),
            'action': json.dumps(self.action.numpy().tolist()),
            'reward': float(self.reward.item()),      # Store as scalar float
            'next_state': json.dumps(self.next_state.numpy().tolist()),
            'done': int(self.done.item()),            # Store as int (0 or 1)
        }
        df = pd.DataFrame([data])

        # Explicitly set data types to prevent unintended type conversions
        df = df.astype({
            'reward': 'float64',
            'done': 'int8'    # Store 'done' as int8
        })

        return df

    @staticmethod
    def from_dataframe(row: pd.Series) -> 'ExperienceData':
        """
        Deserializes a Pandas DataFrame row back into an ExperienceData instance.
        - `state` and `action` are deserialized from JSON strings.
        - `reward` is converted to a 1-element float tensor.
        - `next_state` is deserialized from a JSON string.
        - `done` is converted to a 1-element bool tensor.
        """
        # Deserialize state and action
        state = torch.tensor(json.loads(row['state']))
        action = torch.tensor(json.loads(row['action']))

        # Deserialize reward
        reward_value = row['reward']
        if isinstance(reward_value, (float, int)):
            reward_value = float(reward_value)
        elif isinstance(reward_value, np.floating):
            reward_value = float(reward_value)
        elif isinstance(reward_value, np.integer):
            reward_value = float(reward_value)
        else:
            raise ValueError(f"Unsupported type for reward: {type(reward_value)}")
        reward = torch.tensor([reward_value], dtype=torch.float32)  # 1-element tensor

        # Deserialize next_state
        next_state = torch.tensor(json.loads(row['next_state']))

        # Deserialize done
        done_value = row['done']
        if isinstance(done_value, (int, float)):
            done_bool = bool(done_value != 0)
        elif isinstance(done_value, np.integer):
            done_bool = bool(done_value)
        elif isinstance(done_value, np.floating):
            done_bool = bool(done_value != 0)
        else:
            raise ValueError(f"Unsupported type for done: {type(done_value)}")
        done = torch.tensor([done_bool], dtype=torch.bool)  # 1-element tensor

        return ExperienceData(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done
        )