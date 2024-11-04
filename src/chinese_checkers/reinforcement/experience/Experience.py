from dataclasses import dataclass
from torch import Tensor

@dataclass
class Experience:
    state: Tensor
    action:Tensor
    reward: Tensor
    next_state: Tensor
    done: Tensor




