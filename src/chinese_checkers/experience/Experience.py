from dataclasses import dataclass
from ..game.ChineseCheckersGame import ChineseCheckersGame
from ..game.Move import Move


@dataclass
class Experience:
    """
    A single experience tuple representing the state transition and reward in reinforcement learning.

    Attributes:
        state (ChineseCheckersGame): The game state before the action was taken.
        action (Move): The action taken in the current state.
        reward (float): The reward received after taking the action.
        next_state (ChineseCheckersGame): The resulting game state after the action.
        done (bool): Whether the game has ended after this action.
    """
    state: ChineseCheckersGame
    action: Move
    reward: float
    next_state: ChineseCheckersGame
    done: bool
