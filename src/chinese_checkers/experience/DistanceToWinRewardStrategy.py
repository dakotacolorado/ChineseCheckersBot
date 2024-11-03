from ..experience.IRewardStrategy import IRewardStrategy
from ..game.ChineseCheckersGame import ChineseCheckersGame
from typing import List

class DistanceToWinRewardStrategy(IRewardStrategy):
    """
    A reward strategy that calculates rewards based on the distance to a win or loss state.
    Rewards are higher when the player wins in fewer turns and negative if the player loses.
    """

    def calculate_reward(
        self,
        game_sequence: List[ChineseCheckersGame],
        winning_player: int,
        turn_index: int
    ) -> float:
        """
        Calculate the reward for the player at the specified turn index based on their
        proximity to a win or loss.

        Args:
            game_sequence (List[ChineseCheckersGame]): The sequence of game states in the simulation.
            winning_player (int): The ID of the winning player, if any.
            turn_index (int): The current turn index in the game simulation.

        Returns:
            float: The calculated reward for this turn.
        """
        total_turns = len(game_sequence) - 1  # Total turns based on sequence length

        # Get the current game state and player
        current_game_state = game_sequence[turn_index]
        current_player = current_game_state.get_current_player()

        # If the current player is the winning player, calculate positive reward based on turn
        if winning_player == current_player.player_id:
            reward = (turn_index + 1) / total_turns  # Reward proportional to speed of win
        else:
            # Negative reward for a loss
            reward = -((turn_index + 1) / total_turns)

        return reward
