from typing import List
from ..simulation.GameSimulation import GameSimulation
from ..experience.IRewardStrategy import IRewardStrategy
from ..experience.Experience import Experience


class ExperienceGenerator:
    """
    Generates experiences from a GameSimulation for use in reinforcement learning.

    This class iterates over all turns in a game simulation to produce a sequence of
    experience tuples, each representing a transition between game states.
    """

    def __init__(self, reward_strategy: IRewardStrategy):
        """
        Initialize the ExperienceGenerator with a reward strategy.

        Args:
            reward_strategy (IRewardStrategy): Strategy for calculating rewards for each turn.
        """
        self.reward_strategy = reward_strategy

    def generate_experiences_from_simulation(self, simulation: GameSimulation) -> List[Experience]:
        """
        Generate a list of experiences from a single game simulation.

        Args:
            simulation (GameSimulation): The game simulation to generate experiences from.

        Returns:
            List[Experience]: A list of Experience objects for each turn in the simulation.
        """
        experiences = []
        game_sequence = simulation._to_game_sequence()  # Generate sequence once
        winning_player = simulation.metadata.winning_player  # Retrieve the winning player ID

        for turn_index in range(len(game_sequence) - 1):
            current_game_state = game_sequence[turn_index]
            next_game_state = game_sequence[turn_index + 1]
            is_terminal = next_game_state.is_game_won()

            # Get the action taken at this turn
            action = simulation.data.historical_moves[turn_index]

            # Calculate reward using the strategy with the new parameters
            reward = self.reward_strategy.calculate_reward(
                game_sequence=game_sequence,
                winning_player=winning_player,
                turn_index=turn_index
            )

            # Create an Experience object and add it to the list
            experience = Experience(
                state=current_game_state,
                action=action,
                reward=reward,
                next_state=next_game_state,
                done=is_terminal
            )
            experiences.append(experience)

        return experiences
