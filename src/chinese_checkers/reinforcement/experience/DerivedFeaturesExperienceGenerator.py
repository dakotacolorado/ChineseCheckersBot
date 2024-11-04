from typing import List

from chinese_checkers.game import ChineseCheckersGame
from chinese_checkers.reinforcement.experience import Experience
from chinese_checkers.reinforcement.experience.BaseExperienceGenerator import BaseExperienceGenerator
from chinese_checkers.simulation import GameSimulation


class DerivedFeaturesExperienceGenerator(BaseExperienceGenerator):

    def __init__(self):
        move_encoder_factory =
        self.move_encoder = move_encoder_factory.create("spatial_board_metrics")



    def generate(self, simulation: GameSimulation) -> List[Experience]:
        game_sequence = simulation._to_game_sequence()
        winning_player = simulation.metadata.winning_player

        for turn_index in range(len(game_sequence) - 1):
            current_game_state = game_sequence[turn_index]
            next_game_state = game_sequence[turn_index + 1]
            is_terminal = next_game_state.is_game_won()

            # Get the action taken at this turn
            action = simulation.data.historical_moves[turn_index]

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

    def encode_game_state(self, game_state: ChineseCheckersGame) -> torch.Tensor:
        """
        Encodes the game state into a tensor representation.

        Args:
            game_state (ChineseCheckersGame): The game state to encode.

        Returns:
            torch.Tensor: Encoded game state tensor.
        """
        return self.game_encoder.encode(game_state)

