import logging
from typing import List

import torch

from .CnnEncoderMove import CnnEncoderMove
from chinese_checkers.simulation import GameSimulation
from .CnnEncoderState import CnnEncoderState
from .RewardEncoder import RewardEncoder
from chinese_checkers.experience import Experience, ExperienceData, ExperienceMetadata
from ...game import ChineseCheckersGame


class CnnEncoderExperience:
    logger = logging.getLogger(__name__)

    def __init__(self, encoder_version: str):
        self.encoder_version = encoder_version

        if self.encoder_version == "v005":
            self.logger.info(f"Using CnnExperienceEncoder version: {self.encoder_version}")
            self._encoder = CnnEncoderExperience.encode_v005
        else:
            raise ValueError(f"Unknown encoder version: {self.encoder_version}")

    def encode(
            self,
            simulation: GameSimulation
    ):
        return self._encoder(simulation)

    @staticmethod
    def encode_v005(
            simulation: GameSimulation
    ):
        move_encoder = CnnEncoderMove(simulation.metadata.board_size)
        reward_encoder = RewardEncoder()
        state_encoder = CnnEncoderState(simulation.metadata.board_size)

        rewards = reward_encoder.encode(simulation)
        game_sequence: List[ChineseCheckersGame] = simulation.to_game_sequence()

        experiences = []
        for reward, move, current_game_state in zip(
                rewards,
                simulation.data.historical_moves,
                game_sequence
        ):
            next_game_state = current_game_state.apply_move(move)
            is_terminal = torch.tensor(next_game_state.is_game_won(), dtype=torch.float32)  # Encoded as tensor

            experience = Experience(
                ExperienceData(
                    state=state_encoder.encode(current_game_state),
                    action=move_encoder.encode(move),
                    reward=reward,
                    next_state=state_encoder.encode(next_game_state),
                    done=is_terminal
                ),
                ExperienceMetadata.from_simulation_metadata(
                    simulation.metadata,
                    generator_name=f"CnnExperienceEncoder-v005",
                    current_player=current_game_state.get_current_player().player_id,
                )
            )
            experiences.append(experience)

        return experiences