import logging
from typing import List

import torch
from uuid import uuid4

from .CnnEncoderMove import CnnEncoderMove
from chinese_checkers.simulation import GameSimulation
from .CnnEncoderState import CnnEncoderState
from .RewardEncoder import RewardEncoder
from chinese_checkers.experience import Experience, ExperienceData, ExperienceMetadata
from chinese_checkers.game import ChineseCheckersGame


class CnnEncoderExperience:
    logger = logging.getLogger(__name__)

    def __init__(self, encoder_version: str):
        self.encoder_version = encoder_version

        if self.encoder_version == "v006":
            self.logger.info(f"Using CnnExperienceEncoder version: {self.encoder_version}")
            self._encoder = CnnEncoderExperience.encode_v006
        else:
            raise ValueError(f"Unknown encoder version: {self.encoder_version}")

    def encode(
            self,
            simulation: GameSimulation
    ):
        return self._encoder(simulation)

    @staticmethod
    def encode_v006(
            simulation: GameSimulation
    ):
        move_encoder = CnnEncoderMove(simulation.metadata.board_size)
        reward_encoder = RewardEncoder()
        state_encoder = CnnEncoderState(simulation.metadata.board_size)

        game_sequence: List[ChineseCheckersGame] = simulation.to_game_sequence()
        rewards = reward_encoder.encode(simulation, game_sequence)

        experiences = []
        current_game_state_encoding = None
        next_game_state_encoding = None
        for turn, (reward, move, current_game_state) in enumerate(
                zip(
                    rewards,
                    simulation.data.historical_moves,
                    game_sequence
                )
        ):
            current_game_state_encoding = state_encoder.encode(current_game_state) if current_game_state_encoding is None else next_game_state_encoding
            next_game_state = current_game_state.apply_move(move)
            next_game_state_encoding = state_encoder.encode(next_game_state)

            is_terminal = torch.tensor(next_game_state.is_game_won(), dtype=torch.float32)  # Encoded as tensor

            experience = Experience(
                ExperienceData(
                    state=current_game_state_encoding,
                    action=move_encoder.encode(move),
                    reward=reward,
                    next_state=next_game_state_encoding,
                    done=is_terminal,
                    turn=turn,  # Add the turn number
                    game_uuid=str(uuid4())  # Generate a unique UUID for each experience
                ),
                ExperienceMetadata.from_simulation_metadata(
                    simulation.metadata,
                    generator_name=f"CnnExperienceEncoder-v006",
                    current_player=current_game_state.get_current_player().player_id,
                )
            )
            experiences.append(experience)

        return experiences
