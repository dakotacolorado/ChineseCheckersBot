from dataclasses import dataclass
from typing import List
import torch

from .ExperienceData import ExperienceData
from .ExperienceMetadata import ExperienceMetadata
from ..encode import SpatialMoveMetricsEncoder, DistanceToWinRewardSimulationEncoder, SpatialBoardMetricsEncoder
from ...catalog import IDataMetadata
from ...game import ChineseCheckersGame
from ...simulation import GameSimulation, SimulationMetadata


@dataclass(frozen=True)
class Experience(IDataMetadata[ExperienceData, ExperienceMetadata]):

    data: ExperienceData
    metadata: ExperienceMetadata

    @staticmethod
    def from_data_metadata(data: ExperienceData, metadata: ExperienceMetadata) -> 'Experience':
        return Experience(data, metadata)

    @staticmethod
    def generate_experiences_from_simulation(
            simulation: GameSimulation,
            generator_name: str
    ) -> List["Experience"]:

        if generator_name == "v0.0.1":
            return Experience._generate_experiences_v0_0_1(simulation, generator_name)
        else:
            raise ValueError(f"Unsupported encoder name '{generator_name}'")

    @staticmethod
    def _generate_experiences_v0_0_1(
            simulation: GameSimulation,
            generator_name: str
    ) -> List["Experience"]:
        move_encoder = SpatialMoveMetricsEncoder()
        reward_encoder = DistanceToWinRewardSimulationEncoder()
        board_state_encoder = SpatialBoardMetricsEncoder(simulation.metadata.board_size)

        rewards = reward_encoder.encode(simulation)
        game_sequence: List[ChineseCheckersGame] = simulation._to_game_sequence()

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
                    state=board_state_encoder.encode(current_game_state),
                    action=move_encoder.encode(move, current_game_state),
                    reward=reward,
                    next_state=board_state_encoder.encode(next_game_state),
                    done=is_terminal
                ),
                ExperienceMetadata.from_simulation_metadata(
                    simulation.metadata,
                    generator_name=generator_name,
                    current_player= current_game_state.get_current_player().player_id,
                )
            )
            experiences.append(experience)

        return experiences