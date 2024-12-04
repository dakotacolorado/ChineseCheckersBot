import os
import time
from collections import deque
from typing import List, Tuple, Dict

import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from tqdm import tqdm

from chinese_checkers.game import ChineseCheckersGame
from chinese_checkers.model import IModel, BootstrapModel
from chinese_checkers.reinforcement import DeepQModel
from chinese_checkers.reinforcement.Encoder import Encoder
from chinese_checkers.reinforcement.Experience import Experience
from chinese_checkers.reinforcement.ReplayBuffer import ReplayBuffer
from chinese_checkers.simulation import GameSimulation


class Validator:

    def __init__(
            self,
            board_size: int,
            player_count: int,
            replay_buffer: ReplayBuffer,
            encoder: Encoder,
            max_turns: int = 1000,
            benchmark_model: IModel = BootstrapModel(),
            baseline_model: IModel = BootstrapModel(),
            simulation_name: str = "validation_simulation",
            simulation_version: str = "v1.0.0",
            gamma: float = 0.99,
            base_validation_size: int = 100,
            training_sim_max: int = 50
    ):
        self.board_size = board_size
        self.player_count = player_count
        self.replay_buffer = replay_buffer
        self.encoder = encoder
        self.max_turns = max_turns
        self.baseline_model = baseline_model
        self.benchmark_model = benchmark_model
        self.simulation_name = simulation_name
        self.simulation_version = simulation_version
        self.validation_iteration = 0
        self.gamma = gamma
        self.baseline_v_sims = [
            GameSimulation.simulate_game(
                models=[self.baseline_model for _ in range(player_count)],
                name=self.simulation_name,
                version=self.simulation_version,
                max_turns=self.max_turns,
                board_size=self.board_size,
            ) for _ in range(base_validation_size)
        ]
        self.encoded_baseline_sims = list(self.encode_simulations(self.encoder, self.baseline_v_sims))
        self.encoded_training_sims = deque(maxlen=training_sim_max)

    def _simulate_validation_game(self, model_to_validate: DeepQModel, model_to_validate_player_id: int) -> GameSimulation:
        """
        Simulates a game between the model to validate and the opponent.
        """
        player_ids = ChineseCheckersGame.STARTING_PLAYER_CORNERS[self.player_count]
        return GameSimulation.simulate_game(
            models=[model_to_validate if i == model_to_validate_player_id else self.benchmark_model for i in player_ids],
            name=self.simulation_name,
            version=self.simulation_version,
            max_turns=self.max_turns,
            board_size=self.board_size,
        )


    @staticmethod
    def encode_simulations(encoder: Encoder, simulations: List[GameSimulation]) -> List[List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
        for simulation in simulations:
            game_sequence = simulation.to_game_sequence()
            moves = simulation.data.historical_moves
            rewards = encoder.encode_reward(simulation)

            states = [encoder.encode_game(game) for game in game_sequence]
            actions = [encoder.encode_move(move) for move in moves]

            experiences = list(zip(states, actions, rewards))
            yield experiences


    def validate(
            self,
            model_to_validate: DeepQModel,
            model_to_validate_player_id: int,
            training_simulations: List[GameSimulation],
            generation: int,
            new_generation: bool,
            validation_size: int = 50,
            save_animation: bool = False,
            save_report_location: str = None,
            display_report: bool = True,
    ):
        print("Starting validation...")
        self.encoded_training_sims.extend(self.encode_simulations(model_to_validate.encoder, training_simulations))

        exp_v_sims = [
            self._simulate_validation_game(model_to_validate, model_to_validate_player_id)
            for _ in tqdm(range(validation_size), desc="Running validation games")
        ]

        # Calculate Player Wins for Explored Simulations
        exp_v_sim_wins = self._calc_player_wins(exp_v_sims)

        # Calculate Average Turns for Explored Simulations
        exp_v_sim_turns = sum(len(sim.data.historical_moves) for sim in exp_v_sims) / len(exp_v_sims)

        # Calculate Errors for Explored Simulations
        exp_reg_err, exp_dis_err = self._calc_error_summary(model_to_validate, self.encode_simulations(model_to_validate.encoder, exp_v_sims))

        # Calculate Baseline Errors
        baseline_reg_err, baseline_dis_err = self._calc_error_summary(model_to_validate, self.encoded_baseline_sims)

        # Calculate Training Errors
        train_reg_err, train_dis_err = self._calc_error_summary(model_to_validate, list(self.encoded_training_sims))

        validation_data = {
            "generation": generation,
            "new_generation": new_generation,
            "training_samples_seen": model_to_validate.training_samples_seen,
            "validation_iteration": self.validation_iteration,
            "player_id": model_to_validate_player_id,
            "validation_size": validation_size,
            "baseline_avg_regular_error": baseline_reg_err,
            "baseline_avg_discounted_error": baseline_dis_err,
            "explored_avg_regular_error": exp_reg_err,
            "explored_avg_discounted_error": exp_dis_err,
            "training_avg_regular_error": train_reg_err,
            "training_avg_discounted_error": train_dis_err,
            "average_turn_count": exp_v_sim_turns,
            "percent_games_with_no_winner": exp_v_sim_wins[-1],
            "player_0_win_percent": exp_v_sim_wins.get(0, 0),
            "player_1_win_percent": exp_v_sim_wins.get(1, 0),
            "player_2_win_percent": exp_v_sim_wins.get(2, 0),
            "player_3_win_percent": exp_v_sim_wins.get(3, 0),
            "player_4_win_percent": exp_v_sim_wins.get(4, 0),
            "player_5_win_percent": exp_v_sim_wins.get(5, 0)
        }

        validation_df = pd.DataFrame([validation_data])
        if save_report_location:
            validation_df = self._save_validation_results(save_report_location, validation_df)

        if display_report and save_report_location:
            self._display_report(save_report_location)

        if save_animation:
            exp_v_sims[-1].save_animation(
                f"{self.simulation_name}_{self.simulation_version}_validation_{self.validation_iteration}.mp4")

        self.validation_iteration += 1

    def _calc_player_wins(self, simulations: List[GameSimulation]) -> Dict[int | str, float]:
        """ Calculate the percent wins for each player """
        player_ids = ChineseCheckersGame.STARTING_PLAYER_CORNERS[self.player_count]
        player_win_map = {player_id: 0 for player_id in player_ids}
        player_win_map[-1] = 0  # no player wins

        for sim in simulations:
            winner_id = sim.metadata.winning_player
            if winner_id is None:
                player_win_map[-1] += 1
            elif int(winner_id) in player_ids:
                player_win_map[int(winner_id)] += 1
        return {
            i: val / len(simulations) * 100
            for i, val in player_win_map.items()
        }

    def _calc_error_summary(
            self,
            model_to_train: DeepQModel,
            simulations: List[List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]
    ) -> Tuple[float, float]:
        total_regular_error = 0
        total_regular_count = 0
        total_discounted_error = 0
        total_discounted_count = 0

        for sim in simulations:
            update_regular_error, update_regular_count = self._calc_regular_error(model_to_train, sim)
            update_discounted_error, update_discounted_count = self._calc_discounted_error(model_to_train, sim)
            total_regular_error += update_regular_error
            total_regular_count += update_regular_count
            total_discounted_error += update_discounted_error
            total_discounted_count += update_discounted_count

        avg_regular_error = total_regular_error / total_regular_count if total_regular_count else 0
        avg_discounted_error = total_discounted_error / total_discounted_count if total_discounted_count else 0

        return avg_regular_error, avg_discounted_error

    def _calc_discounted_error(
            self,
            model_to_train: DeepQModel,
            experiences: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
        ) -> Tuple[float, int]:
        """
        ENSURE THE EXPERIENCES ARE IN ORDER
        Returns the total discounted error and total number of experiences.
        """
        total_error = 0.0
        loss_fn = nn.MSELoss()
        discounted_return = torch.tensor(0.0, device=model_to_train.device)
        gamma = getattr(model_to_train, 'gamma', self.gamma)

        for state_tensor, action_tensor, reward_tensor in reversed(experiences):
            reward_tensor = reward_tensor.to(model_to_train.device)
            discounted_return = reward_tensor + gamma * discounted_return

            state_tensor = state_tensor.unsqueeze(0).to(model_to_train.device)
            action_tensor = action_tensor.unsqueeze(0).to(model_to_train.device)

            with torch.no_grad():
                predicted_q_value = model_to_train.network(state_tensor, action_tensor).squeeze()

            total_error += loss_fn(predicted_q_value, discounted_return).item()

        return total_error, len(experiences)

    def _calc_regular_error(
            self,
            model_to_train: DeepQModel,
            experiences: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    ) -> Tuple[float, int]:
        """
        Returns the total regular error and total number of experiences.
        """
        total_error = 0.0
        loss_fn = nn.MSELoss()

        for state_tensor, action_tensor, reward_tensor in experiences:
            state_tensor = state_tensor.unsqueeze(0).to(model_to_train.device)
            action_tensor = action_tensor.unsqueeze(0).to(model_to_train.device)
            reward_tensor = reward_tensor.to(model_to_train.device)

            with torch.no_grad():
                predicted_q_value = model_to_train.network(state_tensor, action_tensor).squeeze()

            target_q_value = reward_tensor.squeeze()
            total_error += loss_fn(predicted_q_value, target_q_value).item()

        return total_error, len(experiences)

    @staticmethod
    def _save_validation_results(save_report_location: str, validation_df: pd.DataFrame) -> pd.DataFrame:
        if os.path.exists(save_report_location):
            existing_df = pd.read_csv(save_report_location)
            validation_df = pd.concat([existing_df, validation_df], ignore_index=True)

        validation_df = validation_df.sort_values(by="generation", ascending=False)
        validation_df.to_csv(save_report_location, index=False)
        return validation_df

    def _add_vertical_lines(self, ax, validation_df, column_name="new_generation", x_column="generation", line_color="grey",
                            line_style="--", alpha=0.7):
        """
        Add vertical lines to the plot for each point where a specific condition is True.
        """
        for x_val in validation_df.loc[validation_df[column_name] == True, x_column]:
            ax.axvline(x=x_val, color=line_color, linestyle=line_style, alpha=alpha)

    def _display_report(self, save_report_location: str):
        validation_df = pd.read_csv(save_report_location)

        # Create a figure and subplots
        fig, axs = plt.subplots(4, 1, figsize=(12, 24), sharex=True)

        # Plot 1: Player Win Percentages and No Winner Percentage
        for player_id in ChineseCheckersGame.STARTING_PLAYER_CORNERS[self.player_count]:
            axs[0].plot(
                validation_df["generation"],
                validation_df[f"player_{player_id}_win_percent"],
                label=f"Player {player_id} Win %"
            )
        axs[0].plot(
            validation_df["generation"],
            validation_df["percent_games_with_no_winner"],
            label="No Winner %",
            linestyle="--"
        )
        axs[0].set_title("Player Win Percentages and No Winner Percentage")
        axs[0].set_ylabel("Percentage")
        axs[0].legend()
        axs[0].grid(True)
        self._add_vertical_lines(axs[0], validation_df)

        # Plot 2: Regular Error
        axs[1].plot(
            validation_df["generation"],
            validation_df["explored_avg_regular_error"],
            label="Explored Moves Error (Regular)",
            color="blue"
        )
        axs[1].plot(
            validation_df["generation"],
            validation_df["baseline_avg_regular_error"],
            label="Baseline Validation Error (Regular)",
            color="red",
            linestyle="--"
        )
        axs[1].plot(
            validation_df["generation"],
            validation_df["training_avg_regular_error"],
            label="Training Error (Regular)",
            color="purple",
            linestyle=":"
        )
        axs[1].set_title("Regular Error")
        axs[1].set_ylabel("Error")
        axs[1].legend()
        axs[1].grid(True)
        self._add_vertical_lines(axs[1], validation_df)

        # Plot 3: Discounted Error
        axs[2].plot(
            validation_df["generation"],
            validation_df["explored_avg_discounted_error"],
            label="Explored Moves Error (Discounted)",
            color="green"
        )
        axs[2].plot(
            validation_df["generation"],
            validation_df["baseline_avg_discounted_error"],
            label="Baseline Validation Error (Discounted)",
            color="orange",
            linestyle="--"
        )
        axs[2].plot(
            validation_df["generation"],
            validation_df["training_avg_discounted_error"],
            label="Training Error (Discounted)",
            color="purple",
            linestyle=":"
        )
        axs[2].set_title("Discounted Error")
        axs[2].set_ylabel("Error")
        axs[2].legend()
        axs[2].grid(True)
        self._add_vertical_lines(axs[2], validation_df)

        # Plot 4: Average Turn Count
        axs[3].plot(
            validation_df["generation"],
            validation_df["average_turn_count"],
            label="Average Turn Count",
            color="purple"
        )
        axs[3].set_title("Average Turn Count")
        axs[3].set_ylabel("Turns")
        axs[3].legend()
        axs[3].grid(True)
        self._add_vertical_lines(axs[3], validation_df)

        # Adjust layout and show
        plt.tight_layout()
        save_fig_location = save_report_location.replace(".csv", "") + ".png"
        plt.savefig(save_fig_location, dpi=300)
        plt.show()

        # Display the top of the DataFrame for quick inspection
        from IPython.display import display
        display(validation_df.head())

