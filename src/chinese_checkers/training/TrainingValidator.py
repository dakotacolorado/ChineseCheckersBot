import os
import time
from typing import List, Tuple

import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from tqdm import tqdm

from chinese_checkers.game import ChineseCheckersGame
from chinese_checkers.model import TrainableModel, IModel, BootstrapModel
from chinese_checkers.simulation import GameSimulation


class TrainingValidator:

    def __init__(
            self,
            board_size: int,
            player_count: int,
            max_turns: int = 1000,
            benchmark_model: IModel = BootstrapModel(),
            simulation_name: str = "validation_simulation",
            simulation_version: str = "v1.0.0"
    ):
        self.board_size = board_size
        self.player_count = player_count
        self.max_turns = max_turns
        self.benchmark_model = benchmark_model
        self.simulation_name = simulation_name
        self.simulation_version = simulation_version
        self.validation_iteration = 0

    def _simulate_game(self, model_to_validate: TrainableModel, model_to_validate_player_id: int) -> GameSimulation:
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

    def validate(
            self,
            model_to_validate: TrainableModel,
            model_to_validate_player_id: int,
            validation_size: int = 50,
            save_animation: bool = False,
            save_report_location: str = None,
            display_report: bool = True
    ):
        print(f"Validating model with {validation_size} simulations as player {model_to_validate_player_id}...")
        validation_start_time = time.time()

        total_regular_error = 0
        total_discounted_error = 0
        total_turn_count = 0
        player_wins = [0] * self.player_count
        no_winner_count = 0

        player_id_map = {pid: idx for idx, pid in enumerate(ChineseCheckersGame.STARTING_PLAYER_CORNERS[self.player_count])}

        for _ in tqdm(range(validation_size), desc="Validating games"):
            validation_game = self._simulate_game(model_to_validate, model_to_validate_player_id)
            winner_id_raw = validation_game.metadata.winning_player
            winner_id = int(winner_id_raw) if winner_id_raw is not None else None
            validation_results = self._validate_model(model_to_validate, validation_game)

            total_regular_error += validation_results['regular_error']
            total_discounted_error += validation_results['discounted_error']
            total_turn_count += len(validation_game.data.historical_moves)

            if winner_id is None:
                no_winner_count += 1
            elif winner_id in player_id_map:
                player_wins[player_id_map[winner_id]] += 1

        average_regular_error = total_regular_error / validation_size
        average_discounted_error = total_discounted_error / validation_size
        average_turn_count = total_turn_count / validation_size

        validation_duration = time.time() - validation_start_time
        print(f"Validation completed in {validation_duration:.2f} seconds with averaged results: "
              f"regular_error={average_regular_error:.6f}, discounted_error={average_discounted_error:.6f}, "
              f"average_turn_count={average_turn_count:.2f}")

        games_won = player_wins[player_id_map[model_to_validate_player_id]]
        percent_games_won = games_won / validation_size * 100
        percent_no_winner = no_winner_count / validation_size * 100

        validation_data = {
            "validation_timestamp": time.time(),
            "validation_iteration": self.validation_iteration,
            "player_id": model_to_validate_player_id,
            "validation_size": validation_size,
            "average_regular_error": average_regular_error,
            "average_discounted_error": average_discounted_error,
            "average_turn_count": average_turn_count,
            "number_of_games_won": games_won,
            "percent_games_won": percent_games_won,
            "number_of_games_with_no_winner": no_winner_count,
            "percent_games_with_no_winner": percent_no_winner,
        }

        for player_id, mapped_id in player_id_map.items():
            validation_data[f"player_{player_id}_win_count"] = player_wins[mapped_id]
            validation_data[f"player_{player_id}_win_percent"] = player_wins[mapped_id] / validation_size * 100

        validation_df = pd.DataFrame([validation_data])
        if save_report_location:
            validation_df = self._save_validation_results(save_report_location, validation_df)

        if display_report and save_report_location:
            self._display_report(save_report_location)

        if save_animation:
            validation_game.save_animation(f"{self.simulation_name}_{self.simulation_version}_validation_{self.validation_iteration}.mp4")

        self.validation_iteration += 1

    @staticmethod
    def _save_validation_results(save_report_location: str, validation_df: pd.DataFrame) -> pd.DataFrame:
        if os.path.exists(save_report_location):
            existing_df = pd.read_csv(save_report_location)
            validation_df = pd.concat([existing_df, validation_df], ignore_index=True)

        validation_df = validation_df.sort_values(by="validation_timestamp", ascending=False)
        validation_df.to_csv(save_report_location, index=False)
        return validation_df

    def _display_report(self, save_report_location: str):
        validation_df = pd.read_csv(save_report_location)

        # Plot 1: Player Win Percentages and No Winner Percentage
        plt.figure(figsize=(12, 8))
        for player_id in ChineseCheckersGame.STARTING_PLAYER_CORNERS[self.player_count]:
            plt.plot(
                validation_df["validation_timestamp"],
                validation_df[f"player_{player_id}_win_percent"],
                label=f"Player {player_id} Win %"
            )
        plt.plot(
            validation_df["validation_timestamp"],
            validation_df["percent_games_with_no_winner"],
            label="No Winner %",
            linestyle="--"
        )
        plt.title("Player Win Percentages and No Winner Percentage Over Time")
        plt.xlabel("Validation Timestamp")
        plt.ylabel("Percentage")
        plt.legend()
        plt.grid(True)

        # Plot 2: Regular Error Over Time
        plt.figure(figsize=(12, 8))
        plt.plot(validation_df["validation_timestamp"], validation_df["average_regular_error"], label="Regular Error",
                 color="blue")
        plt.title("Regular Error Over Time")
        plt.xlabel("Validation Timestamp")
        plt.ylabel("Regular Error")
        plt.legend()
        plt.grid(True)

        # Plot 3: Discounted Error Over Time
        plt.figure(figsize=(12, 8))
        plt.plot(validation_df["validation_timestamp"], validation_df["average_discounted_error"],
                 label="Discounted Error", color="green")
        plt.title("Discounted Error Over Time")
        plt.xlabel("Validation Timestamp")
        plt.ylabel("Discounted Error")
        plt.legend()
        plt.grid(True)

        # Plot 4: Average Turn Count Over Time
        plt.figure(figsize=(12, 8))
        plt.plot(validation_df["validation_timestamp"], validation_df["average_turn_count"],
                 label="Average Turn Count", color="purple")
        plt.title("Average Turn Count Over Time")
        plt.xlabel("Validation Timestamp")
        plt.ylabel("Average Turn Count")
        plt.legend()
        plt.grid(True)

        plt.show()

        # Display the top of the DataFrame for quick inspection
        from IPython.display import display
        display(validation_df.head())

    @staticmethod
    def _validate_model(model_to_train: TrainableModel, validation_game: GameSimulation) -> dict:
        game_sequence = validation_game.to_game_sequence()
        moves = validation_game.data.historical_moves
        rewards = model_to_train.encode_reward(validation_game)

        states = [model_to_train.encode_game(game) for game in game_sequence]
        actions = [model_to_train.encode_move(move) for move in moves]

        experiences = list(zip(states, actions, rewards))

        regular_error = TrainingValidator._compute_regular_error(model_to_train, experiences)
        discounted_error = TrainingValidator._compute_discounted_error(model_to_train, experiences)

        return {
            "regular_error": regular_error,
            "discounted_error": discounted_error,
        }

    @staticmethod
    def _compute_regular_error(model_to_train: TrainableModel, experiences: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> float:
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

        return total_error / len(experiences) if experiences else 0.0

    @staticmethod
    def _compute_discounted_error(model_to_train: TrainableModel, experiences: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> float:
        total_error = 0.0
        loss_fn = nn.MSELoss()
        discounted_return = torch.tensor(0.0, device=model_to_train.device)
        gamma = getattr(model_to_train, 'gamma', 0.99)

        for state_tensor, action_tensor, reward_tensor in reversed(experiences):
            reward_tensor = reward_tensor.to(model_to_train.device)
            discounted_return = reward_tensor + gamma * discounted_return

            state_tensor = state_tensor.unsqueeze(0).to(model_to_train.device)
            action_tensor = action_tensor.unsqueeze(0).to(model_to_train.device)

            with torch.no_grad():
                predicted_q_value = model_to_train.network(state_tensor, action_tensor).squeeze()

            total_error += loss_fn(predicted_q_value, discounted_return).item()

        return total_error / len(experiences) if experiences else 0.0
