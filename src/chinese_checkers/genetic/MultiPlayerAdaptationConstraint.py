from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from chinese_checkers.game import ChineseCheckersGame
from chinese_checkers.model import IModel
from chinese_checkers.simulation import GameSimulation


class MultiPlayerAdaptationConstraint:
    def __init__(self, validation_opponent: IModel, validation_game_count: int, max_turns: int, board_size: int):
        self.validation_game_count = validation_game_count
        self.validation_opponent = validation_opponent
        self.max_turns = max_turns
        self.board_size = board_size

    def check(self, model_to_validate: IModel):
        # Colors for players by ID
        player_colors = {
            0: "#0072B2",  # Blue
            1: "#D55E00",  # Vermillion
            2: "#F0E442",  # Yellow
            3: "#009E73",  # Green
            4: "#CC79A7",  # Pink
            5: "#56B4E9",  # Sky Blue
            -1: "#999999"  # Grey for no winner
        }

        # Prepare data for plotting
        bar_width = 0.8
        player_counts = list(ChineseCheckersGame.STARTING_PLAYER_CORNERS.keys())
        results = []
        turn_counts = []

        for i in player_counts:
            simulations = []
            print(f"Simulating {i}-Player Games:")
            for _ in tqdm(range(self.validation_game_count), desc=f"{i}-Player Games"):
                simulation = GameSimulation.simulate_game(
                    models=[model_to_validate] + [self.validation_opponent for _ in range(i - 1)],
                    name=f"validation_{i}",
                    version="v0.0.1",
                    max_turns=self.max_turns * i,
                    board_size=self.board_size,
                )
                simulations.append(simulation)
            player_wins = self._calc_player_wins(simulations, i)

            results.append(player_wins)
            print(f"Player Wins ({i} players): {player_wins}")
            turn_count = sum([len(sim.data.historical_moves) for sim in simulations])/len(simulations)
            print(f"Average turns ({i} players): {turn_count}")
            turn_counts.append(turn_count)
            for player_id, win_percent in player_wins.items():
                print(f"Player {player_id}: {win_percent:.2f}% wins")
                winning_sims = [sim for sim in simulations if sim.metadata.winning_player == str(player_id)]
                for k in range(min([3, len(winning_sims)])):
                    winning_sims[k].save_animation(f"{i}_player_game_winner_{player_id}_{k}.mp4")

        # Plotting
        x_positions = np.arange(len(player_counts))
        fig, ax = plt.subplots(figsize=(12, 8))

        cumulative_bottom = np.zeros(len(player_counts))
        for player_id, color in player_colors.items():
            percentages = [results[idx].get(player_id, 0) for idx in range(len(player_counts))]
            ax.bar(
                x_positions,
                percentages,
                bar_width,
                label=f"Player {player_id}" if player_id != -1 else "No Winner",
                color=color,
                bottom=cumulative_bottom,
                align="center"
            )
            cumulative_bottom += percentages  # Update bottom for stacking

        # Beautify the chart
        ax.set_xticks(x_positions)
        ax.set_xticklabels([f"{i}-Player Games" for i in player_counts])
        ax.set_ylabel("Percentage")
        ax.set_title("Win vs Loss Composition by Player Count")
        ax.legend(title="Players")
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.show()

    def _calc_player_wins(self, simulations: List[GameSimulation], player_count: int) -> Dict[int, float]:
        """ Calculate the percent wins for each player """
        player_ids = ChineseCheckersGame.STARTING_PLAYER_CORNERS[player_count]
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
