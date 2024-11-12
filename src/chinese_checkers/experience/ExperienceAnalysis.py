from typing import List

import matplotlib.pyplot as plt

from chinese_checkers.experience import ExperienceData


class ExperienceAnalysis:

    experience: List[ExperienceData]

    def __init__(self, experiences: List[ExperienceData]):
        self.experiences: List[ExperienceData] = experiences

    def print_winner_counts(self):
        p0_win_count = len([e for e in self.experiences if e.metadata.winning_player == "0"])
        p3_win_count = len([e for e in self.experiences if e.metadata.winning_player == "3"])
        print(f"p0_win_count {p0_win_count}, p3_win_count {p3_win_count}")


    def check_feature_overlap(self):
        # Extract rewards for each player group
        win_rewards = [e.data.reward.item() for e in self.experiences if e.metadata.winning_player == "0"]
        loss_rewards = [e.data.reward.item() for e in self.experiences if e.metadata.winning_player == "3"]


        plt.figure(figsize=(10, 6))
        plt.hist(win_rewards, bins=60, alpha=0.6, color='blue', edgecolor='black', label="Winning Game: Moves Rewards")
        plt.hist(loss_rewards, bins=60, alpha=0.6, color='green', edgecolor='black', label="Losing Game: Move Rewards")

        plt.title("Overlayed reward distributions for winning and losing move rewards")
        plt.xlabel("Reward")
        plt.ylabel("Frequency")
        plt.legend()
        plt.show()

        # Plot reward distributions
        # plt.figure(figsize=(12, 5))

        # # Player 0 rewards distribution
        # plt.subplot(1, 2, 1)
        # plt.hist(win_rewards, bins=30, alpha=0.7, color='blue', edgecolor='black')
        # plt.title("Reward Distribution for Player 0 (Winning)")
        # plt.xlabel("Reward")
        # plt.ylabel("Frequency")
        #
        # # Player 3 rewards distribution
        # plt.subplot(1, 2, 2)
        # plt.hist(loss_rewards, bins=30, alpha=0.7, color='green', edgecolor='black')
        # plt.title("Reward Distribution for Player 0 (Losing)")
        # plt.xlabel("Reward")
        # plt.ylabel("Frequency")
        #
        # plt.tight_layout()
        # plt.show()