from typing import List

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from ..game.ChineseCheckersGame import ChineseCheckersGame
from ..simulation.GameSimulationData import GameSimulationData


class GameSimulationAnimation:

    def __init__(self, game_sequence: List[ChineseCheckersGame]):
        self.game_sequence = game_sequence
        self.fig, self.ax = plt.subplots(1, figsize=(10, 10))
        self.anim = self._create_animation()

    def _create_animation(self):
        return animation.FuncAnimation(
            self.fig,
            self._update_plot,
            frames=len(self.game_sequence),
            repeat=False
        )

    @staticmethod
    def from_simulation_data(game_simulation: GameSimulationData) -> "GameSimulationAnimation":
        game_sequence = GameSimulationData.to_game_sequence(game_simulation)
        return GameSimulationAnimation(game_sequence)

    def _update_plot(self, i: int):
        game = self.game_sequence[i]
        self.ax.clear()
        game.print()
        return self.ax,

    def display(self):
        plt.show()

    def save_to_file(self, file_path: str):
        self.anim.save(file_path, writer="ffmpeg")
