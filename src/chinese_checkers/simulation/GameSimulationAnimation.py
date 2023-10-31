import logging

from tqdm import tqdm
from IPython.core.display import HTML

from .GameSimulation import GameSimulation
from ..game.ChineseCheckersGame import ChineseCheckersGame

from typing import List
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers


class GameSimulationAnimation:

    @staticmethod
    def from_simulation_data(game_simulation: GameSimulation) -> "GameSimulationAnimation":
        game_sequence = game_simulation.to_game_sequence()
        return GameSimulationAnimation(game_sequence)

    def __init__(
            self,
            game_sequence: List[ChineseCheckersGame],
            plot_size: int = 100,
            dpi: int = 5
    ):
        self.game_sequence = game_sequence
        self.fig, self.ax = plt.subplots(1, figsize=(plot_size, plot_size), dpi=dpi)
        plt.close(self.fig)
        self.pbar = tqdm(total=len(self.game_sequence), desc="Creating Animation")
        self.anim: FuncAnimation = self._create_animation()

    def _create_animation(self) -> FuncAnimation:
        animation = FuncAnimation(
            self.fig,
            self._update_plot,
            frames=len(self.game_sequence),
            repeat=False
        )
        self.pbar.close()
        return animation

    def _update_plot(self, i: int):
        game = self.game_sequence[i]
        self.ax.clear()
        game.print(axes=self.ax, show_plot=False)
        self.pbar.update(1)
        return self.ax,

    def display(self) -> HTML:
        return HTML(self.anim.to_jshtml())

    def save_to_file(self, file_path: str, fps: int = 30):
        writer = writers['ffmpeg'](fps=fps, bitrate=1800)
        self.anim.save(file_path, writer=writer)
        logging.info(f"Saved animation to {file_path}")
