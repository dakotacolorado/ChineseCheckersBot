import logging
import os
import tempfile
from typing import List

import matplotlib.pyplot as plt
from IPython.display import Image
from matplotlib.animation import FuncAnimation, writers
from tqdm import tqdm

from . import GameSimulation
from ..game.ChineseCheckersGame import ChineseCheckersGame


class GameSimulationAnimation:

    @classmethod
    def from_simulation_data(cls, simulation_data: GameSimulation, plot_size: int = 100, dpi: int = 5):
        game_sequence = simulation_data.to_game_sequence()
        return cls(game_sequence, plot_size, dpi)

    def __init__(
            self,
            game_sequence: List[ChineseCheckersGame],
            plot_size: int = 100,
            dpi: int = 5
    ):
        """Initializes the GameSimulationAnimation with a sequence of game states.

        Args:
            game_sequence (List[ChineseCheckersGame]): List of game states to animate.
            plot_size (int): Size of the plot.
            dpi (int): Dots per inch for figure resolution.
        """
        if not game_sequence:
            raise ValueError("Game sequence cannot be empty.")
        self.game_sequence = game_sequence
        self.fig, self.ax = plt.subplots(1, figsize=(plot_size, plot_size), dpi=dpi)
        plt.close(self.fig)
        self.pbar = tqdm(total=len(self.game_sequence), desc="Creating Animation")
        self.anim: FuncAnimation = self._create_animation()

        # Set a hidden temp GIF file path for display
        self.temp_gif_path = os.path.join(tempfile.gettempdir(), ".temp_animation.gif")

    def _create_animation(self) -> FuncAnimation:
        """Creates the animation by iterating over game states."""
        animation = FuncAnimation(
            self.fig,
            self._update_plot,
            frames=len(self.game_sequence),
            repeat=False
        )
        self.pbar.close()
        return animation

    def _update_plot(self, i: int):
        """Updates the plot for each frame in the animation.

        Args:
            i (int): The current frame index.
        """
        game = self.game_sequence[i]
        self.ax.clear()
        game.print(axes=self.ax, show_plot=False)  # No plt.show() during animation
        self.fig.canvas.draw()  # Ensures the canvas updates for each frame
        self.pbar.update(1)
        return self.ax,

    def display(self, fps: int = 10):
        """Displays the animation as a GIF in Jupyter Notebook or JupyterLab."""
        writer = writers['pillow'](fps=fps)
        self.anim.save(self.temp_gif_path, writer=writer)
        logging.info(f"Temporary GIF saved to {self.temp_gif_path}")
        return Image(filename=self.temp_gif_path)

    def save_to_file(self, file_path: str, fps: int = 30):
        """Saves the animation to a video file.

        Args:
            file_path (str): Path where the video file will be saved.
            fps (int): Frames per second for the saved video.
        """
        writer = writers['ffmpeg'](fps=fps, bitrate=1800)
        self.anim.save(file_path, writer=writer)
        logging.info(f"Saved animation to {file_path}")
