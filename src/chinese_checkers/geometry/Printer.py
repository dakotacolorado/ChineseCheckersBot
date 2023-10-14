import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
from typing import List
from pydash import map_

from .Vector import Vector


class Printer:

    def __init__(self, print_size: int = 10, print_coordinates: bool = False):
        self.print_size = print_size
        self.print_coordinates = print_coordinates

    @staticmethod
    def _regularize_vector(vector: Vector) -> tuple:
        return -vector.i - vector.j / 2, np.sqrt(3) * vector.j / 2

    @staticmethod
    def _get_color(point, regular_colored_points) -> str:
        if point in regular_colored_points["green"]:
            return "Green"
        elif point in regular_colored_points["red"]:
            return "Red"
        elif point in regular_colored_points["yellow"]:
            return "Yellow"
        return "Blue"

    def _plot_points(self, ax, regular_grid_points, regular_colored_points):
        for point in regular_grid_points:
            ax.add_patch(
                RegularPolygon(
                    point,
                    numVertices=6,
                    radius=0.58,
                    facecolor=self._get_color(point, regular_colored_points),
                    alpha=0.3,
                    edgecolor='k'
                )
            )
        ax.scatter([point[0] for point in regular_grid_points],
                   [point[1] for point in regular_grid_points],
                   alpha=0.1)

    def _annotate_points(self, ax, grid_points, regular_grid_points):
        for original, regular in zip(grid_points, regular_grid_points):
            ax.annotate(
                f"({original.i}, {original.j})",
                (regular[0], regular[1]),
                ha='center',
                va='center',
                fontsize=self.print_size / 2 + 1
            )

    def print_grid(self, grid_points: List[Vector],
                   green_points: List[Vector] = None,
                   red_points: List[Vector] = None,
                   yellow_points: List[Vector] = None):
        if green_points is None:
            green_points = []
        if red_points is None:
            red_points = []
        if yellow_points is None:
            yellow_points = []

        regular_colored_points = {
            "green": map_(green_points, self._regularize_vector),
            "red": map_(red_points, self._regularize_vector),
            "yellow": map_(yellow_points, self._regularize_vector)
        }

        regular_grid_points = map_(grid_points, self._regularize_vector)

        fig, ax = plt.subplots(1, figsize=(self.print_size, self.print_size))
        ax.set_aspect('equal')

        self._plot_points(ax, regular_grid_points, regular_colored_points)

        if self.print_coordinates:
            self._annotate_points(ax, grid_points, regular_grid_points)

        plt.show()
