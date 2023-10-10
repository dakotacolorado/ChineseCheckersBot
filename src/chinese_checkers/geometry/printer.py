from typing import List

from pydash import map_

from .vector import Vector
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon


class Printer:

    def __init__(self, print_size: int = 10):
        self.print_size = print_size

    def print_grid(
            self,
            grid_points: List[Vector],
            green_points: List[Vector] = [],
            red_points: List[Vector] = [],
            yellow_points: List[Vector] = []
    ):

        regularization = lambda v: (
            -v.i - v.j / 2,
            np.sqrt(3) * v.j / 2
        )

        regular_grid_points = map_(grid_points, regularization)
        regular_green_points = map_(green_points, regularization)
        regular_red_points = map_(red_points, regularization)
        regular_yellow_points = map_(yellow_points, regularization)

        fig, ax = plt.subplots(1, figsize=(self.print_size, self.print_size))
        ax.set_aspect('equal')

        def __get_color(point):
            if point in regular_green_points:
                return "Green"
            elif point in regular_red_points:
                return "Red"
            elif point in regular_yellow_points:
                return "Yellow"
            else:
                return "Blue"

        for c in regular_grid_points:
            ax.add_patch(
                RegularPolygon(
                    c,
                    numVertices=6,
                    radius=0.58,
                    facecolor=__get_color(c),
                    alpha=0.3,
                    edgecolor='k'
                )
            )

        ax.scatter(
            [c[0] for c in regular_grid_points],
            [c[1] for c in regular_grid_points],
            alpha=0.1
        )
        plt.show()
