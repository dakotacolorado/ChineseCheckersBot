from typing import List

from .vector import Vector
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon


class Printer:

    def __init__(self, print_size: int = 10):
        self.print_size = print_size

    def print_grid(self, grid: List[Vector]):
        regular_coordinates = [
            (
                -v.i - v.j / 2,
                np.sqrt(3) * v.j / 2
            ) for v in grid
        ]

        fig, ax = plt.subplots(1, figsize=(
            self.print_size, self.print_size))
        ax.set_aspect('equal')

        # Add some coloured hexagons
        for coordinate in regular_coordinates:
            hex = RegularPolygon(
                coordinate,
                numVertices=6,
                radius=0.58,
                facecolor="Red",
                alpha=0.3,
                edgecolor='k'
            )
            ax.add_patch(hex)

        # Also add scatter points in hexagon centres
        ax.scatter([
            c[0] for c in regular_coordinates
        ], [c[1] for c in regular_coordinates], alpha=0.1)

        plt.show()