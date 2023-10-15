import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
from matplotlib import colors
from typing import List, Dict
from pydash import flatten

from .Vector import Vector


class Printer:

    def __init__(self, print_size: int = 10, print_coordinates: bool = False):
        self.print_size = print_size
        self.print_coordinates = print_coordinates

    @staticmethod
    def _regularize_vector(vector: Vector) -> Vector:
        """Convert a vector to a regular hexagon coordinate system."""
        return Vector(-vector.i - vector.j / 2, np.sqrt(3) * vector.j / 2)

    @staticmethod
    def _get_point_colors(points_list: List[List[Vector]]) -> Dict[Vector, str]:
        """Assign colors to points based on their list."""
        return {
            point: list(colors.TABLEAU_COLORS.keys())[i]
            for i, points in enumerate(points_list)
            for point in points
        }

    @staticmethod
    def _plot_hexagon_points(ax: plt.Axes, points: List[Vector], point_colors: Dict[Vector, str]) -> None:
        """Plot the hexagons on the provided axes."""
        for point in points:
            ax.add_patch(
                RegularPolygon(
                    (point.i, point.j),
                    numVertices=6,
                    radius=0.58,
                    facecolor=point_colors[point],
                    alpha=0.3,
                    edgecolor='k'
                )
            )
        ax.scatter([point.i for point in points],
                   [point.j for point in points],
                   alpha=0)  # make this > 0 to see the points

    def _annotate_points(self, ax: plt.Axes, points: List[Vector], point_annotations: List[Vector]) -> None:
        """Annotate the hexagons with coordinates."""
        for point, annotation in zip(points, point_annotations):
            ax.annotate(
                f"({annotation.i}, {annotation.j})",
                (point.i, point.j),
                ha='center',
                va='center',
                fontsize=self.print_size / 2 + 1
            )

    def print_grid(self, point_group: List[Vector], *additional_point_groups: List[List[Vector]]) -> None:
        """Render the grid of hexagons. Colored by group."""
        all_point_groups = [point_group] + list(additional_point_groups)

        regularized_point_groups = [
            [
                self._regularize_vector(point)
                for point in points_group
            ]
            for points_group in all_point_groups
        ]

        point_colors = self._get_point_colors(regularized_point_groups)

        all_points = flatten(all_point_groups)
        regularized_points = flatten(regularized_point_groups)

        fig, ax = plt.subplots(1, figsize=(self.print_size, self.print_size))
        ax.set_aspect('equal')

        self._plot_hexagon_points(ax, regularized_points, point_colors)

        if self.print_coordinates:
            self._annotate_points(ax, regularized_points, all_points)

        plt.show()
