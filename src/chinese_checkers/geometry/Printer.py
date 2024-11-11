import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
from matplotlib import colors
from typing import List, Dict
from pydash import flatten

from .Vector import Vector


class Printer:
    """Visualize points as hexagons on a plot."""

    def __init__(
            self,
            plot_size: int = 10,
            show_coordinates: bool = False,
    ):
        self.plot_size = plot_size
        self.show_coordinates = show_coordinates

    def print(
            self,
            *point_groups: List[List[Vector]],
            axes: plt.Axes = None,
            show_plot: bool = True,
            game_metrics: Dict[str, any] = None,
            show_coordinates: bool = False,
    ) -> None:
        """Render hexagons for the primary and any additional point groups and display game metrics."""

        # Set up the axes.
        axes = self._setup_axes(self.plot_size, axes)

        # Regularize the points to hexagon coordinates and map them to colors.
        regularized_point_groups = [
            [self._regularize_vector(point) for point in points_group]
            for points_group in point_groups
        ]
        point_colors = self._get_point_colors(regularized_point_groups)
        all_points = flatten(point_groups)
        regularized_points = flatten(regularized_point_groups)

        # Plot the points.
        self._plot_hexagon_points(axes, regularized_points, point_colors)
        if self.show_coordinates or show_coordinates:
            self._annotate_points(axes, regularized_points, all_points, self.plot_size)

        # Display the game metrics if provided
        if game_metrics:
            self._display_game_metrics(axes, game_metrics)

        if show_plot and not axes:
            plt.show()

    @staticmethod
    def _regularize_vector(vector: Vector) -> Vector:
        """Transform a vector to hexagon coordinates."""
        return Vector(-vector.i - vector.j / 2, np.sqrt(3) * vector.j / 2)

    @staticmethod
    def _get_point_colors(points_list: List[List[Vector]]) -> Dict[Vector, str]:
        """Map points to tableau colors based on list membership."""
        return {
            point: list(colors.TABLEAU_COLORS.keys())[i]
            for i, points in enumerate(points_list)
            for point in points
        }

    @staticmethod
    def _plot_hexagon_points(
            axes: plt.Axes,
            points: List[Vector],
            point_colors: Dict[Vector, str]
    ) -> None:
        """Display points as hexagons on the given axes."""
        for point in points:
            axes.add_patch(
                RegularPolygon(
                    (point.i, point.j),
                    numVertices=6,
                    radius=0.58,
                    facecolor=point_colors[point],
                    alpha=0.3,
                    edgecolor='k'
                )
            )
        # Add invisible points to the axes to mark the hexagon centers.
        axes.scatter(
            [point.i for point in points],
            [point.j for point in points],
            alpha=0
        )

    @staticmethod
    def _annotate_points(
            axes: plt.Axes,
            points: List[Vector],
            point_annotations: List[Vector],
            plot_size: int
    ) -> None:
        """Add coordinate annotations to the points."""
        for point, annotation in zip(points, point_annotations):
            axes.annotate(
                f"({annotation.i}, {annotation.j})",
                (point.i, point.j),
                ha='center',
                va='center',
                fontsize=plot_size / 2 + 1
            )

    @staticmethod
    def _setup_axes(
            plot_size: int,
            provided_axes: plt.Axes = None
    ) -> plt.Axes:
        """Set up and return the axes, either the provided one or a new one."""
        if provided_axes:
            return provided_axes
        else:
            fig, axes = plt.subplots(1, figsize=(plot_size, plot_size))
            axes.set_aspect('equal')
            return axes

    @staticmethod
    def _display_game_metrics(axes: plt.Axes, game_metrics: Dict[str, any]) -> None:
        """Display game metrics on the plot as key-value pairs in a text box."""
        metric_text = "\n".join([f"{key}: {value}" for key, value in game_metrics.items()])

        # Place the text box on the right side of the plot
        axes.text(
            1.05, 0.5, metric_text,
            transform=axes.transAxes,
            verticalalignment='center',
            bbox=dict(boxstyle="round,pad=0.5", edgecolor="black", facecolor="lightgrey"),
            fontsize=10
        )