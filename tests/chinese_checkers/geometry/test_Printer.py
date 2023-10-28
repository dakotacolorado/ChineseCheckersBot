from unittest import TestCase
from unittest.mock import patch, Mock

import mock as mock
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import RegularPolygon
from matplotlib import colors

from src.chinese_checkers.geometry.Printer import Printer
from src.chinese_checkers.geometry.Vector import Vector


class TestPrinter(TestCase):

    @mock.patch.object(Printer, "_setup_axes")
    @mock.patch.object(Printer, "_regularize_vector")
    @mock.patch.object(Printer, "_get_point_colors")
    @mock.patch.object(Printer, "_plot_hexagon_points")
    @mock.patch.object(Printer, "_annotate_points")
    def test_print(self, mock_annotate, mock_plot_hexagons, mock_get_colors, mock_regularize, mock_setup_axes):

        point_groups = [[Vector(1, 2), Vector(3, 4)], [Vector(5, 6)]]

        # Mock return values for the methods
        mock_setup_axes.return_value = mock.MagicMock(spec=plt.Axes)
        mock_regularize.side_effect = lambda x: x
        mock_get_colors.return_value = {Vector(1, 2): 'blue', Vector(3, 4): 'red', Vector(5, 6): 'green'}

        printer = Printer()
        printer.print(*point_groups)

        # Assertions
        mock_setup_axes.assert_called_once()
        mock_regularize.assert_has_calls([
            mock.call(Vector(1, 2)),
            mock.call(Vector(3, 4)),
            mock.call(Vector(5, 6))
        ])
        mock_get_colors.assert_called_once_with([
            [Vector(1, 2), Vector(3, 4)],
            [Vector(5, 6)]
        ])
        mock_plot_hexagons.assert_called_once_with(
            mock_setup_axes.return_value,
            [Vector(1, 2), Vector(3, 4), Vector(5, 6)],
            mock_get_colors.return_value
        )
        mock_annotate.assert_not_called()

        # Repeat the test with `show_coordinates=True`
        printer = Printer(show_coordinates=True)
        printer.print(*point_groups)

        mock_annotate.assert_called_once_with(
            mock_setup_axes.return_value,
            [Vector(1, 2), Vector(3, 4), Vector(5, 6)],
            [Vector(1, 2), Vector(3, 4), Vector(5, 6)],
            printer.plot_size
        )

    def test_regularize_vector(self):
        # Setup
        test_vector = Vector(1, 2)
        expected_regularized_vector = Vector(
            -test_vector.i - test_vector.j / 2,
            np.sqrt(3) * test_vector.j / 2
        )

        # Exercise
        regularized_test_vector = Printer._regularize_vector(test_vector)

        # Verify
        self.assertEqual(expected_regularized_vector, regularized_test_vector)

    @patch('src.chinese_checkers.geometry.Printer.colors.TABLEAU_COLORS',
           {'tab:blue': '#1f77b4', 'tab:orange': '#ff7f0e'})
    def test_get_point_colors(self):
        # Setup
        test_points_list = [
            [Vector(1, 2), Vector(3, 4)],
            [Vector(5, 6), Vector(7, 8)]
        ]
        expected_point_colors = {
            Vector(1, 2): 'tab:blue',
            Vector(3, 4): 'tab:blue',
            Vector(5, 6): 'tab:orange',
            Vector(7, 8): 'tab:orange'
        }

        # Exercise
        point_colors = Printer._get_point_colors(test_points_list)

        # Verify
        self.assertEqual(expected_point_colors, point_colors)

    def test_plot_hexagon_points(self):
        # Setup
        mock_axes = Mock(spec=plt.Axes)

        test_points = [Vector(1, 2), Vector(3, 4)]
        test_point_colors = {
            Vector(1, 2): 'tab:blue',
            Vector(3, 4): 'tab:orange'
        }

        # Exercise
        Printer._plot_hexagon_points(mock_axes, test_points, test_point_colors)

        # Verify
        self.assertEqual(mock_axes.add_patch.call_count, len(test_points))

        for idx, point in enumerate(test_points):
            actual_patch = mock_axes.add_patch.call_args_list[idx][0][0]
            self.assertIsInstance(actual_patch, RegularPolygon)
            self.assertEqual(actual_patch.xy, (point.i, point.j))

            # Convert color name to RGBA tuple for comparison
            expected_rgba = colors.to_rgba(test_point_colors[point], alpha=0.3)
            self.assertEqual(actual_patch._facecolor, expected_rgba)

        mock_axes.scatter.assert_called_once_with(
            [point.i for point in test_points],
            [point.j for point in test_points],
            alpha=0
        )

    def test_annotate_points(self):
        # Setup
        mock_axes = Mock(spec=plt.Axes)

        test_points = [Vector(1, 2), Vector(3, 4)]
        test_annotations = [Vector(5, 6), Vector(7, 8)]
        print_size = 10

        # Exercise
        Printer._annotate_points(mock_axes, test_points, test_annotations, print_size)

        # Verify
        self.assertEqual(mock_axes.annotate.call_count, len(test_points))

        for point, annotation in zip(test_points, test_annotations):
            annotation_text = f"({annotation.i}, {annotation.j})"
            mock_axes.annotate.assert_any_call(
                annotation_text,
                (point.i, point.j),
                ha='center',
                va='center',
                fontsize=print_size / 2 + 1
            )

    def test_setup_axes_with_provided_axes(self):
        # Setup
        mock_axes = Mock(spec=plt.Axes)
        plot_size = 10

        # Exercise
        returned_axes = Printer._setup_axes(plot_size, provided_axes=mock_axes)

        # Verify
        self.assertEqual(returned_axes, mock_axes, "The method did not return the provided axes.")

    def test_setup_axes_without_provided_axes(self):
        # Setup
        plot_size = 10

        # Exercise
        returned_axes = Printer._setup_axes(plot_size)

        # Verify
        self.assertIsInstance(returned_axes, plt.Axes, "The returned object is not an Axes instance.")
        self.assertEqual(returned_axes.get_aspect(), 1.0, "Aspect ratio of the axes is not 'equal'.")
        self.assertEqual(returned_axes.figure.get_size_inches()[0], plot_size, "The width of the figure is incorrect.")
        self.assertEqual(returned_axes.figure.get_size_inches()[1], plot_size, "The height of the figure is incorrect.")

