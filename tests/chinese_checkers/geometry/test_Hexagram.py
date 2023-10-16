from unittest import TestCase

from pydash import flatten

from src.chinese_checkers.geometry.Hexagram import Hexagram
from src.chinese_checkers.geometry.Vector import Vector


class TestHexagram(TestCase):
    def setUp(self):
        self.hexagrams = [Hexagram(r) for r in range(1, 11)]

    def test_centered_on_origin(self):
        """Ensure the centroid of the hexagram is at the origin."""
        for H in self.hexagrams:
            hexagram_centroids = sum([p.i + p.j for p in H.hexagram_points])
            self.assertEqual(hexagram_centroids, 0, f"Failed for radius {H.radius}")

    def test_symmetry_about_origin(self):
        """Verify that the hexagram is symmetrical about the origin."""
        for H in self.hexagrams:
            for p in H.hexagram_points:
                inverse_point = Vector(-p.i, -p.j)
                self.assertIn(inverse_point, H.hexagram_points)

    def test_symmetry_about_origin_for_corner_points(self):
        """Verify that the hexagram corner points are symmetrical about the origin."""
        for H in self.hexagrams:
            corner_points = flatten(H.hexagram_corner_points.values())
            for p in corner_points:
                inverse_point = Vector(-p.i, -p.j)
                self.assertIn(inverse_point, corner_points, f"Failed for {H}")

    def test_no_duplicate_points(self):
        """Check that there are no duplicate points within the hexagram."""
        for H in self.hexagrams:
            unique_points = set(H.hexagram_points)
            self.assertEqual(len(unique_points), len(H.hexagram_points),
                             f"Duplicate points found for radius {H.radius}")

    def test_absolute_sum_condition(self):
        """Ensure each point adheres to the absolute sum condition."""
        for H in self.hexagrams:
            for p in H.hexagram_points:
                self.assertLessEqual(abs(p.i) + abs(p.j), 3 * H.radius)

    def test_equal(self):
        for H in self.hexagrams:
            self.assertEqual(H, H)

    def test_not_equal_different_radius(self):
        h1 = Hexagram(10)
        h2 = Hexagram(11)

        self.assertNotEqual(h1, h2)

    def test_not_equal_different_hexagon_points(self):
        h1 = Hexagram(10)
        h2 = Hexagram(10)
        h2.hexagram_points = [Vector(1, 1)]

        self.assertNotEqual(h1, h2)

    def test_not_equal_different_hexagon_edges(self):
        h1 = Hexagram(10)
        h2 = Hexagram(10)
        h2.hexagon_edges = [Vector(1, 1)]

        self.assertNotEqual(h1, h2)

    def test_not_equal_different_rhombus_grid(self):
        h1 = Hexagram(10)
        h2 = Hexagram(10)
        h2.rhombus_grid = [Vector(1, 1)]

        self.assertNotEqual(h1, h2)

    def test_not_equal_different_hexagram_points(self):
        h1 = Hexagram(10)
        h2 = Hexagram(10)
        h2.hexagram_points = [Vector(1, 1)]

        self.assertNotEqual(h1, h2)

    def test_not_equal_different_triangle_grid(self):
        h1 = Hexagram(10)
        h2 = Hexagram(10)
        h2.triangle_grid = [Vector(1, 1)]

        self.assertNotEqual(h1, h2)

    def test_not_equal_different_hexagram_corner_points(self):
        h1 = Hexagram(10)
        h2 = Hexagram(10)
        h2.hexagram_corner_points = [Vector(1, 1)]

        self.assertNotEqual(h1, h2)
