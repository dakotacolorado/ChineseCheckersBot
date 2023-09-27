from unittest import TestCase

from src.chinese_checkers_game.hexagram import Hexagram


class TestHexagram(TestCase):

    def setUp(self):
        # Setting up a range of hexagrams for testing
        self.hexagrams = [Hexagram(r) for r in range(1, 11)]

    def test_centered_on_origin(self):
        for H in self.hexagrams:
            hexagram_centroids = sum([p[0] + p[1] for p in H.hexagram_points])
            self.assertEqual(hexagram_centroids, 0, f"Failed for radius {H.radius}")

    def test_symmetry_about_origin(self):
        for H in self.hexagrams:
            for p in H.hexagram_points:
                self.assertIn((-p[0], -p[1]), H.hexagram_points)

    def test_no_duplicate_points(self):
        for H in self.hexagrams:
            unique_points = set(H.hexagram_points)
            self.assertEqual(len(unique_points), len(H.hexagram_points),
                             f"Duplicate points found for radius {H.radius}")

    def test_absolute_sum_condition(self):
        for H in self.hexagrams:
            for p in H.hexagram_points:
                self.assertLessEqual(abs(p[0]) + abs(p[1]), 3 * H.radius)
