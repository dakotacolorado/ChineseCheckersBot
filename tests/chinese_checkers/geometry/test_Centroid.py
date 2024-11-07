import unittest

from src.chinese_checkers.geometry import Centroid
from src.chinese_checkers.geometry import Vector


class TestCentroid(unittest.TestCase):
    def test_centroid_with_multiple_vectors(self):
        vectors = [Vector(1, 2), Vector(3, 4), Vector(5, 6)]
        centroid = Centroid.from_vectors(vectors)
        expected_centroid = Centroid(3, 4)  # Average of (1+3+5)/3, (2+4+6)/3
        self.assertEqual(centroid, expected_centroid)

    def test_centroid_with_single_vector(self):
        vectors = [Vector(5, 10)]
        centroid = Centroid.from_vectors(vectors)
        expected_centroid = Centroid(5, 10)  # Should be the same as the only vector
        self.assertEqual(centroid, expected_centroid)

    def test_centroid_with_empty_vector_list(self):
        vectors = []
        with self.assertRaises(ValueError):
            Centroid.from_vectors(vectors)

    def test_centroid_with_negative_coordinates(self):
        vectors = [Vector(-2, -3), Vector(-4, -6), Vector(-6, -9)]
        centroid = Centroid.from_vectors(vectors)
        expected_centroid = Centroid(-4, -6)  # Average of (-2-4-6)/3, (-3-6-9)/3
        self.assertEqual(centroid, expected_centroid)

    def test_centroid_with_mixed_coordinates(self):
        vectors = [Vector(1, -1), Vector(-1, 1), Vector(3, -3)]
        centroid = Centroid.from_vectors(vectors)
        expected_centroid = Centroid(1, -1)  # Average of (1-1+3)/3, (-1+1-3)/3
        self.assertEqual(centroid, expected_centroid)

