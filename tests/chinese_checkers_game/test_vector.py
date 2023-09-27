import unittest

from chinese_checkers_game.vector import Vector


class TestVector(unittest.TestCase):

    def test_initialization(self):
        v = Vector(1, 2)
        self.assertEqual(v.i, 1)
        self.assertEqual(v.j, 2)

    def test_equality_same_vectors(self):
        v1 = Vector(1, 2)
        v2 = Vector(1, 2)
        self.assertTrue(v1 == v2)

    def test_equality_different_vectors(self):
        v1 = Vector(1, 2)
        v2 = Vector(2, 3)
        self.assertFalse(v1 == v2)

    def test_equality_with_non_vector(self):
        v = Vector(1, 2)
        self.assertFalse(v == "not a vector")