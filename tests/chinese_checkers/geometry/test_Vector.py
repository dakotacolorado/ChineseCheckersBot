from unittest import TestCase

from src.chinese_checkers.geometry.Vector import Vector


class TestVector(TestCase):

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

    def test_hash_for_equal_vectors(self):
        v1 = Vector(1, 2)
        v2 = Vector(1, 2)
        self.assertEqual(hash(v1), hash(v2))

    def test_vector_in_set(self):
        v1 = Vector(1, 2)
        v2 = Vector(1, 2)  # Equal to v1
        v3 = Vector(2, 1)  # Different from v1

        s = {v1}

        # Since v1 and v2 are equal, adding v2 shouldn't increase the set size
        s.add(v2)
        self.assertEqual(len(s), 1)

        # Adding v3 should increase the set size
        s.add(v3)
        self.assertEqual(len(s), 2)

    def test_repr_method(self):
        v = Vector(3, 4)
        self.assertEqual(repr(v), "(3, 4)")

    def test_add_method(self):
        v1 = Vector(1, 2)
        v2 = Vector(3, 4)
        self.assertEqual(v1 + v2, Vector(4, 6))

    def test_sub_method(self):
        v1 = Vector(1, 2)
        v2 = Vector(3, 4)
        self.assertEqual(v1 - v2, Vector(-2, -2))

    def test_scalar_multiplication(self):
        v = Vector(1, 2)
        self.assertEqual(v * 2, Vector(2, 4))

    def test_scalar_multiplication_with_float(self):
        v = Vector(1, 2)
        self.assertEqual(v * 2.0, Vector(2.0, 4.0))

    def test_element_wise_multiplication(self):
        v1 = Vector(1, 2)
        v2 = Vector(3, 4)
        self.assertEqual(v1 * v2, Vector(3, 8))

    def test_string_multiplication(self):
        v = Vector(1, 2)
        with self.assertRaises(TypeError):
            v * "not a vector"

    def test_distance(self):
        v1 = Vector(1, 1)
        v2 = Vector(4, 5)

        # verify
        self.assertEqual(v1.distance(v2), 5.0)
        self.assertEqual(v1.distance(v1), 0)



