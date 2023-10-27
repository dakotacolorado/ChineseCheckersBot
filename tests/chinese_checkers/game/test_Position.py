from unittest import TestCase

from src.chinese_checkers.game.Position import Position


class TestPosition(TestCase):

    def test_initialization(self):
        pos = Position(1, 2)
        self.assertEqual(pos.i, 1)
        self.assertEqual(pos.j, 2)

    def test_to_tuple(self):
        pos = Position(1, 2)
        expected_tuple = (1, 2)
        self.assertEqual(expected_tuple, pos.to_tuple())

    def test_from_tuple(self):
        pos = Position.from_tuple((1, 2))
        expected_pos = Position(1, 2)
        self.assertEqual(expected_pos, pos)