from unittest import TestCase

from src.chinese_checkers.game.Position import Position


class TestPosition(TestCase):

    def test_initialization(self):
        pos = Position(1, 2)
        self.assertEqual(pos.i, 1)
        self.assertEqual(pos.j, 2)