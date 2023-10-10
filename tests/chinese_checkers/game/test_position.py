from unittest import TestCase

from chinese_checkers.game.position import Position


class TestPosition(TestCase):

    def test_initialization(self):
        pos = Position(1, 2)
        self.assertEqual(pos.i, 1)
        self.assertEqual(pos.j, 2)