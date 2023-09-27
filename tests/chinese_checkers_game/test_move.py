from unittest import TestCase

from src.chinese_checkers_game.move import Move
from src.chinese_checkers_game.position import Position


class TestMove(TestCase):

    def test_initialization(self):
        pos = Position(1, 2)
        move = Move(1, 1, pos)
        self.assertEqual(move.i, 1)
        self.assertEqual(move.j, 1)
        self.assertEqual(move.position, pos)

    def test_apply(self):
        pos = Position(1, 2)
        move = Move(1, 1, pos)
        new_pos = move.apply()
        self.assertEqual(new_pos.i, 2)
        self.assertEqual(new_pos.j, 3)