from unittest import TestCase

from src.chinese_checkers.game.Move import Move
from src.chinese_checkers.game.Position import Position


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

    def test_str(self):
        pos = Position(1, 2)
        move = Move(1, 1, pos)

        # Check if the __str__ method returns the expected string representation
        expected_str = "Move(1, 1), Position(1, 2)"
        self.assertEqual(str(move), expected_str)

    def test_equal(self):
        pos = Position(1, 2)
        move1 = Move(1, 1, pos)
        move2 = Move(1, 1, pos)
        self.assertEqual(move1, move2)

    def test_not_equal_position(self):
        pos1 = Position(1, 2)
        pos2 = Position(1, 3)
        move1 = Move(1, 1, pos1)
        move2 = Move(1, 1, pos2)
        self.assertNotEqual(move1, move2)

    def test_not_equal_move(self):
        pos = Position(1, 2)
        move1 = Move(1, 1, pos)
        move2 = Move(1, 2, pos)
        self.assertNotEqual(move1, move2)

    def test_hash_for_equal_moves(self):
        pos = Position(1, 2)
        move1 = Move(1, 1, pos)
        move2 = Move(1, 1, pos)
        self.assertEqual(hash(move1), hash(move2))

    def test_move_in_set(self):
        pos1 = Position(1, 2)
        pos2 = Position(1, 2)

        move1 = Move(1, 1, pos1)
        move2 = Move(1, 1, pos2)  # Equal to move1
        move3 = Move(2, 1, pos1)  # Different from move1

        s = {move1}

        # Since move1 and move2 are equal, adding move2 shouldn't increase the set size
        s.add(move2)
        self.assertEqual(len(s), 1)

        # Adding move3 should increase the set size
        s.add(move3)
        self.assertEqual(len(s), 2)
