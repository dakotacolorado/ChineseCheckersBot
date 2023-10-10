from unittest import TestCase

from src.chinese_checkers.game.move import Move
from src.chinese_checkers.game.player import Player
from src.chinese_checkers.game.position import Position


class TestPlayer(TestCase):

    def test_initialization(self):
        # set-up
        pos1 = Position(1, 2)
        pos2 = Position(3, 4)
        pos3 = Position(5, 6)
        pos4 = Position(7, 8)

        # exercise
        player = Player([pos1, pos2], [pos3, pos4])

        # verify
        self.assertEqual(len(player.positions), 2)
        self.assertEqual(player.positions[0], pos1)
        self.assertEqual(player.positions[1], pos2)
        self.assertEqual(player.target_positions[0], pos3)
        self.assertEqual(player.target_positions[1], pos4)

    def test_apply_move(self):
        # set-up
        pos1 = Position(1, 2)
        pos2 = Position(3, 4)
        player = Player([pos1, pos2], [])
        move = Move(1, 1, pos1)

        # exercise
        new_player = player.apply_move(move)

        # verify
        expected_pos = Position(2, 3)
        self.assertEqual(new_player.positions[0], expected_pos)
        self.assertEqual(new_player.positions[1], pos2)
