from unittest import TestCase

from src.chinese_checkers_game.move import Move
from src.chinese_checkers_game.player import Player
from src.chinese_checkers_game.position import Position


class TestPlayer(TestCase):

    def test_initialization(self):
        pos1 = Position(1, 2)
        pos2 = Position(3, 4)
        player = Player([pos1, pos2])
        self.assertEqual(len(player.positions), 2)
        self.assertEqual(player.positions[0], pos1)
        self.assertEqual(player.positions[1], pos2)

    def test_applyMove(self):
        pos1 = Position(1, 2)
        pos2 = Position(3, 4)
        player = Player([pos1, pos2])
        move = Move(1, 1, pos1)
        new_player = player.applyMove(move)
        expected_pos = Position(2, 3)
        self.assertEqual(new_player.positions[0], expected_pos)
        self.assertEqual(new_player.positions[1], pos2)