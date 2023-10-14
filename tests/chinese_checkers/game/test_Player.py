from unittest import TestCase

from src.chinese_checkers.game.Move import Move
from src.chinese_checkers.game.Player import Player
from src.chinese_checkers.game.Position import Position


class TestPlayer(TestCase):

    def test_initialization(self):
        # set-up
        positions = [Position(1, 1), Position(2, 2), Position(3, 3)]
        target_positions = [Position(1, 1), Position(2, 2), Position(4, 4)]

        # exercise
        player = Player(positions, target_positions, "player1")

        # verify
        self.assertEqual(len(player.positions), 3)
        self.assertEqual(player.positions, positions)
        self.assertEqual(player.target_positions, target_positions)

    def test_apply_move(self):
        # set-up
        pos1 = Position(1, 2)
        pos2 = Position(3, 4)
        player = Player([pos1, pos2], [], "player1")
        move = Move(1, 1, pos1)

        # exercise
        new_player = player.apply_move(move)

        # verify
        expected_pos = Position(2, 3)
        self.assertEqual(new_player.positions[0], expected_pos)
        self.assertEqual(new_player.positions[1], pos2)

    def test_eq_same_player_id(self):
        player1 = Player([], [], "player1")
        player2 = Player([], [], "player1")

        # verify
        self.assertTrue(player1 == player2)

    def test_eq_different_player_id(self):
        player1 = Player([], [], "player1")
        player2 = Player([], [], "player2")

        # verify
        self.assertFalse(player1 == player2)

    def test_player_has_reached_target(self):
        positions = [Position(1, 1), Position(2, 2), Position(3, 3)]
        target_positions = positions
        player = Player(positions, target_positions, "player1")

        # verify
        self.assertTrue(player.has_player_reached_target())

    def test_player_has_not_reached_target(self):
        positions = [Position(1, 1), Position(2, 2), Position(3, 3)]
        target_positions = [Position(1, 1), Position(2, 2), Position(4, 4)]
        player = Player(positions, target_positions, "player2")

        # verify
        self.assertFalse(player.has_player_reached_target())
