from unittest import TestCase

from src.chinese_checkers.game.ChineseCheckersGame import ChineseCheckersGame


class TestChineseCheckersGame(TestCase):

    def test_default_start_turn(self):
        game = ChineseCheckersGame.start_game()
        self.assertEqual(game.turn, 0)
