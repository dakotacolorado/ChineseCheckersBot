from unittest import TestCase

from src.chinese_checkers.geometry.Hexagram import Hexagram
from src.chinese_checkers.game.GameRuleEngine import GameRuleEngine
from src.chinese_checkers.game.Position import Position
from src.chinese_checkers.game.Player import Player
from src.chinese_checkers.game.Move import Move


class TestGameRuleEngine(TestCase):
    def test_simple_team_hop(self):
        current_player = Player([Position(1, 1), Position(2, 1)], [])
        opponent = Player([Position(4, 4), Position(3, 4)], [])
        engine = GameRuleEngine(current_player, [opponent], Hexagram(5))

        self.next_moves = engine.get_next_moves_for_player()

        self.assertIn(Move(0, 1, Position(2, 1)), self.next_moves)
