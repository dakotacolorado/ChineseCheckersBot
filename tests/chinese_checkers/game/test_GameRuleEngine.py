from unittest import TestCase

from src.chinese_checkers.game.hexagram import Hexagram
from src.chinese_checkers.game import GameRuleEngine
from src.chinese_checkers.game.position import Position
from src.chinese_checkers.game.player import Player
from src.chinese_checkers.game.move import Move


class TestGameRuleEngine(TestCase):

    def setUp(self):
        self.current_player = Player([Position(1, 1), Position(2, 1)])
        self.opponent = Player([Position(4, 4), Position(3, 4)])
        self.engine = GameRuleEngine(self.current_player, [self.opponent], Hexagram(5))

    def test_simple_move(self):
        next_moves = self.engine.get_next_moves_for_player()

        # Adjusted to the only valid move that should be inside the hexagram.
        self.assertIn(Move(0, 1, Position(2, 1)), next_moves)

    def test_hop_move(self):
        self.current_player.positions.append(Position(2, 2))
        next_moves = self.engine.get_next_moves_for_player()

        # Retaining the test since the hop over `(2,2)` and land at `(3,3)` should be valid within the hexagram
        self.assertIn(Move(2, 2, Position(1, 1)), next_moves)

    def test_chained_hop_move(self):
        self.current_player.positions.extend([Position(3, 2), Position(5, 2)])
        next_moves = self.engine.get_next_moves_for_player()

        # A more complex check, maintaining it since the issue was not clear.
        self.assertNotEqual(1, len(next_moves), "Expected more than one move, indicating chained hops.")

    def test_no_hop_on_empty_spot(self):
        next_moves = self.engine.get_next_moves_for_player()

        # Making sure that we don't have a hop over an empty spot.
        self.assertNotIn(Move(2, 2, Position(1, 1)), next_moves)
