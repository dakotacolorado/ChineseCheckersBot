from src.chinese_checkers.model.IModel import IModel
from src.chinese_checkers.game.ChineseCheckersGame import ChineseCheckersGame
from src.chinese_checkers.game.Player import Player
from src.chinese_checkers.game.Position import Position
from unittest import TestCase


class BasicModel(IModel):
    def _chose_next_move(self, current_player, other_players, moves):
        """ Chose the first move in the list of moves. """
        return moves[0]


class TestIModel(TestCase):
    def test_make_move(self):
        game = ChineseCheckersGame.start_game()
        current_moves = game.get_next_moves()

        expected_game = ChineseCheckersGame.start_game()
        expected_game = expected_game.apply_move(current_moves[0])

        model = BasicModel()
        game = model.make_move(game)

        self.assertEqual(expected_game, game)
