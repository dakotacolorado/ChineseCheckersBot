from typing import List
from unittest import TestCase

from src.chinese_checkers.game.Move import Move
from src.chinese_checkers.game.ChineseCheckersGame import ChineseCheckersGame
from src.chinese_checkers.game.Player import Player
from src.chinese_checkers.game.Position import Position
from src.chinese_checkers.model.CentroidModel import CentroidModel


class TestCentroidModel(TestCase):

    def setUp(self) -> None:
        self.game = ChineseCheckersGame.start_game()
        self.model = CentroidModel(self.game)

    def test_calculate_centroid_at_zero(self):
        positions = [
            Position(0, 0),
            Position(1, 0),
            Position(0, 1),
            Position(1, 1),
            Position(-1, -1),
            Position(-1, 0),
            Position(0, -1),
        ]

        centroid = CentroidModel._calculate_centroid(positions)

        self.assertEqual(centroid, Position(0, 0))

    def test_calculate_centroid_at_non_zero(self):
        positions = [
            Position(1, 1),
            Position(2, 2),
            Position(3, 3)
        ]

        centroid = CentroidModel._calculate_centroid(positions)

        self.assertEqual(centroid, Position(2, 2))

    def test_calculate_centroid_no_positions(self):
        positions = []

        centroid = CentroidModel._calculate_centroid(positions)

        self.assertIsNone(centroid)

    def test_chose_next_move(self):
        current_player = Player([], [], 'Player 1')
        other_players = [Player([], [], 'Player 2')]
        moves = []
        with self.assertRaises(Exception):
            self.model._chose_next_move(current_player, other_players, moves)

    @staticmethod
    def _unit_moves_for_position(position: Position) -> List[Move]:
        return [
            Move(1, 0, position),
            Move(0, 1, position),
            Move(-1, 1, position),
            Move(-1, 0, position),
            Move(0, -1, position),
            Move(1, -1, position),
        ]

    def test_chose_next_unit_move(self):
        position = Position(0, 0)
        target_position = Position(3, 0)
        expected_move = Move(1, 0, position)

        current_player = Player([position], [target_position], 'Player 1')
        other_players = []

        moves = TestCentroidModel._unit_moves_for_position(position)

        next_move = self.model._chose_next_move(current_player, other_players, moves)

        self.assertEqual(expected_move, next_move)

    def test_chose_next_hop_move(self):
        position = Position(0, 0)
        hop_position = Position(1, 0)
        target_position = Position(3, 1)
        expected_move = Move(2, 1, position)

        current_player = Player([position, hop_position], [target_position], 'Player 1')
        other_players = []

        moves = [
            *TestCentroidModel._unit_moves_for_position(position),
            expected_move
        ]

        next_move = self.model._chose_next_move(current_player, other_players, moves)

        self.assertEqual(expected_move, next_move)
