from typing import List
from unittest import TestCase

from src.chinese_checkers.game.Move import Move
from src.chinese_checkers.geometry.Hexagram import Hexagram
from src.chinese_checkers.game.GameRuleEngine import GameRuleEngine
from src.chinese_checkers.game.Position import Position
from src.chinese_checkers.game.Player import Player


class TestGameRuleEngine(TestCase):

    @staticmethod
    def __basic_engine_setup(
            current_player_positions: List[Position] = [],
            opponent_player_positions: List[Position] = [],
    ) -> GameRuleEngine:
        current_player = Player(current_player_positions, [], "player1")
        opponent_player = Player(opponent_player_positions, [], "player2")
        hexagram = Hexagram(5)
        engine = GameRuleEngine(current_player, [opponent_player], hexagram)
        return engine

    def test_occupied_positions_initialization(self):
        # set-up
        current_player_positions = {Position(0, 1), Position(1, 1)}
        current_player = Player(current_player_positions, [], "player1")

        opponent_positions = {Position(-1, 0), Position(-1, 1)}
        opponent_player = Player(opponent_positions, [], "player2")

        hexagram = Hexagram(5)

        # exercise
        engine = GameRuleEngine(current_player, [opponent_player], hexagram)

        # verify
        self.assertEqual(engine.occupied_positions,
                         opponent_positions | current_player_positions)

    def test_unit_moves(self):
        # set-up
        start_position = Position(0, 0)
        engine = TestGameRuleEngine.__basic_engine_setup([start_position])

        # exercise
        next_moves = engine.get_next_moves_for_player()

        # verify
        self.assertEqual(
            set(next_moves),
            {
                Move(1, 0, start_position),
                Move(0, 1, start_position),
                Move(1, -1, start_position),
                Move(-1, 1, start_position),
                Move(-1, 0, start_position),
                Move(0, -1, start_position)
            }
        )

    def test_unit_moves_for_multiple_positions(self):
        # set-up
        start_position_1 = Position(0, 0)
        start_position_2 = Position(0, -1)
        engine = TestGameRuleEngine.__basic_engine_setup([start_position_1, start_position_2])

        # exercise
        next_moves = engine.get_next_moves_for_player()

        # verify
        self.assertEqual(
            set(next_moves),
            {
                # position 1 moves
                Move(1, 0, start_position_1),
                Move(0, 1, start_position_1),
                Move(1, -1, start_position_1),
                Move(-1, 1, start_position_1),
                Move(-1, 0, start_position_1),
                # position 1 jump move
                Move(0, -2, start_position_1),
                # position 2 moves
                Move(1, 0, start_position_2),
                Move(-1, 1, start_position_2),
                Move(-1, 0, start_position_2),
                Move(0, -1, start_position_2),
                Move(1, -1, start_position_2),
                # position 2 jump move
                Move(0, 2, start_position_2)
            }
        )

    def test_unit_moves_with_opponent_blocking(self):
        # set-up
        current_position = Position(0, 0)
        opponent_position = Position(0, -1)
        engine = TestGameRuleEngine.__basic_engine_setup([current_position], [opponent_position])

        # exercise
        next_moves = engine.get_next_moves_for_player()

        # verify
        self.assertEqual(
            set(next_moves),
            {
                # current moves
                Move(1, 0, current_position),
                Move(0, 1, current_position),
                Move(1, -1, current_position),
                Move(-1, 1, current_position),
                Move(-1, 0, current_position),
                # position jump moves
                Move(0, -2, current_position),
            },
        )

    def test_blocked_moves(self):
        # set-up
        start_position = Position(0, 0)
        engine = TestGameRuleEngine.__basic_engine_setup([start_position])

        # verify
        next_moves = engine.get_next_moves_for_player()
        self.assertEqual(
            set(next_moves),
            {
                Move(1, 0, start_position),
                Move(0, 1, start_position),
                Move(1, -1, start_position),
                Move(-1, 1, start_position),
                Move(-1, 0, start_position),
                Move(0, -1, start_position)
            }
        )

    def test_single_hop_move(self):
        # Test the single hop move base case.  (n=1 case)
        position = Position(0, 0)
        occupied_positions = [Position(0, 1)]
        visited = {position}
        unit_move = Position(0, 1)

        expected_hop = Move(0, 2, position)

        # exercise
        engine = TestGameRuleEngine.__basic_engine_setup(occupied_positions)
        hops = engine._get_hop_moves(position, visited, unit_move)
        print(hops)
        # verify
        self.assertEqual(len(hops), 1)
        self.assertEqual(hops[0], expected_hop)

    def test_double_hop_move(self):
        # Test the double hop move inductive step. (n+1 case)
        position = Position(0, 0)
        occupied_positions = [Position(0, 1), Position(0, 3)]
        visited = {position}
        unit_move = Position(0, 1)

        expected_hops = {Move(0, 2, position), Move(0, 4, position)}

        # exercise
        engine = TestGameRuleEngine.__basic_engine_setup(occupied_positions)
        hops = engine._get_hop_moves(position, visited, unit_move)

        # verify
        self.assertEqual(set(hops), expected_hops)

    def test_position_in_bounds(self):
        # set-up
        engine = TestGameRuleEngine.__basic_engine_setup()
        in_bounds_position = Position(1, 1)

        # verify
        self.assertIs(True, engine._is_position_in_bounds(in_bounds_position))

    def test_position_not_in_bounds(self):
        # set-up
        engine = TestGameRuleEngine.__basic_engine_setup()
        out_of_bounds_position = Position(100, 100)

        # verify
        self.assertIsNot(True, engine._is_position_in_bounds(out_of_bounds_position))

    def test_position_is_open(self):
        # set-up
        open_position = Position(1, 1)
        engine = TestGameRuleEngine.__basic_engine_setup()

        # verify
        self.assertIs(True, engine._is_position_open(open_position))

    def test_position_not_open(self):
        # set-up
        occupied_position = Position(1, 1)
        engine = TestGameRuleEngine.__basic_engine_setup([occupied_position])

        # verify
        self.assertIsNot(True, engine._is_position_open(occupied_position))
