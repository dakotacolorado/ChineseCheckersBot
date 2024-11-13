from unittest import TestCase

from src.chinese_checkers.game.ChineseCheckersGame import ChineseCheckersGame


class TestChineseCheckersGame(TestCase):

    def test_default_start_turn(self):
        game = ChineseCheckersGame.start_game()
        self.assertEqual(game.turn, 0)

    def test_game_equality_default_board(self):
        game1 = ChineseCheckersGame.start_game()
        game2 = ChineseCheckersGame.start_game()
        self.assertEqual(game1, game2)

    def test_game_equality_not_default_board(self):
        game1 = ChineseCheckersGame.start_game(number_of_players=6, board_size=10)
        game2 = ChineseCheckersGame.start_game(number_of_players=6, board_size=10)
        self.assertEqual(game1, game2)

    def test_game_not_equal_different_turns(self):
        game1 = ChineseCheckersGame.start_game()
        game2 = ChineseCheckersGame.start_game()
        game2.turn = 1
        self.assertNotEqual(game1, game2)

    def test_game_not_equal_different_players(self):
        game1 = ChineseCheckersGame.start_game()
        game2 = ChineseCheckersGame.start_game()
        game2.players[0].positions = []
        self.assertNotEqual(game1, game2)

    def test_game_not_equal_different_player_count(self):
        game1 = ChineseCheckersGame.start_game(number_of_players=2)
        game2 = ChineseCheckersGame.start_game(number_of_players=3)
        self.assertNotEqual(game1, game2)

    def test_game_not_equal_different_board(self):
        game1 = ChineseCheckersGame.start_game()
        game2 = ChineseCheckersGame.start_game(board_size=10)
        self.assertNotEqual(game1, game2)

    def test_hash(self):
        game = ChineseCheckersGame.start_game()
        self.assertEqual(hash(game), hash((tuple(game.players), game.board.radius)))

    def test_current_player(self):
        game = ChineseCheckersGame.start_game()
        self.assertEqual(game.get_current_player(), game.players[0])

        game = game.apply_move(game.get_next_moves()[0])
        self.assertEqual(game.get_current_player(), game.players[1])

    def test_other_players(self):
        game = ChineseCheckersGame.start_game()
        self.assertEqual(game.get_other_players(), game.players[1:])

        game = game.apply_move(game.get_next_moves()[0])
        self.assertEqual(game.get_other_players(), [game.players[0]] + game.players[2:])

    def test_is_game_not_won(self):
        game = ChineseCheckersGame.start_game()

        # verify
        self.assertFalse(game.is_game_won())

    def test_is_game_won(self):
        game = ChineseCheckersGame.start_game()
        game.players[0].positions = game.players[0].target_positions

        # verify
        self.assertTrue(game.is_game_won())

    def test_get_winner(self):
        game = ChineseCheckersGame.start_game()
        game.players[0].positions = game.players[0].target_positions

        # verify
        self.assertTrue(game.get_winner() == game.players[0])

    def test_no_backwards_moves_initial_state(self):
        game = ChineseCheckersGame.start_game()

        # Get moves with and without backwards filter in the initial state
        moves_without_filter = game.get_next_moves(remove_backwards_moves=False)
        moves_with_filter = game.get_next_moves(remove_backwards_moves=True)

        # Verify that the move lists are the same size initially, as there should be no backwards moves
        self.assertEqual(len(moves_with_filter), len(moves_without_filter),
                         "In the initial state, there should be no backwards moves to filter out")

    def test_moves_after_two_moves_applied(self):
        game = ChineseCheckersGame.start_game()

        # Apply two moves
        game = game.apply_move(game.get_next_moves()[0])
        game = game.apply_move(game.get_next_moves()[0])

        # Get moves with and without backwards filter after two moves
        moves_without_filter = game.get_next_moves(remove_backwards_moves=False)
        moves_with_filter = game.get_next_moves(remove_backwards_moves=True)

        # Verify that one move is removed with the filter enabled
        self.assertEqual(len(moves_without_filter) - 1, len(moves_with_filter),
                         "After applying two moves, the move list should be one less with the backwards filter")

    def test_moves_after_three_moves_applied(self):
        game = ChineseCheckersGame.start_game()

        # Apply three moves
        game = game.apply_move(game.get_next_moves()[0])
        game = game.apply_move(game.get_next_moves()[0])
        game = game.apply_move(game.get_next_moves()[0])

        # Get moves with and without backwards filter after three moves
        moves_without_filter = game.get_next_moves(remove_backwards_moves=False)
        moves_with_filter = game.get_next_moves(remove_backwards_moves=True)

        # Verify that one move is removed with the filter enabled, testing for both players
        self.assertEqual(len(moves_without_filter) - 1, len(moves_with_filter),
                         "After applying three moves, the move list should be one less with the backwards filter")
