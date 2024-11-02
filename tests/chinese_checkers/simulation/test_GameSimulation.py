from unittest import TestCase
from src.chinese_checkers.game.ChineseCheckersGame import ChineseCheckersGame


class TestChineseCheckersGame(TestCase):

    def test_default_start_turn(self):
        game = ChineseCheckersGame.start_game()
        self.assertEqual(game.turn, 0, "Expected game turn to start at 0, but got {game.turn}.")

    def test_game_equality_default_board(self):
        game1 = ChineseCheckersGame.start_game()
        game2 = ChineseCheckersGame.start_game()
        self.assertEqual(game1, game2, "Expected two games with default settings to be equal, but they were not.")

    def test_game_equality_not_default_board(self):
        game1 = ChineseCheckersGame.start_game(number_of_players=6, board_size=10)
        game2 = ChineseCheckersGame.start_game(number_of_players=6, board_size=10)
        self.assertEqual(game1, game2,
                         "Expected two games with 6 players and board size 10 to be equal, but they were not.")

    def test_game_not_equal_different_turns(self):
        game1 = ChineseCheckersGame.start_game()
        game2 = ChineseCheckersGame.start_game()
        game2.turn = 1
        self.assertNotEqual(game1, game2, "Expected games with different turns (0 and 1) to be unequal.")

    def test_game_not_equal_different_players(self):
        game1 = ChineseCheckersGame.start_game()
        game2 = ChineseCheckersGame.start_game()
        game2.players[0].positions = []
        self.assertNotEqual(game1, game2, "Expected games with different player positions to be unequal.")

    def test_game_not_equal_different_player_count(self):
        game1 = ChineseCheckersGame.start_game(number_of_players=2)
        game2 = ChineseCheckersGame.start_game(number_of_players=3)
        self.assertNotEqual(game1, game2, "Expected games with different player counts (2 vs. 3) to be unequal.")

    def test_game_not_equal_different_board(self):
        game1 = ChineseCheckersGame.start_game()
        game2 = ChineseCheckersGame.start_game(board_size=10)
        self.assertNotEqual(game1, game2, "Expected games with different board sizes (default vs. 10) to be unequal.")

    def test_hash(self):
        game = ChineseCheckersGame.start_game()
        expected_hash = hash((tuple(game.players), game.board.radius))
        self.assertEqual(hash(game), expected_hash, f"Expected hash to match {expected_hash}, but got {hash(game)}.")

    def test_current_player(self):
        game = ChineseCheckersGame.start_game()
        self.assertEqual(game.get_current_player(), game.players[0],
                         "Expected current player to be the first player initially.")

        game = game.apply_move(game.get_next_moves()[0])
        self.assertEqual(game.get_current_player(), game.players[1],
                         "Expected current player to switch to the second player after one move.")

    def test_other_players(self):
        game = ChineseCheckersGame.start_game()
        self.assertEqual(game.get_other_players(), game.players[1:],
                         "Expected other players to exclude the first player initially.")

        game = game.apply_move(game.get_next_moves()[0])
        self.assertEqual(game.get_other_players(), [game.players[0]] + game.players[2:],
                         "Expected other players to update after one move.")

    def test_is_game_not_won(self):
        game = ChineseCheckersGame.start_game()
        self.assertFalse(game.is_game_won(),
                         "Expected game to not be won at the start, but is_game_won() returned True.")

    def test_is_game_won(self):
        game = ChineseCheckersGame.start_game()
        game.players[0].positions = game.players[0].target_positions
        self.assertTrue(game.is_game_won(),
                        "Expected game to be won when player 1 reaches target positions, but is_game_won() returned False.")

    def test_get_winner(self):
        game = ChineseCheckersGame.start_game()
        game.players[0].positions = game.players[0].target_positions
        self.assertTrue(game.get_winner() == game.players[0],
                        "Expected player 1 to be the winner when reaching target positions, but get_winner() returned a different player.")

    # New tests for input validation
    def test_invalid_number_of_players(self):
        with self.assertRaises(ValueError) as context:
            ChineseCheckersGame.start_game(number_of_players=5)
        self.assertEqual(str(context.exception), "Invalid number of players: 5. Must be one of {2, 3, 4, 6}.",
                         "Expected ValueError with specific message for invalid number of players, but got a different message.")

    def test_invalid_board_size_zero(self):
        with self.assertRaises(ValueError) as context:
            ChineseCheckersGame.start_game(board_size=0)
        self.assertEqual(str(context.exception), "Invalid board size: 0. Board size must be a positive integer.",
                         "Expected ValueError with specific message for board size of 0, but got a different message.")

    def test_invalid_board_size_negative(self):
        with self.assertRaises(ValueError) as context:
            ChineseCheckersGame.start_game(board_size=-1)
        self.assertEqual(str(context.exception), "Invalid board size: -1. Board size must be a positive integer.",
                         "Expected ValueError with specific message for negative board size, but got a different message.")

    def test_invalid_board_size_non_integer(self):
        with self.assertRaises(ValueError) as context:
            ChineseCheckersGame.start_game(board_size="large")
        self.assertEqual(str(context.exception), "Invalid board size: large. Board size must be a positive integer.",
                         "Expected ValueError with specific message for non-integer board size, but got a different message.")
