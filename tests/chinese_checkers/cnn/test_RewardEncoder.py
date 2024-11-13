import unittest
from src.chinese_checkers.game import ChineseCheckersGame, Player
from chinese_checkers.cnn import RewardEncoder

class TestRewardEncoder(unittest.TestCase):
    def setUp(self):
        """Set up the RewardEncoder and initial game states for testing."""
        self.reward_encoder = RewardEncoder()
        self.start_game = ChineseCheckersGame.start_game(number_of_players=2, board_size=4)

    def test_all_positions_in_start_should_return_negative_one_not_in_start_and_zero_in_target(self):
        """If all pieces are in starting positions, expect _player_positions_not_in_start = -1, _player_positions_in_target = 0."""
        self.assertAlmostEqual(
            RewardEncoder._player_positions_in_target(self.start_game), 0,
            msg="Expected player positions in target to be 0 at game start"
        )
        self.assertAlmostEqual(
            RewardEncoder._player_positions_not_in_start(self.start_game), -1,
            msg="Expected player positions not in start to be -1 at game start"
        )

    def test_all_positions_in_target_should_return_one_in_target_and_zero_not_in_start(self):
        """If all pieces are in target positions, expect _player_positions_in_target = 1, _player_positions_not_in_start = 0."""
        player_1 = self.start_game.players[0]
        player_2 = self.start_game.players[1]
        player_1_target = player_1.target_positions
        player_2_target = player_2.target_positions
        game_t1 = ChineseCheckersGame(
            players=[
                Player(player_1_target, player_1_target, player_1.player_id),
                Player(player_2_target, player_2_target, player_2.player_id)
            ],
            turn=1,
            board=self.start_game.board,
            printer=self.start_game.printer
        )
        self.assertAlmostEqual(
            RewardEncoder._player_positions_in_target(game_t1), 1,
            msg="Expected player positions in target to be 1 after update"
        )
        self.assertAlmostEqual(
            RewardEncoder._player_positions_not_in_start(game_t1), 0,
            msg="Expected player positions not in start to be 0 after update"
        )

    def test_one_piece_out_of_target_positions_should_return_point_nine_in_target(self):
        """If one piece is out of target positions, expect _player_positions_in_target = 0.9, _player_positions_not_in_start = 0."""
        player_1 = self.start_game.players[0]
        player_2 = self.start_game.players[1]
        player_1_target = player_1.target_positions
        player_2_target = player_2.target_positions
        game_t1 = ChineseCheckersGame(
            players=[
                Player(player_1_target, player_1_target, player_1.player_id),
                Player(player_2_target, player_2_target, player_2.player_id)
            ],
            turn=self.start_game.turn + 1,
            board=self.start_game.board,
            printer=self.start_game.printer
        )
        move_t2 = game_t1.get_next_moves()[0]
        game_t2 = game_t1.apply_move(move_t2)
        self.assertAlmostEqual(
            RewardEncoder._player_positions_in_target(game_t2), 0.9,
            msg="Expected player positions in target to be 0.9 after moving one piece out"
        )
        self.assertAlmostEqual(
            RewardEncoder._player_positions_not_in_start(game_t2), 0,
            msg="Expected player positions not in start to be 0 after move"
        )

    def test_one_piece_out_of_start_positions_should_return_negative_point_nine_not_in_start(self):
        """If one piece is out of start positions, expect _player_positions_not_in_start = -0.9, _player_positions_in_target = 0."""
        player_1 = self.start_game.players[0]
        player_2 = self.start_game.players[1]
        player_1_start = player_1.positions
        player_2_start = player_2.positions
        game_t1 = ChineseCheckersGame(
            players=[
                Player(player_1_start, player_1.target_positions, player_1.player_id),
                Player(player_2_start, player_2.target_positions, player_2.player_id)
            ],
            turn=1,
            board=self.start_game.board,
            printer=self.start_game.printer
        )
        move_t3 = game_t1.get_next_moves()[0]
        game_t2 = game_t1.apply_move(move_t3)
        self.assertAlmostEqual(
            RewardEncoder._player_positions_in_target(game_t2), 0,
            msg="Expected player positions in target to be 0 after move"
        )
        self.assertAlmostEqual(
            RewardEncoder._player_positions_not_in_start(game_t2), -0.9,
            msg="Expected player positions not in start to be -0.9 after moving one piece out"
        )

    # Additional tests for distance_from_win_loss to ensure coverage of reward progression

    def test_positive_progression_for_winner_in_distance_from_win_loss(self):
        """Test that the winning player's reward progresses positively based on their turn progress."""
        winner_id = self.start_game.players[0].player_id
        turn = 50
        game_length = 100
        distance = RewardEncoder._distance_from_win_loss(self.start_game, turn, game_length, winner_id)
        expected_distance = (turn + 1) / game_length
        self.assertAlmostEqual(
            distance, expected_distance, places=2,
            msg=f"Expected positive reward for the winner, calculated as {(turn + 1) / game_length}"
        )

    def test_negative_progression_for_loser_in_distance_from_win_loss(self):
        """Test that the losing player's reward progresses negatively based on their turn progress."""
        winner_id = self.start_game.players[0].player_id
        turn = 50
        game_length = 100
        losing_player_game = self.start_game
        losing_player_game.turn = 1  # Switch to second player to simulate the losing player

        distance = RewardEncoder._distance_from_win_loss(losing_player_game, turn, game_length, winner_id)
        expected_distance = -1 * (turn + 1) / game_length
        self.assertAlmostEqual(
            distance, expected_distance, places=2,
            msg=f"Expected negative reward for the loser, calculated as {-1 * (turn + 1) / game_length}"
        )

    def test_minimal_distance_from_win_loss_at_game_start(self):
        """Verify minimal reward progression at the start of the game."""
        winner_id = self.start_game.players[0].player_id
        turn = 0
        game_length = 100
        distance = RewardEncoder._distance_from_win_loss(self.start_game, turn, game_length, winner_id)
        expected_distance = (turn + 1) / game_length
        self.assertAlmostEqual(
            distance, expected_distance, places=2,
            msg=f"Expected minimal reward distance at game start, calculated as {(turn + 1) / game_length}"
        )

    def test_maximal_distance_from_win_loss_at_game_end(self):
        """Verify maximal reward progression at the end of the game."""
        winner_id = self.start_game.players[0].player_id
        turn = 99  # Last turn
        game_length = 100
        distance = RewardEncoder._distance_from_win_loss(self.start_game, turn, game_length, winner_id)
        expected_distance = (turn + 1) / game_length
        self.assertAlmostEqual(
            distance, expected_distance, places=2,
            msg=f"Expected maximal reward distance at game end, calculated as {(turn + 1) / game_length}"
        )

if __name__ == "__main__":
    unittest.main()
