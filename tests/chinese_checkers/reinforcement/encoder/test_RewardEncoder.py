import unittest
import random
from src.chinese_checkers.game import ChineseCheckersGame, Player
from src.chinese_checkers.reinforcement.encode import RewardEncoder

class TestRewardEncoder(unittest.TestCase):
    def setUp(self):
        """Set up the RewardEncoder and initial game states for testing."""
        self.reward_encoder = RewardEncoder()
        self.start_game = ChineseCheckersGame.start_game(number_of_players=2, board_size=4)
        self.end_game = ChineseCheckersGame(
            players=[
                Player(p.target_positions, p.target_positions, p.player_id) for p in self.start_game.players
            ],
            turn=100
        )

    def test_group_distance_from_target_is_zero_at_game_start(self):
        """Test that the group distance to the target is minimal (0) at game start."""
        inverse_distance = RewardEncoder._group_distance_from_target(self.start_game)
        print("Start game inverse distance:", inverse_distance)
        # Expect the inverse distance to be near zero at the start of the game
        self.assertAlmostEqual(
            inverse_distance, 0.000, places=2,
            msg="Expected inverse distance to be approximately 0.000 at game start"
        )

    def test_group_distance_near_max_after_random_perturbations(self):
        """Test that the maximum inverse distance is achieved after random perturbations."""
        inverse_distances = []

        for _ in range(100):
            # Apply two random moves to perturb the end game state
            end_game_perturbed = self.end_game.apply_move(random.choice(self.end_game.get_next_moves()))
            end_game_perturbed = end_game_perturbed.apply_move(random.choice(end_game_perturbed.get_next_moves()))

            # Calculate the inverse distance and store it
            inverse_distance = RewardEncoder._group_distance_from_target(end_game_perturbed)
            inverse_distances.append(inverse_distance)

        # Compute the maximum of the inverse distances
        max_inverse_distance = max(inverse_distances)
        print("Max end game inverse distance after random perturbation:", max_inverse_distance)

        # Assert the maximum inverse distance is close to expected max
        self.assertAlmostEqual(
            max_inverse_distance, 0.99999, places=2,
            msg="Expected maximum inverse distance to be close to 0.99999 after random perturbations"
        )

    def test_group_distance_is_one_at_game_end(self):
        """Test that the group distance to the target is maximal (1) at the game end state."""
        inverse_distance = self.reward_encoder._group_distance_from_target(self.end_game)
        print("End game inverse distance:", inverse_distance)
        # Expect the inverse distance to be 1 when players reach the target position
        self.assertAlmostEqual(
            inverse_distance, 1, places=2,
            msg="Expected inverse distance to be 1 when the game is at the end state"
        )


    def test_distance_from_win_loss_positive_for_winner(self):
        """Test that the winning player's reward is positive and based on their turn progress."""
        winner_id = self.start_game.players[0].player_id
        turn = 50
        game_length = 100
        distance = RewardEncoder._distance_from_win_loss(self.start_game, turn, game_length, winner_id)
        expected_distance = (turn + 1) / game_length
        self.assertAlmostEqual(
            distance, expected_distance, places=2,
            msg=f"Expected positive reward for the winner, calculated as {(turn + 1) / game_length}"
        )

    def test_distance_from_win_loss_negative_for_loser(self):
        """Test that the losing player's reward is negative and based on their turn progress."""
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

    def test_distance_from_win_loss_at_game_start(self):
        """Test that the win/loss distance is minimal at the start of the game."""
        winner_id = self.start_game.players[0].player_id
        turn = 0
        game_length = 100
        distance = RewardEncoder._distance_from_win_loss(self.start_game, turn, game_length, winner_id)
        expected_distance = (turn + 1) / game_length
        self.assertAlmostEqual(
            distance, expected_distance, places=2,
            msg=f"Expected minimal reward distance at game start, calculated as {(turn + 1) / game_length}"
        )

    def test_distance_from_win_loss_at_game_end(self):
        """Test that the win/loss distance is maximal at the end of the game."""
        winner_id = self.start_game.players[0].player_id
        turn = 99  # Last turn
        game_length = 100
        distance = RewardEncoder._distance_from_win_loss(self.start_game, turn, game_length, winner_id)
        expected_distance = (turn + 1) / game_length
        self.assertAlmostEqual(
            distance, expected_distance, places=2,
            msg=f"Expected maximal reward distance at game end, calculated as {(turn + 1) / game_length}"
        )
