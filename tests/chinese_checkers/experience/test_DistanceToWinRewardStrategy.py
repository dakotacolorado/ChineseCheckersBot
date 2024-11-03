import unittest
from src.chinese_checkers.experience.DistanceToWinRewardStrategy import DistanceToWinRewardStrategy
from src.chinese_checkers.game.ChineseCheckersGame import ChineseCheckersGame
from unittest.mock import MagicMock

class TestDistanceToWinRewardStrategy(unittest.TestCase):

    def setUp(self):
        self.strategy = DistanceToWinRewardStrategy()

    def test_calculate_reward_win(self):
        # Setup mock game sequence
        game_sequence = [MagicMock(spec=ChineseCheckersGame) for _ in range(10)]
        winning_player = 1
        turn_index = 5

        # Mock the current player to be the winning player at turn_index
        mock_player = MagicMock()
        mock_player.player_id = winning_player
        game_sequence[turn_index].get_current_player.return_value = mock_player

        # Calculate reward
        reward = self.strategy.calculate_reward(game_sequence, winning_player, turn_index)

        # Adjusted expected reward
        self.assertAlmostEqual(reward, 0.67, places=2)

    def test_calculate_reward_loss(self):
        # Setup mock game sequence
        game_sequence = [MagicMock(spec=ChineseCheckersGame) for _ in range(10)]
        winning_player = 2
        turn_index = 5

        # Mock the current player to be a losing player at turn_index
        mock_player = MagicMock()
        mock_player.player_id = 1
        game_sequence[turn_index].get_current_player.return_value = mock_player

        # Calculate reward
        reward = self.strategy.calculate_reward(game_sequence, winning_player, turn_index)

        # Adjusted expected reward
        self.assertAlmostEqual(reward, -0.67, places=2)
