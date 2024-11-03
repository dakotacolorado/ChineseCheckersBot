import unittest
from src.chinese_checkers.experience.RewardStrategyFactory import RewardStrategyFactory
from src.chinese_checkers.experience.DistanceToWinRewardStrategy import DistanceToWinRewardStrategy

class TestRewardStrategyFactory(unittest.TestCase):

    def test_create_distance_to_win_strategy(self):
        # Test creating the DistanceToWinRewardStrategy
        strategy = RewardStrategyFactory.create("distance_to_win")
        self.assertIsInstance(strategy, DistanceToWinRewardStrategy, "Factory should return an instance of DistanceToWinRewardStrategy")

    def test_create_invalid_strategy(self):
        # Test that creating an invalid strategy raises a ValueError
        with self.assertRaises(ValueError):
            RewardStrategyFactory.create("non_existent_strategy")

    def test_create_with_case_insensitivity(self):
        # Test that strategy creation is case-insensitive
        strategy = RewardStrategyFactory.create("DiStAnCe_To_Win")
        self.assertIsInstance(strategy, DistanceToWinRewardStrategy, "Factory should handle case-insensitive strategy names")