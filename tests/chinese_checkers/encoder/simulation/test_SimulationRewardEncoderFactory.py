import unittest
from src.chinese_checkers.encoder.simulation.SimulationEncoderFactory import SimulationRewardEncoderFactory
from src.chinese_checkers.encoder.simulation.DistanceToWinRewardStrategyEncoder import DistanceToWinRewardStrategyEncoder

class TestSimulationRewardEncoderFactory(unittest.TestCase):

    def setUp(self):
        self.factory = SimulationRewardEncoderFactory()

    def test_create_distance_to_win_reward_strategy(self):
        encoder = self.factory.create("distance_to_win_reward_strategy")
        self.assertIsInstance(encoder, DistanceToWinRewardStrategyEncoder,
                              "Expected factory to create an instance of DistanceToWinRewardStrategyEncoder")

    def test_invalid_encoder_name(self):
        with self.assertRaises(ValueError, msg="Expected ValueError for invalid encoder name"):
            self.factory.create("non_existent_strategy")
