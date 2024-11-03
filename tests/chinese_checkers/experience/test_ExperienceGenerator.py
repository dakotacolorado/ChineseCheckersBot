import unittest
from unittest.mock import MagicMock, patch
from src.chinese_checkers.experience.ExperienceGenerator import ExperienceGenerator
from src.chinese_checkers.experience.DistanceToWinRewardStrategy import DistanceToWinRewardStrategy
from src.chinese_checkers.simulation.GameSimulation import GameSimulation
from src.chinese_checkers.simulation.SimulationData import SimulationData
from src.chinese_checkers.simulation.SimulationMetadata import SimulationMetadata
from src.chinese_checkers.game.ChineseCheckersGame import ChineseCheckersGame
from src.chinese_checkers.game.Move import Move
from src.chinese_checkers.experience.Experience import Experience


class TestExperienceGenerator(unittest.TestCase):

    def setUp(self):
        reward_strategy = DistanceToWinRewardStrategy()
        self.generator = ExperienceGenerator(reward_strategy)

    def test_generate_experiences_from_simulation(self):
        # Setup mock game simulation metadata and data
        metadata = SimulationMetadata(2, 4, 10, "1", "test", "v1.0.0")
        # Bypass frozen restriction to set `winning_player`
        object.__setattr__(metadata, 'winning_player', "1")

        moves = [MagicMock(spec=Move) for _ in range(5)]
        data = SimulationData(["1", "2"], [], [], moves)
        simulation = GameSimulation(metadata, data)

        # Define a mock `game_sequence` with 6 states, marking the last state as a game-ending state
        game_sequence = [MagicMock(spec=ChineseCheckersGame) for _ in range(6)]
        for i, game_state in enumerate(game_sequence):
            game_state.is_game_won = MagicMock(return_value=(i == 5))  # Only last state is terminal

        # Patch `_to_game_sequence` to return the mock `game_sequence`
        with patch.object(GameSimulation, '_to_game_sequence', return_value=game_sequence):
            # Generate experiences
            experiences = self.generator.generate_experiences_from_simulation(simulation)

            # Check if experiences are correctly generated
            self.assertEqual(len(experiences), 5)
            for idx, experience in enumerate(experiences):
                self.assertIsInstance(experience, Experience)
                self.assertIsInstance(experience.state, ChineseCheckersGame)
                self.assertIsInstance(experience.action, Move)
                self.assertIsInstance(experience.reward, float)
                self.assertIsInstance(experience.next_state, ChineseCheckersGame)
                self.assertIsInstance(experience.done, bool)
                # Verify terminal state only in the last experience
                if idx == len(experiences) - 1:
                    self.assertTrue(experience.done)
                else:
                    self.assertFalse(experience.done)
