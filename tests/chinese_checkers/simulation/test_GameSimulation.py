from unittest import TestCase

from src.chinese_checkers.simulation.GameSimulation import GameSimulation
from src.chinese_checkers.model.CentroidModel import CentroidModel


class TestGameSimulation(TestCase):

    def test_centroid_model_simulation_convergence(self):
        model_1 = CentroidModel()
        model_2 = CentroidModel()
        end_game = GameSimulation([model_1, model_2], print_period=19, print_coordinates=True).simulate_game()
        end_game.print()
        print(f"Game Winner: {end_game.get_current_player()}")
        self.assertTrue(end_game.is_game_won())


