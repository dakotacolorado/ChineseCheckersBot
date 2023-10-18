from unittest import TestCase

from torch import zeros_like, equal

from src.chinese_checkers.game.ChineseCheckersGame import ChineseCheckersGame
from src.chinese_checkers.model.IModel import IModel
from src.chinese_checkers.simulation.GameSimulation import GameSimulation
from src.chinese_checkers.model.CentroidModel import CentroidModel


class TestGameSimulation(TestCase):

    def test_centroid_model_simulation_convergence(self):
        """
        This is a stochastic test that tests if the centroid model converges to a winning state.
        It will occasionally fail, but should pass most of the time.
        """
        model_1 = CentroidModel()
        model_2 = CentroidModel()
        games = GameSimulation([model_1, model_2], print_period=19, print_coordinates=True).simulate_game()
        end_game = games[-1]
        end_game.print()
        print(f"Game Winner: {end_game.get_current_player()}")
        self.assertTrue(end_game.is_game_won())

    def test_tensor(self):
        # set up
        simulation = GameSimulation([IModel(), IModel()])
        game = ChineseCheckersGame.start_game()
        test_turns = 300
        simulation.games = [game for i in range(test_turns)]
        max_turns = 400
        simulation_tensor = simulation.tensor(400)

        for i in range(test_turns):
            self.assertTrue(equal(simulation_tensor[i], game.tensor()))

        for i in range(test_turns, max_turns):
            self.assertTrue(equal(simulation_tensor[i], zeros_like(game.tensor())))


