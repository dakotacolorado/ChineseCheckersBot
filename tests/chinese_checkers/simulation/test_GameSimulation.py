from unittest import TestCase

from torch import zeros_like, equal

from src.chinese_checkers.game.ChineseCheckersGame import ChineseCheckersGame
from src.chinese_checkers.model.IModel import IModel
from src.chinese_checkers.simulation.GameSimulation import GameSimulation
from src.chinese_checkers.model.BootstrapModel import BootstrapModel


class TestGameSimulation(TestCase):

    def test_bootstrap_model_simulation_convergence(self):
        """
        A stochastic test to check if the centroid model converges to a winning state.
        This test can occasionally fail but should pass most of the time.
        """
        model_1 = BootstrapModel()
        model_2 = BootstrapModel()

        simulation = GameSimulation([model_1, model_2], print_period=19, print_coordinates=True)
        winner = simulation.simulate_game()

        # Display game state and winner
        simulation.game.print()
        print(f"Game Winner: {winner}")

        # Assert game is won
        self.assertTrue(simulation.game.is_game_won())

b    # def test_get_game_tensor(self):
    #     """
    #     Test if tensor representation of the game is constructed correctly.
    #     """
    #     # Set up
    #     simulation = GameSimulation([IModel(), IModel()])
    #     game = ChineseCheckersGame.start_game()
    #     test_turns = 300
    #
    #     # Simulate `test_turns` number of games
    #     simulation.games = [game for _ in range(test_turns)]
    #     max_turns = 400
    #     simulation_tensor = simulation.get_game_tensor(max_turns)
    #
    #     # Check if the first `test_turns` tensors match the game tensor
    #     for i in range(test_turns):
    #         self.assertTrue(equal(simulation_tensor[i], game.tensor()))
    #
    #     # Check if the remaining tensors are zero tensors
    #     for i in range(test_turns, max_turns):
    #         self.assertTrue(equal(simulation_tensor[i], zeros_like(game.tensor())))
