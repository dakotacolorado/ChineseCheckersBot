from unittest import TestCase
from unittest.mock import MagicMock, patch

from src.chinese_checkers.game.Player import Player
from src.chinese_checkers.game.Position import Position
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

        simulation = GameSimulation.simulate_game(
            [model_1, model_2],
            "test_simulation",
            "0.0",
            print_period=19,
            show_coordinates=True,
        )
        winner = simulation.metadata.winning_player

        # Display game state and winner
        last_game = simulation.to_game_sequence()[-1]
        last_game.print()
        print(f"Game Winner: {winner}")

        # Assert game is won
        self.assertTrue(last_game.is_game_won())


