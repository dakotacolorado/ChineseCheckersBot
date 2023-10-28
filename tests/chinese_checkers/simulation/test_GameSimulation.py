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

        simulation = GameSimulation([model_1, model_2], print_period=19, show_coordinates=True)
        winner = simulation.simulate_game()

        # Display game state and winner
        simulation.game.print()
        print(f"Game Winner: {winner}")

        # Assert game is won
        self.assertTrue(simulation.game.is_game_won())

    @patch("src.chinese_checkers.simulation.GameSimulation.ChineseCheckersGame.start_game")
    def test_export_simulation_data(self, mock_start_game: MagicMock):
        # set up
        random_models = [BootstrapModel(), BootstrapModel()]
        random_max_turns = 100
        random_board_size = 3
        random_winning_player_id = "1"
        random_players = [
            Player([Position(0, 0)], [Position(0, 1)], "1"),
            Player([Position(1, 1)], [Position(1, 1)], "2")
        ]

        random_name = "test"
        random_version = "1.0"

        mock_game_instance = MagicMock()
        mock_game_instance.players = random_players
        mock_game_instance.board.radius = random_board_size
        mock_game_instance.get_winner.return_value.player_id = random_winning_player_id

        mock_start_game.return_value = mock_game_instance

        # exercise
        simulation = GameSimulation(random_models, random_max_turns, random_board_size)
        simulation_data = simulation.export_simulation_data(random_name, random_version)

        # verify
        self.assertEqual(simulation_data.metadata.player_count, len(random_players))
        self.assertEqual(simulation_data.metadata.board_size, random_board_size)
        self.assertEqual(simulation_data.metadata.max_game_length, random_max_turns)
        self.assertEqual(simulation_data.metadata.name, random_name)
        self.assertEqual(simulation_data.metadata.version, random_version)
        self.assertEqual(simulation_data.metadata.winning_player, random_winning_player_id)

        self.assertEqual(simulation_data.positions.player_ids, [p.player_id for p in random_players])
        self.assertEqual(simulation_data.positions.player_start_positions, [p.positions for p in random_players])
        self.assertEqual(simulation_data.positions.player_target_positions, [p.target_positions for p in random_players])
        self.assertEqual(simulation_data.positions.historical_moves, simulation.move_history)


