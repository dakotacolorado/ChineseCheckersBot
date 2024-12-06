import random
import time
from collections import defaultdict
from typing import List

from tqdm import tqdm

from chinese_checkers.model import IModel
from .DeepQModel import DeepQModel
from chinese_checkers.simulation import GameSimulation
from .GeneticSelection import GeneticSelector
from .Validator import Validator
from .ReplayBuffer import ReplayBuffer


class TrainingAgent:
    def __init__(
            self,
            model_to_train: DeepQModel,
            opponent: IModel,
            player_count: int = 2,
            max_turns=1000,
            board_size=4,
            name="training_simulation",
            version="v1.0.0",
            replay_buffer_size: int = 10000,
            train_win_bias: float = 2.0,
            max_hash_queue_size: int = 10,
            genetic_selection_validation_size: int = 20,
            genetic_selection_generation_size: int = 5,
    ):
        """
        Initializes the TrainingAgent.

        Args:
            model_to_train (TrainableModel): The model to train, implementing `TrainableModel`.
            opponent (IModel): The opponent model, implementing `IModel`.
            max_turns (int): Maximum number of turns in a game (default: 1000).
            board_size (int): Size of the game board (default: 4).
            name (str): Name of the simulation (default: "training_simulation").
            version (str): Version of the simulation (default: "v1.0.0").
            replay_buffer_size (int): Maximum size of the replay buffer (default: 10000).
        """
        # Simulation configuration
        self.model_to_train = model_to_train
        self.opponent = opponent
        self.player_count = player_count
        self.max_turns = max_turns
        self.board_size = board_size
        self.simulation_name = name
        self.simulation_version = version

        # Replay buffer for storing experiences
        self.replay_buffer = ReplayBuffer(
            buffer_size=replay_buffer_size,
            player_id="0",
            board_size=board_size,
            win_bias=train_win_bias,
            max_hash_queue_size=max_hash_queue_size,
        )

        # Validator for model evaluation
        self.validator = Validator(
            board_size=board_size,
            player_count=self.player_count,
            replay_buffer=self.replay_buffer,
            encoder=model_to_train.encoder,
            max_turns=max_turns,
            simulation_name=name,
            simulation_version=version,
            gamma=self.model_to_train.gamma,
        )
        self.genetic_selection_generation_size = genetic_selection_generation_size
        self.genetic_selector = GeneticSelector(
            baseline_model=self.opponent,
            player_count=player_count,
            validation_size=genetic_selection_validation_size,
            max_turns=max_turns,
            board_size=board_size,
        )

    def bootstrap_training(self, bootstrap_game_count: int) -> List[GameSimulation]:
        """
        Plays bootstrap games with the baseline model and logs the win rates.

        Args:
            bootstrap_game_count (int): Number of games to simulate during bootstrap.
        """
        print(f"Starting bootstrap reinforcement with {bootstrap_game_count} games...")
        win_counts = defaultdict(int)
        training_simulations = []

        with tqdm(total=bootstrap_game_count, desc="Simulating bootstrap games", unit="game",
                  dynamic_ncols=True) as sim_bar:
            for _ in range(bootstrap_game_count):
                try:
                    simulation = self._simulate_game(opponent_only=True)
                    training_simulations.append(simulation)
                    self.replay_buffer.add(simulation)

                    # Update win statistics
                    winner = simulation.metadata.winning_player
                    win_counts[winner] += 1

                    sim_bar.update(1)
                except Exception as e:
                    print(f"Warning: Failed to simulate game. Error: {e}")
                    continue

        total_games = sum(win_counts.values())
        print(f"Bootstrap reinforcement completed.")
        print(f"Total games simulated: {total_games}")

        # Print win rates for all players and None
        for player, wins in sorted(win_counts.items(), key=lambda x: (x[0] is None, x[0])):
            player_label = 'None' if player is None else player
            rate = (wins / total_games) * 100 if total_games > 0 else 0
            print(f"Player {player_label} win rate: {rate:.2f}%")

        self.replay_buffer.print_status()
        return training_simulations

    def train(
            self,
            training_period: int,
            training_size: int,
            training_batch_size: int,
            validation_size: int = 50,
            bootstrap_game_count: int = 20,
            bootstrap_update_frequency: int = 1000,
            train_before_simulating_first_batch: bool = True,
    ):
        """
        Trains the model by simulating games and using the replay buffer for experience storage and sampling.

        Args:
            training_period (int): Number of games to simulate before reinforcement.
            training_size (int): Total number of games to simulate.
            training_batch_size (int): Batch size for reinforcement.
            validation_size (int): Number of validation games to simulate.
        """
        total_simulations = 0
        start_time = time.time()

        print(f"Training started with the following parameters:")
        print(f"- Training period: {training_period}")
        print(f"- Total reinforcement size: {training_size}")
        print(f"- Training batch size: {training_batch_size}")

        if train_before_simulating_first_batch:
            training_sims = self.bootstrap_training(bootstrap_game_count)
            self._train_and_validate(training_batch_size, validation_size, training_sims)
        simulations_since_bootstrap = 0

        while total_simulations < training_size:
            training_sims = []
            simulations_to_run = min(training_period, training_size - total_simulations)
            # print(f"Simulating {simulations_to_run} games...")
            with tqdm(total=simulations_to_run, desc="Simulating games", unit="game", dynamic_ncols=True) as sim_bar:
                for _ in range(simulations_to_run):
                    try:
                        simulation = self._simulate_game()
                        training_sims.append(simulation)
                        self.replay_buffer.add(simulation)
                        total_simulations += 1
                        simulations_since_bootstrap += 1
                        sim_bar.update(1)
                    except Exception as e:
                        print(f"Warning: Failed to simulate game. Error: {e}")
                        continue

            # Train the model with experiences from the replay buffer
            self._train_and_validate(training_batch_size, validation_size, training_sims)
            # Save the model
            self.model_to_train.save(f"{self.simulation_name}-{self.simulation_version}.pt")
            # print(f"Simulations since last bootstrap: {simulations_since_bootstrap} / {bootstrap_update_frequency}")
            if simulations_since_bootstrap >= bootstrap_update_frequency:
                # print(f"Running bootstrap reinforcement with {bootstrap_game_count} games...")
                self.bootstrap_training(bootstrap_game_count)
                simulations_since_bootstrap = 0

        total_duration = time.time() - start_time
        print(f"Training process completed in {total_duration:.2f} seconds.")
        print(f"Total simulations run: {total_simulations}")
        self.replay_buffer.print_status()

    def _train_and_validate(self, training_batch_size: int, validation_size: int, training_simulations: List[GameSimulation]):
        self.replay_buffer.print_status()
        print(f"Training model with {training_batch_size} experiences from the replay buffer.")

        self.model_to_train = self.genetic_selector.evolve_model(
            self.model_to_train, self.replay_buffer, training_batch_size,
            generation_size=self.genetic_selection_generation_size
        )
        # experiences = self.replay_buffer.sample(training_batch_size)
        # self.model_to_train.train(experiences)

        self.validator.validate(
            self.model_to_train,
            generation=self.genetic_selector.generation,
            new_generation=self.genetic_selector.new_generation,
            model_to_validate_player_id=0,
            training_simulations=training_simulations,
            validation_size=validation_size,
            save_animation=True,
            save_report_location=f"{self.simulation_name}_{self.simulation_version}_validation_report.csv",
        )


    def _simulate_game(self, opponent_only=False):
        """
        Simulates a game between the model to train and the opponent.

        Args:
            opponent_only (bool): If True, simulate games with only the opponent models.

        Returns:
            GameSimulation: The simulation result.
        """
        models = []
        if opponent_only:
            models = [self.opponent for _ in range(self.player_count)]
        else:
            models = [self.model_to_train] + [self.opponent for i in range(self.player_count - 1)]

        return GameSimulation.simulate_game(
            models=models,
            name=self.simulation_name,
            version=self.simulation_version,
            max_turns=self.max_turns,
            board_size=self.board_size,
        )
