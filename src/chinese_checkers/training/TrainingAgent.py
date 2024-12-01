import random
import time

from tqdm import tqdm

from chinese_checkers.model import IModel
from chinese_checkers.model import TrainableModel
from chinese_checkers.simulation import GameSimulation, S3SimulationCatalog, SimulationMetadata
from chinese_checkers.training.TrainingValidator import TrainingValidator


class TrainingAgent:
    def __init__(
            self,
            model_to_train: TrainableModel,
            opponent: IModel,
            simulation_catalog: S3SimulationCatalog,
            max_turns=1000,
            board_size=4,
            name="training_simulation",
            version="v1.0.0",
            swap_players: bool = False,
            validator: TrainingValidator = None
    ):
        """
        Initializes the TrainingAgent.

        Args:
            model_to_train (TrainableModel): The model to train, implementing `TrainableModel`.
            simulation_catalog (S3SimulationCatalog): The simulation catalog for saving/loading games.
            opponent (IModel): The opponent model, implementing `IModel`.
            max_turns (int): Maximum number of turns in a game (default: 1000).
            board_size (int): Size of the game board (default: 4).
            name (str): Name of the simulation (default: "training_simulation").
            version (str): Version of the simulation (default: "v1.0.0").
        """

        if validator is None:
            validator = TrainingValidator(
                board_size=board_size,
                player_count=2,
                max_turns=max_turns,
                simulation_name=name,
                simulation_version=version,
            )
        self.validator = validator

        self.model_to_train = model_to_train
        self.simulation_catalog = simulation_catalog
        self.opponent = opponent
        self.swap_players = swap_players

        # Simulation configuration
        self.max_turns = max_turns
        self.board_size = board_size
        self.simulation_name = name
        self.simulation_version = version

    def bootstrap_training(self, bootstrap_game_count: int):
        """
        Plays bootstrap games with the baseline model and logs the win rates.

        Args:
            bootstrap_game_count (int): Number of games to simulate during bootstrap.
        """
        print(f"Starting bootstrap training with {bootstrap_game_count} games...")
        simulations = []
        player_0_wins = 0
        player_3_wins = 0
        player_none_wins = 0

        # Simulate bootstrap games
        with tqdm(total=bootstrap_game_count, desc="Simulating bootstrap games", unit="game",
                  dynamic_ncols=True) as sim_bar:
            for _ in range(bootstrap_game_count):
                try:
                    simulation = GameSimulation.simulate_game(
                        models=[self.opponent, self.opponent],
                        name=self.simulation_name,
                        version=self.simulation_version,
                        max_turns=self.max_turns,
                        board_size=self.board_size,
                    )
                    self.simulation_catalog.add_record(simulation)
                    simulations.append(simulation)

                    # Update win statistics
                    if simulation.metadata.winning_player == "0":
                        player_0_wins += 1
                    elif simulation.metadata.winning_player == "3":
                        player_3_wins += 1
                    else:
                        player_none_wins += 1

                    sim_bar.update(1)
                except Exception as e:
                    print(f"Warning: Failed to simulate game. Error: {e}")
                    continue

        self.simulation_catalog.flush()  # Ensure all records are saved

        # Log win rates
        total_games = player_0_wins + player_3_wins + player_none_wins
        print(f"Bootstrap training completed.")
        print(f"Total games simulated: {total_games}")
        print(f"Player 0 win rate: {(player_0_wins / total_games) * 100:.2f}%")
        print(f"Player 3 win rate: {(player_3_wins / total_games) * 100:.2f}%")
        print(f"Draw rate: {(player_none_wins / total_games) * 100:.2f}%")

    def train(
            self,
            training_period: int,
            training_size: int,
            training_batch_size: int,
            validation_size: int = 50,
    ):
        """
        Trains the model by simulating games and using the catalog for experience storage and sampling.

        Args:
            training_period (int): Number of games to simulate before training.
            training_size (int): Total number of games to simulate.
            training_batch_size (int): Batch size for training.
        """
        total_simulations = 0
        start_time = time.time()

        print(f"Training started with the following parameters:")
        print(f"- Training period: {training_period}")
        print(f"- Total training size: {training_size}")
        print(f"- Training batch size: {training_batch_size}")

        print(f"Loading simulations for {self.simulation_name}, version {self.simulation_version}...")

        simulations = []
        for winning_player in ["0", "3", None]:
            try:
                loaded_simulations = self.simulation_catalog.load_dataset(
                    metadata=SimulationMetadata(
                        player_count=2,
                        board_size=self.board_size,
                        max_game_length=self.max_turns,
                        winning_player=winning_player,
                        name=self.simulation_name,
                        version=self.simulation_version
                    )
                )
                print(f"Loaded {len(loaded_simulations)} simulations for winning_player={winning_player}.")
                simulations.extend(loaded_simulations)
            except Exception as e:
                print(f"Warning: Failed to load dataset for winning_player={winning_player}. Error: {e}")
        print(f"Loaded {len(simulations)} simulations for training.")

        while total_simulations < training_size:

            simulations_to_run = min(training_period, training_size - total_simulations)
            print(f"Simulating {simulations_to_run} games...")
            new_simulations = []
            with tqdm(total=simulations_to_run, desc="Simulating games", unit="game", dynamic_ncols=True) as sim_bar:
                for _ in range(simulations_to_run):

                    try:
                        simulation = self._simulate_game()
                        self.simulation_catalog.add_record(simulation)
                        simulations.append(simulation)
                        new_simulations.append(simulation)
                        total_simulations += 1
                        sim_bar.update(1)
                    except Exception as e:
                        print(f"Warning: Failed to simulate game. Error: {e}")
                        continue
            self.simulation_catalog.flush()
            winning_player_0 = len([s for s in new_simulations if s.metadata.winning_player == "0"])
            winning_player_3 = len([s for s in new_simulations if s.metadata.winning_player == "3"])
            winning_player_none = len([s for s in new_simulations if s.metadata.winning_player is None])
            print(f"Winnings: 0={winning_player_0}, 3={winning_player_3}, None={winning_player_none}")
            if self.swap_players:
                print(f"Player 3 win rate: {(winning_player_3 / len(new_simulations))*100:.2f}%")
            else:
                print(f"Player 0 win rate: {(winning_player_0 / len(new_simulations))*100:.2f}%")
            del new_simulations

            # Load random simulations and trai
            print(f"Training model with {training_batch_size} random simulations from {len(simulations)} total.")
            random_simulations = random.sample(simulations, min(training_batch_size, len(simulations)))
            print(f"Loaded {len(random_simulations)} random simulations for training.")

            train_start_time = time.time()
            self.model_to_train.train(random_simulations)
            train_duration = time.time() - train_start_time
            print(f"Training completed in {train_duration:.2f} seconds.")

            # Validate the model
            self.validator.validate(
                self.model_to_train,
                model_to_validate_player_id= 3 if self.swap_players else 0,
                validation_size=validation_size,
                save_animation=True,
                save_report_location=f"{self.simulation_name}_{self.simulation_version}_validation_report.csv",
            )

            # Save the model
            self.model_to_train.save(f"{self.simulation_name}-{self.simulation_version}.pt")

        total_duration = time.time() - start_time
        print(f"Training process completed in {total_duration:.2f} seconds.")
        print(f"Total simulations run: {total_simulations}")

    def _simulate_game(self):
        """
        Simulates a game between the model to train and the opponent.

        Returns:
            GameSimulation: The simulation result.
        """
        return GameSimulation.simulate_game(
            models=[self.opponent, self.model_to_train] if self.swap_players else [self.model_to_train, self.opponent],
            name=self.simulation_name,
            version=self.simulation_version,
            max_turns=self.max_turns,
            board_size=self.board_size,
        )


