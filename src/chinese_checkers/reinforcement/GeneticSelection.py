import uuid
from typing import List

from chinese_checkers.game import ChineseCheckersGame
from chinese_checkers.model import IModel
from chinese_checkers.reinforcement import DeepQModel
from chinese_checkers.reinforcement.ReplayBuffer import ReplayBuffer
from chinese_checkers.simulation import GameSimulation


class GeneticSelector:
    def __init__(self, baseline_model: IModel, validation_size: int = 50, max_turns: int = 100, board_size: int = 4):
        self.baseline_model = baseline_model
        self.validation_size = validation_size
        self.max_turns = max_turns
        self.board_size = board_size
        self.generation = 0

    def evolve_model(self, model: DeepQModel, replay_buffer: ReplayBuffer, training_batch_size,  generation_size: int = 5) -> DeepQModel:
        """
        Evolves a model by selecting the best models from the population.
        """

        model_save_id = "legacy/" + str(uuid.uuid4()) + "_genetic_checkpoint.pt"
        model.save(model_save_id)


        children = [DeepQModel.load(model_save_id) for _ in range(generation_size)]
        for child in children:
            experiences = replay_buffer.sample(training_batch_size)
            child.train(experiences)
        population = [model] + children
        self.generation += 1
        return self.select_winner(population, self.validation_size)


    def select_winner(self, population: List[DeepQModel], simulation_count: int) -> DeepQModel:
        """
        Evolves a generation of models by selecting the best models from the population.
        """
        max_win_rate = 0
        min_draw_rate = 1
        best_model = population[0]
        best_model_index = -1
        print(f"Comparing {len(population)} models with {simulation_count} simulations against the baseline model...")
        for i, p in enumerate(population):
            simulations = [self._simulate_game(p) for i in range(simulation_count)]
            win_rate = sum([s.metadata.winning_player == "0" for s in simulations]) / simulation_count
            draw_rate = sum([s.metadata.winning_player is None for s in simulations]) / simulation_count
            print(f"Model {i} win rate: {win_rate}, draw rate: {draw_rate}")
            if min_draw_rate >= draw_rate:
                if max_win_rate < win_rate:
                    max_win_rate = win_rate
                    min_draw_rate = draw_rate
                    best_model = p
                    best_model_index = i
        print(f"Best model is model {best_model_index} with win rate: {max_win_rate}, draw rate: {min_draw_rate}")
        return best_model


    def _simulate_game(self, model_to_validate: DeepQModel) -> GameSimulation:
        """
        Simulates a game between the model to validate and the opponent.
        """
        return GameSimulation.simulate_game(
            models=[model_to_validate, self.baseline_model],
            name="genetic_selection",
            version="v1.0.0",
            max_turns=self.max_turns,
            board_size=self.board_size,
        )