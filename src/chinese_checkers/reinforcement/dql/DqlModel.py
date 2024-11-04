import torch
from chinese_checkers.game.ChineseCheckersGame import ChineseCheckersGame
from chinese_checkers.game.Move import Move
from chinese_checkers.reinforcement.encode import ExperienceEncoder
from chinese_checkers.model import IModel
from chinese_checkers.reinforcement.dql.DqlNetwork import DQLNetwork


class DqlModel(IModel):
    def __init__(self, q_network_path: str, encoder: ExperienceEncoder, state_dim: int, action_dim: int):
        """
        Initializes the DQL-based model for move selection in Chinese Checkers.

        Args:
            q_network_path (str): Path to the trained Q-network model.
            encoder (ExperienceEncoder): Encoder for game states and moves.
            state_dim (int): Dimension of the encoded game state.
            action_dim (int): Dimension of the encoded action.
        """
        # Set input_dim to match the concatenated state and action dimensions
        input_dim = state_dim + action_dim
        self.q_network = DQLNetwork(input_dim)

        # Load the trained model weights with weights_only=True for safer loading
        self._load_model(q_network_path)

        # Set the network to evaluation mode
        self.q_network.eval()

        self.encoder = encoder

    def _load_model(self, path: str):
        """
        Safely loads the model weights from a specified path.

        Args:
            path (str): Path to the trained model file.
        """
        try:
            self.q_network.load_state_dict(torch.load(path, map_location=torch.device('cpu'), weights_only=True))
            print(f"Model loaded successfully from {path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {path}: {e}")

    def _chose_next_move(self, game: ChineseCheckersGame) -> Move:
        """
        Selects the next best move by evaluating Q-values for each possible action.

        Args:
            game (ChineseCheckersGame): The current game state.

        Returns:
            Move: The move with the highest Q-value.
        """
        # Encode the current game state
        encoded_state = torch.FloatTensor(self.encoder.game_encoder.encode(game)).unsqueeze(0)

        best_move = None
        max_q_value = float('-inf')

        # Evaluate each possible move
        for move in game.get_next_moves():
            encoded_action = torch.FloatTensor(self.encoder.move_encoder.encode(move)).unsqueeze(0)

            # Concatenate state and action for input to Q-network
            state_action_combined = torch.cat((encoded_state, encoded_action), dim=1)

            # Calculate Q-value for the state-action pair
            with torch.no_grad():
                q_value = self.q_network(state_action_combined).item()

            # Track the move with the highest Q-value
            if q_value > max_q_value:
                max_q_value = q_value
                best_move = move

        return best_move
