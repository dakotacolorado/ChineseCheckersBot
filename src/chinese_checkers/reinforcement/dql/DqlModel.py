import torch
from chinese_checkers.game.ChineseCheckersGame import ChineseCheckersGame
from chinese_checkers.game.Move import Move
from chinese_checkers.model import IModel
from chinese_checkers.reinforcement.dql.DqlNetwork import DQLNetwork
from chinese_checkers.reinforcement.encode.SpatialBoardMetricsEncoder import SpatialBoardMetricsEncoder
from chinese_checkers.reinforcement.encode.SpatialMoveMetricsEncoder import SpatialMoveMetricsEncoder


class DqlModel(IModel):
    def __init__(self, q_network_path: str, state_dim: int, action_dim: int, board_size: int):
        """
        Initializes the DQL-based model for move selection in Chinese Checkers.

        Args:
            q_network_path (str): Path to the trained Q-network model.
            state_dim (int): Dimension of the encoded game state.
            action_dim (int): Dimension of the encoded action.
            board_size (int): Size of the board for encoding purposes.
        """
        # Initialize the Q-network and load the trained model weights
        input_dim = state_dim + action_dim
        self.q_network = DQLNetwork(input_dim)
        self._load_model(q_network_path)
        self.q_network.eval()  # Set the network to evaluation mode

        # Initialize encoders for game states and moves
        self.board_encoder = SpatialBoardMetricsEncoder(board_size)
        self.move_encoder = SpatialMoveMetricsEncoder()

    def _load_model(self, path: str):
        """
        Loads the model weights from the specified path.

        Args:
            path (str): Path to the trained model file.
        """
        try:
            self.q_network.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
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
        encoded_state = self.board_encoder.encode(game).unsqueeze(0)  # Shape: (1, state_dim)

        best_move = None
        max_q_value = float('-inf')

        # Evaluate each possible move
        for move in game.get_next_moves():
            # Encode the move
            encoded_action = self.move_encoder.encode(move, game).unsqueeze(0)  # Shape: (1, action_dim)

            # Concatenate state and action for input to Q-network
            state_action_combined = torch.cat((encoded_state, encoded_action), dim=1)  # Shape: (1, input_dim)

            # Calculate Q-value for the state-action pair
            with torch.no_grad():
                q_value = self.q_network(state_action_combined).item()

            # Track the move with the highest Q-value
            if q_value > max_q_value:
                max_q_value = q_value
                best_move = move

        return best_move
