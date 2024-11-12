import torch
from chinese_checkers.game.ChineseCheckersGame import ChineseCheckersGame
from chinese_checkers.game.Move import Move
from chinese_checkers.model import IModel
from chinese_checkers.reinforcement.cnn.CnnNetworkState import CnnNetworkState
from chinese_checkers.reinforcement.cnn.CnnNetworkMove import CnnNetworkMove
from chinese_checkers.reinforcement.cnn.DqlCnnNetwork import DqlCnnNetwork
from chinese_checkers.reinforcement.cnn.CnnEncoderState import CnnEncoderState
from chinese_checkers.reinforcement.cnn.CnnEncoderMove import CnnEncoderMove


class DqlCnnModel(IModel):
    def __init__(self, model_path: str, board_size: int):
        """
        Initializes the CNN DQL-based model for move selection in Chinese Checkers.

        Args:
            model_path (str): Path to the trained model file containing state_cnn, move_cnn, and q_network.
            board_size (int): Size of the board for encoding purposes.
        """
        # Check if CUDA is available and set the device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Initialize the encoders
        self.state_encoder = CnnEncoderState(board_size)
        self.move_encoder = CnnEncoderMove(board_size)

        # Retrieve dimensions from the encoders
        state_dim, state_grid_h, state_grid_w = self.state_encoder.shape()
        move_dim, move_grid_h, move_grid_w = self.move_encoder.shape()

        # Initialize the CNN networks
        self.state_cnn = CnnNetworkState(state_dim, state_grid_h, state_grid_w, output_dim=64).to(self.device)
        self.move_cnn = CnnNetworkMove(move_dim, move_grid_h, move_grid_w, output_dim=64).to(self.device)

        # Initialize the Q-network
        self.q_network = DqlCnnNetwork(state_output_dim=64, move_output_dim=64).to(self.device)

        # Load the trained model weights
        self._load_model(model_path)

        # Set the networks to evaluation mode
        self.state_cnn.eval()
        self.move_cnn.eval()
        self.q_network.eval()

    def _load_model(self, path: str):
        """
        Loads the model weights from the specified path.

        Args:
            path (str): Path to the trained model file.
        """
        try:
            # Load the state dict containing all model weights
            checkpoint = torch.load(path, map_location=self.device)
            self.state_cnn.load_state_dict(checkpoint['state_cnn'])
            self.move_cnn.load_state_dict(checkpoint['move_cnn'])
            self.q_network.load_state_dict(checkpoint['q_network'])
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
        encoded_state = self.state_encoder.encode(game).unsqueeze(0).to(self.device)  # Shape: (1, channels, height, width)
        # Pass through state_cnn
        with torch.no_grad():
            state_encoding = self.state_cnn(encoded_state)  # Shape: (1, state_output_dim)

        best_move = None
        max_q_value = float('-inf')

        # Evaluate each possible move
        for move in game.get_next_moves():
            # Encode the move
            encoded_move = self.move_encoder.encode(move).unsqueeze(0).to(self.device)  # Shape: (1, channels, height, width)
            # Pass through move_cnn
            with torch.no_grad():
                move_encoding = self.move_cnn(encoded_move)  # Shape: (1, move_output_dim)
                # Pass through q_network
                q_value = self.q_network(state_encoding, move_encoding).item()

            # Track the move with the highest Q-value
            if q_value > max_q_value:
                max_q_value = q_value
                best_move = move

        return best_move
