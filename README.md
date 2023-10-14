## ChineseCheckersGameEngine

The **ChineseCheckersGameEngine** is a Python-based game engine for Chinese Checkers. It provides a comprehensive set of tools for game initialization, move validation, and visually rendering the hexagram board state.

### Features

- **Game Initialization**: Easily begin a game with customizable board size and number of players.
- **Move Validation**: Determine the valid next moves for the current player.
- **Visual Rendering**: Print the game board using matplotlib with up to three different colored pieces.
- **Geometry and Linear Algebra**: Advanced game mechanics are achieved using geometry and linear algebra concepts in the `hexagram` class.
- **Extensive Testing**: A robust suite of unit tests to ensure game integrity.

### How to Use

#### Starting a Game:

```python
from chinese_checkers.game.ChineseCheckersGame import ChineseCheckersGame

game = ChineseCheckersGame.start_game(number_of_players=2, board_size=4)
```

#### Game Operations:

- `get_next_moves()`: Returns a list of possible moves for the current player.
- `apply_move(move: Move)`: Apply a specified move for the current player.
- `is_game_won()`: Checks if a player has won the game.

#### Printing the Game Board:

The `printer` class provides methods to visually render the game board.

```python
from chinese_checkers.geometry.Printer import Printer

printer = Printer(print_size=10, print_coordinates=False)
printer.print_grid(grid_points, green_points, red_points, yellow_points)
```

For more detailed examples, refer to the provided `ipynb` files.

### Dependencies

- **matplotlib**==3.8.0
- **pydash**==7.0.6

Install directly from the git repository:
```
pip install -e git+https://github.com/dakotacolorado/ChineseCheckersGameEngine.git#egg=chinese_checkers
```

### Contribution

This package is a personal project and is near completion. It's currently not open for external development contributions.

### License

Please refer to the `LICENSE` file for licensing information.
