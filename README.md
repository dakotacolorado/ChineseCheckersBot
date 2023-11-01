## ChineseCheckersGameEngine

The **ChineseCheckersGameEngine** is an advanced Python-based game engine designed for the simulation and analysis of Chinese Checkers gameplay. It offers a suite of tools ranging from game initialization and move validation to machine learning model integration and visual game simulation.

### Features

- **Game Initialization**: Kickstart your game with flexibility in board size and player count.
- **Move Validation**: Sophisticated logic to determine the valid moves for any player state.
- **Visual Rendering**: Rich visualization capabilities to depict the game board using matplotlib, showcasing up to three differently colored pieces.
- **Geometry and Linear Algebra**: Utilize geometry and linear algebra for advanced game mechanics in the `hexagram` class.
- **Machine Learning Model Integration**: Incorporate various game models, including the naive greedy implementation in `BootstrapModel.py` to enhance gameplay.
- **Game Simulation and Animation**: Simulate games between different models and visualize the gameplay through generated animations.
- **Data Cataloging**: Organize and manage game and simulation data efficiently.
- **Extensive Testing**: A thorough suite of unit tests ensuring game mechanics, model interactions, and simulations are executed seamlessly.

### How to Use

#### Starting a Game:

```python
from chinese_checkers.game.ChineseCheckersGame import ChineseCheckersGame

game = ChineseCheckersGame.start_game(number_of_players=2, board_size=4)
```

#### Simulating a Game Between Models:

```python
from chinese_checkers.model.BootstrapModel import BootstrapModel
from chinese_checkers.simulation.GameSimulation import GameSimulation

model_1 = BootstrapModel()
model_2 = BootstrapModel()
game_simulation = GameSimulation.simulate_game([model_1, model_2], "bootstrap_model", "1.0")
```


#### Printing the Game Board:

Use the `printer` class for visual representation of the game board.

```python
from chinese_checkers.geometry.Printer import Printer

printer = Printer(print_size=10, print_coordinates=False)
printer.print_grid(grid_points, green_points, red_points, yellow_points)
```

<p float="center">
  <img src="/images/game_screenshot.JPG?raw=true" width="500" />
</p>

#### Visualizing the Game through Animation:

After simulating a game, you can visualize the entire gameplay through an animation.

See [Sample Animation on YouTube](https://www.youtube.com/shorts/5G_hqv_NYUs).

```python
from chinese_checkers.simulation.GameSimulation import GameSimulation
from chinese_checkers.simulation.GameSimulationAnimation import GameSimulationAnimation

simulation_data: GameSimulation =  GameSimulation.simulate_game([model_1, model_2], "bootstrap_model", "1.0")
simulation_animation: GameSimulationAnimation = GameSimulationAnimation.from_simulation_data(simulation_data)
simulation_animation.display()
```

This feature helps in understanding the gameplay dynamics between different models and in visually validating the efficacy of a given model in the game.


For more hands-on examples, delve into the provided `ipynb` notebooks.


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

### Installation

Install directly from the git repository:
```
pip install -e git+https://github.com/dakotacolorado/ChineseCheckersGameEngine.git#egg=chinese_checkers
```

### Contribution

This package is a personal project and is near completion. It's currently not open for external development contributions.

### License

Please refer to the `LICENSE` file for licensing information.
