## ChineseCheckersGameEngine

The **ChineseCheckersGameEngine** is a Python-based game engine designed for the simulation of Chinese Checkers games between game models. 

### Features

- **Game Setup**: Begin games with varied board sizes and player numbers.
- **Move Options**: Identify the possible moves for a player.
- **Board Visualization**: Display the game board using matplotlib.
- **Model Integration**: Use [`IModel`](src/chinese_checker_game/model/IModel.py) to integrate game models, including a basic greedy strategy in [`BootstrapModel.py`](src/chinese_checker_game/model/BootstrapModel.py).
- **Game Playback**: Run simulations between models and view gameplay animations.
- **Data Management**: Handle game and simulation data for model training.
- **Testing Suite**: Unit tests to verify game functions and model integrations.

### How to Use

#### Starting a Game:

```python
from chinese_checkers.game.ChineseCheckersGame import ChineseCheckersGame

game = ChineseCheckersGame.start_game(number_of_players=2, board_size=4)
```

### Printing a Game: 
```python
game.print(plot_size = 10, show_coordinates = True)
```

<p float="center">
  <img src="/images/game_screenshot.JPG?raw=true" width="500" />
</p>

#### Simulating a Game Between Models:

```python
from chinese_checkers.model.BootstrapModel import BootstrapModel
from chinese_checkers.simulation.GameSimulation import GameSimulation

model_1 = BootstrapModel()
model_2 = BootstrapModel()
game_simulation = GameSimulation.simulate_game([model_1, model_2], "bootstrap_model", "1.0")
```

#### Visualizing the Game through Animation:

After simulating a game, you can visualize the entire gameplay through an animation.

See [Sample Animation on YouTube](https://www.youtube.com/shorts/5G_hqv_NYUs).

```python
# show in jupyter
game_simulation.display()

# save to a file
game_simulation.display(file_path="game_simulation_1.mp4")
```

This feature helps in understanding the gameplay dynamics between different models and in visually validating the efficacy of a given model in the game.

For more hands-on examples, delve into the provided `ipynb` notebooks.

### Installation

Install directly from the git repository:
```
pip install -e git+https://github.com/dakotacolorado/ChineseCheckersGameEngine.git#egg=chinese_checkers
```

### Contribution

This package is a personal project and is near completion. It's currently not open for external development contributions.

### License

Please refer to the `LICENSE` file for licensing information.
