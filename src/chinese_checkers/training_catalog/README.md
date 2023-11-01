# Training Catalog Data Model

The `training_catalog` module contains the data models that support the training and evaluation of game-playing models for the Chinese Checker game. This module captures the state of the game, the players, the moves they make, and the models that are trained to play the game.

## Tables & Relationships

### 1. GamePlayingModel
This table stores details about the machine learning models that play the game.

- **Attributes**:
    - `id`: Primary Key
    - `name`: Name of the model.
    - `version`: Version of the model.

### 2. GameState
This table stores the state of the game at any given point.

- **Attributes**:
    - `id`: Primary Key
    - `state`: The state of the game board.

### 3. Move
This table logs each move made during a game.

- **Attributes**:
    - `id`: Primary Key
    - `game_state_id`: Foreign Key to `GameState`. The state of the game before the move.
    - `player_id`: Foreign Key to `Player`. The player making the move.
    - `move`: The move made by the player.

### 4. Player
This table stores details about the players in the game.

- **Attributes**:
    - `id`: Primary Key
    - `board_size`: The size of the game board.
    - `start_positions`: Starting positions of the player.
    - `target_positions`: Target positions for the player.

### 5. Simulation
This table logs details of each simulated game.

- **Attributes**:
    - `id`: Primary Key
    - `number_of_turns`: Total number of turns in the simulation.
    - `number_of_players`: Total number of players in the simulation.
    - `winning_player_id`: Foreign Key to `Player`. The player who won the simulation.
    - `board_size`: The size of the game board.

### 6. SimulationModelPlayerAssociation
This table is a many-to-many relationship table associating simulations, models, and players.

- **Attributes**:
    - `simulation_id`: Foreign Key to `Simulation`.
    - `game_playing_model_id`: Foreign Key to `GamePlayingModel`.
    - `player_id`: Foreign Key to `Player`.

## Relationships:

- A `GamePlayingModel` can participate in many simulations. 
- A `Simulation` can have many `Move`s and `Player`s associated with it.
- A `Player` can make many `Move`s in multiple `Simulation`s.
- A `GameState` is associated with many `Move`s but a `Move` is linked to only one `GameState`.

## Summary:

This data model captures every aspect of a simulated game, from the state of the game board, the players, the moves they make, to the machine learning models that play the game. It allows for comprehensive tracking of gameplay and provides a strong foundation for training and analyzing game-playing models for the Chinese Checker game.
