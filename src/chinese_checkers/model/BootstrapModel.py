import random
from typing import List
from random import choice

from .IModel import IModel
from ..game import Player, Move, ChineseCheckersGame
from ..geometry.Vector import Vector


class BootstrapModel(IModel):

    def _chose_next_move(self, game: ChineseCheckersGame) -> Move:
        """
        The BootstrapModel provides a foundational strategy for gameplay, serving as a
        bootstrapped or initial mechanism for move selection. This basic approach can
        serve as a starting point for generating initial game data or for complementing
        more advanced strategies.

        The method determines the best next move by evaluating the shortest distance
        to the player's target positions, prioritizing moves that lead pieces towards
        the center of the board. Defensive moves are not considered.

        The following are the key considerations:
        1. Priority is given to moving pieces out of the target positions of other players
           to prevent potential draws.
        2. Moves where a piece starts in the current player's target positions are generally skipped.
           However, a small element of randomness is introduced to occasionally allow such moves.
        3. If there exist priority moves (moves that get a piece out of other player's
           target positions), they are favored over regular moves. However, there's a slight
           randomness to occasionally bypass this preference.

        This approach, combined with the elements of randomness, ensures varied but
        strategic move selection, making the game dynamic and less predictable.

        Args:
            current_player: Current player.
            other_players: Other players in the game.
            moves: List of possible moves for current player.

        Returns:
            Next move to make.
        """
        moves: List[Move] = game.get_next_moves()
        current_player: Player = game.get_current_player()
        other_players: List[Player] = game.get_other_players()
        closest_distance = float('inf')
        best_moves = []

        # Order target positions by distance from the board center
        # Bias the model towards filling target positions closer to starting pieces
        nearest_target_positions = sorted(
            current_player.target_positions,
            key=lambda p: p.distance(Vector(0, 0)),
            reverse=True
        )[0]

        # List positions that are target positions for other players.
        # Ensure our model avoids these to prevent a draw.
        other_players_targets = [
            position
            for player in other_players
            for position in player.target_positions
        ]

        priority_positions = [
            position
            for position in current_player.positions
            if position in other_players_targets
        ]

        for move in moves:
            # Skip moves where the piece starts in one of the current player's target positions
            if move.position in current_player.target_positions and random.random() > 0.05:
                continue

            # Prefer priority moves if any exist
            if priority_positions and move.position not in priority_positions and random.random() > 0.1:
                continue

            new_position = move.apply()
            distance = new_position.distance(nearest_target_positions)

            # Find moves that bring a piece closest to the target position
            if distance < closest_distance:
                closest_distance = distance
                best_moves = [move]
            elif distance == closest_distance:
                best_moves.append(move)

        if not best_moves:
            raise Exception("No viable move found")

        return choice(best_moves)