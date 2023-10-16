from typing import List

from ..game import Player
from ..game.ChineseCheckersGame import ChineseCheckersGame
from ..game.Move import Move


class IModel:

    def __init__(self, game: ChineseCheckersGame):
        self.game = game

    def _chose_next_move(self, current_player: Player, other_players: List[Player], moves: List[Move]) -> Move:
        pass

    def make_move(self) -> ChineseCheckersGame:
        moves = self.game.get_next_moves()
        current_player = self.game.get_current_player()
        other_players = self.game.get_other_players()
        move = self._chose_next_move(current_player, other_players, moves)
        return self.game.apply_move(move)
