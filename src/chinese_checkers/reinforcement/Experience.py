from dataclasses import dataclass

from chinese_checkers.game import ChineseCheckersGame, Move


@dataclass(frozen=True)
class Experience:
    winning_player: str
    board_size: int
    game: ChineseCheckersGame
    move: Move
    turn: int
    total_turns: int

    def __hash__(self):
        return self.move.__hash__()