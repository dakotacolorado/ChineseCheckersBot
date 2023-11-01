from sqlalchemy import Column, Integer, String
from ..Base import Base

TABLE_NAME = 'game_states'


class GameState(Base):
    __tablename__ = TABLE_NAME

    id = Column(Integer, primary_key=True, autoincrement=True)
    board_hash = Column(String, nullable=False)
    player_score = Column(Integer, nullable=False)

    def __repr__(self):
        return f"<GameState(id={self.id}, board_hash='{self.board_hash}', player_score={self.player_score})>"

    class Meta:
        table_name = 'game_states'
        id_column = f"{TABLE_NAME}.id"
