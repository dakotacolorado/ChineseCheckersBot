from sqlalchemy import Column, Integer, ForeignKey, String
from ..Base import Base

TABLE_NAME = 'simulations'


class Simulation(Base):
    __tablename__ = TABLE_NAME

    id = Column(Integer, primary_key=True, autoincrement=True)
    number_of_turns = Column(Integer, nullable=False)
    number_of_players = Column(Integer, nullable=False)
    winning_player_id = Column(Integer, ForeignKey('players.id'), nullable=False)
    board_size = Column(Integer, nullable=False)

    def __repr__(self):
        return f"<Simulation(id={self.id}, number_of_turns={self.number_of_turns}, number_of_players={self.number_of_players}, winning_player_id={self.winning_player_id}, board_size={self.board_size})>"

    class Meta:
        table_name = 'simulations'
        id_column = f"{TABLE_NAME}.id"
