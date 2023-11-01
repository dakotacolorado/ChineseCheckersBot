from sqlalchemy import Column, Integer, String, ForeignKey
from ..Base import Base

TABLE_NAME = 'players'


class Player(Base):
    __tablename__ = TABLE_NAME

    id = Column(Integer, primary_key=True, autoincrement=True)
    board_size = Column(Integer, nullable=False)
    start_positions = Column(String, nullable=False)  # Consider using other types if needed
    target_positions = Column(String, nullable=False)  # Consider using other types if needed

    def __repr__(self):
        return f"<Player(id={self.id}, board_size={self.board_size}, start_positions='{self.start_positions}', target_positions='{self.target_positions}')>"

    class Meta:
        table_name = 'players'
        id_column = f"{TABLE_NAME}.id"
