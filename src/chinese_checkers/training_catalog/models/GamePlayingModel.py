from sqlalchemy import Column, Integer, String
from ..Base import Base

TABLE_NAME = 'game_playing_models'


class GamePlayingModel(Base):
    __tablename__ = TABLE_NAME

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False)
    version = Column(String, nullable=False)

    def __repr__(self):
        return f"<GamePlayingModel(id={self.id}, name='{self.name}', version='{self.version}')>"

    class Meta:
        table_name = 'game_playing_models'
        id_column = f"{TABLE_NAME}.id"
