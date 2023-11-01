from sqlalchemy import Column, Integer, ForeignKey
from ..Base import Base

TABLE_NAME = 'simulation_model_player_associations'


class SimulationModelPlayerAssociation(Base):
    __tablename__ = TABLE_NAME

    id = Column(Integer, primary_key=True, autoincrement=True)
    simulation_id = Column(Integer, ForeignKey('simulations.id'), nullable=False)
    game_playing_model_id = Column(Integer, ForeignKey('game_playing_models.id'), nullable=False)
    player_id = Column(Integer, ForeignKey('players.id'), nullable=False)

    def __repr__(self):
        return f"<SimulationModelPlayerAssociation(id={self.id}, simulation_id={self.simulation_id}, game_playing_model_id={self.game_playing_model_id}, player_id={self.player_id})>"

    class Meta:
        table_name = 'simulation_model_player_associations'
        id_column = f"{TABLE_NAME}.id"
