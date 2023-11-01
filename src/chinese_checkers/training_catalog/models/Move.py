from sqlalchemy import Column, Integer, ForeignKey
from ..Base import Base

TABLE_NAME = 'moves'


class Move(Base):
    __tablename__ = TABLE_NAME

    id = Column(Integer, primary_key=True, autoincrement=True)
    turn = Column(Integer, nullable=False)
    from_position = Column(Integer, nullable=False)
    to_position = Column(Integer, nullable=False)
    simulation_id = Column(Integer, ForeignKey('simulations.id'), nullable=False)

    def __repr__(self):
        return f"<Move(id={self.id}, turn={self.turn}, from_position={self.from_position}, to_position={self.to_position}, simulation_id={self.simulation_id})>"

    class Meta:
        table_name = 'moves'
        id_column = f"{TABLE_NAME}.id"
