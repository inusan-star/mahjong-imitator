from sqlalchemy import Column, Float, ForeignKey, Integer
from sqlalchemy.orm import relationship

from src.db import Base


class GamePlayer(Base):
    """GamePlayer table model."""

    __tablename__ = "game_players"

    id = Column(Integer, primary_key=True, autoincrement=True)
    game_id = Column(Integer, ForeignKey("games.id"), nullable=False)
    player_id = Column(Integer, ForeignKey("players.id"), nullable=False)
    seat_index = Column(Integer, nullable=False)
    dan = Column(Integer, nullable=False)
    rate = Column(Float, nullable=False)
    score = Column(Integer, nullable=False)
    rank = Column(Integer, nullable=False)

    game = relationship("Game")
    player = relationship("Player")
