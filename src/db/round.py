from sqlalchemy import Boolean, Column, ForeignKey, Integer
from sqlalchemy.orm import relationship
from src.db import Base


class Round(Base):
    """Round table model."""

    __tablename__ = "rounds"

    id = Column(Integer, primary_key=True, autoincrement=True)
    game_id = Column(Integer, ForeignKey("games.id"), nullable=False)
    round_index = Column(Integer, nullable=False)
    honba = Column(Integer, nullable=False, default=0)
    is_agari = Column(Boolean, nullable=False, default=False)

    game = relationship("Game")
